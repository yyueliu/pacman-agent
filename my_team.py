# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import traceback
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.root_pos = None
        self.border = []
        self.dist_home = {}

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # Cells on our side right next to the middle line (red side is x < width // 2)
        walls = game_state.get_walls()
        if self.red:
            border_x = walls.width // 2 - 1
        else:
            border_x = walls.width // 2
        self.border = [(border_x, y) for y in range(walls.height) if not walls[border_x][y]]

        # For every free cell, how many steps it takes to get back to our side
        self.dist_home = {}
        for cell in walls.as_list(False):
            self.dist_home[cell] = min(self.get_maze_distance(cell, b) for b in self.border)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        try:
            self.root_pos = game_state.get_agent_position(self.index)
            self.decide_mode(game_state)

            # You can profile your evaluation time by uncommenting these lines
            # start = time.time()
            values = [self.evaluate(game_state, a) for a in actions]
            # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]

            return random.choice(best_actions)
        except Exception:
            # If something goes wrong, print the error and still return a legal move
            traceback.print_exc()
            moves = [a for a in actions if a != Directions.STOP]
            if len(moves) == 0:
                return Directions.STOP
            return random.choice(moves)

    def decide_mode(self, game_state):
        """Called once per turn, before evaluating actions, to pick what to go for."""
        pass

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        successor = self.get_successor(game_state, action)
        value = self.evaluate_state(successor)
        # Avoid standing still and turning back
        if action == Directions.STOP:
            value -= 100
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            value -= 2
        return value

    def evaluate_state(self, game_state):
        """Scores a state: weighted sum of its features."""
        my_pos = game_state.get_agent_position(self.index)
        # Being back at the start from far away means we were eaten
        if my_pos == self.start and self.get_maze_distance(self.root_pos, self.start) > 5:
            return -100000
        features = self.get_features(game_state)
        weights = self.get_weights(game_state)
        return features * weights

    def get_features(self, game_state):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        features['successor_score'] = self.get_score(game_state)
        return features

    def get_weights(self, game_state):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    CARRY_LIMIT = 5  # how much food to carry before heading home

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.returning = False
        self.in_danger = False
        self.entry = None

    def dangerous_ghosts(self, game_state):
        """Positions of the enemy ghosts we can see that are able to eat us."""
        ghosts = []
        for i in self.get_opponents(game_state):
            state = game_state.get_agent_state(i)
            pos = state.get_position()
            if pos is not None and not state.is_pacman and state.scared_timer <= 2:
                ghosts.append(pos)
        return ghosts

    def decide_mode(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        carrying = my_state.num_carrying
        food_left = len(self.get_food(game_state).as_list())
        moves_left = game_state.data.timeleft // 4  # moves left for each agent

        self.in_danger = False
        for ghost in self.dangerous_ghosts(game_state):
            if self.get_maze_distance(my_pos, ghost) <= 5:
                self.in_danger = True

        # Go home with enough food, when food is almost gone, when time is
        # running out, or when a ghost is close and we would lose what we carry
        self.returning = (carrying >= self.CARRY_LIMIT
                          or food_left <= 2
                          or (carrying > 0 and moves_left <= self.dist_home[my_pos] + 5)
                          or (carrying > 0 and self.in_danger))

        # If a ghost guards the way in, head for the crossing farthest from it
        if my_state.is_pacman:
            self.entry = None
        elif self.in_danger:
            walls = game_state.get_walls()
            enemy_x = self.border[0][0] + (1 if self.red else -1)
            entries = [(enemy_x, y) for (x, y) in self.border if not walls[enemy_x][y]]
            ghosts = self.dangerous_ghosts(game_state)
            self.entry = max(entries, key=lambda e: min(self.get_maze_distance(e, g) for g in ghosts))

    def get_features(self, game_state):
        features = util.Counter()
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()

        features['score'] = self.get_score(game_state)
        features['food_left'] = len(food_list)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            features['dist_food'] = min(self.get_maze_distance(my_pos, f) for f in food_list)
        features['dist_home'] = self.dist_home[my_pos]

        # Stay away from ghosts that can eat us
        if my_state.is_pacman:
            for ghost in self.dangerous_ghosts(game_state):
                d = self.get_maze_distance(my_pos, ghost)
                if d <= 1:
                    features['ghost_adjacent'] += 1
                if d <= 5:
                    features['ghost_near'] += 6 - d

        # With a ghost nearby, a power capsule is worth going for
        if self.in_danger:
            capsules = self.get_capsules(game_state)
            features['capsules_left'] = len(capsules)
            if len(capsules) > 0:
                features['dist_capsule'] = min(self.get_maze_distance(my_pos, c) for c in capsules)
        if self.entry is not None:
            features['dist_entry'] = self.get_maze_distance(my_pos, self.entry)
        return features

    def get_weights(self, game_state):
        # The mode is fixed for the whole turn, so all actions use the same weights
        if self.returning:
            return {'score': 100, 'food_left': -20, 'dist_home': -10,
                    'ghost_adjacent': -1000, 'ghost_near': -20, 'capsules_left': -200,
                    'dist_entry': -5}
        return {'score': 100, 'food_left': -100, 'dist_food': -2,
                'ghost_adjacent': -1000, 'ghost_near': -10,
                'capsules_left': -200, 'dist_capsule': -3,
                'dist_entry': -5}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.food_target = None
        self.target = None

    def patrol_point(self, game_state):
        """Border cell closest overall to the food we defend."""
        food = self.get_food_you_are_defending(game_state).as_list()
        if len(food) == 0:
            return self.border[len(self.border) // 2]
        return min(self.border, key=lambda b: sum(self.get_maze_distance(b, f) for f in food))

    def decide_mode(self, game_state):
        my_pos = game_state.get_agent_position(self.index)

        # Food that vanished since our last turn shows where an unseen invader is
        previous = self.get_previous_observation()
        if previous is not None:
            before = self.get_food_you_are_defending(previous).as_list()
            now = self.get_food_you_are_defending(game_state).as_list()
            eaten = [f for f in before if f not in now]
            if len(eaten) > 0:
                self.food_target = min(eaten, key=lambda f: self.get_maze_distance(my_pos, f))
        if self.food_target == my_pos:
            self.food_target = None

        if self.food_target is not None:
            self.target = self.food_target
        else:
            self.target = self.patrol_point(game_state)

    def get_features(self, game_state):
        features = util.Counter()

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            d = min(dists)
            if my_state.scared_timer > 0:
                # While scared we cannot eat them, so keep 2 steps away
                features['scared_gap'] = abs(d - 2)
            else:
                features['invader_distance'] = d
        else:
            features['target_distance'] = self.get_maze_distance(my_pos, self.target)

        return features

    def get_weights(self, game_state):
        return {'on_defense': 100, 'num_invaders': -1000, 'invader_distance': -10,
                'scared_gap': -10, 'target_distance': -5}
