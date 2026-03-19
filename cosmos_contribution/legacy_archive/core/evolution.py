import random
from deap import base, creator, tools, algorithms
from utils.config import Config
import time
import threading

# Define Fitness (Maximize user satisfaction/task success)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

class EvolutionEngine:
    def __init__(self, memory_system):
        self.memory = memory_system
        self.toolbox = base.Toolbox()
        self._setup_deap()
        self.population = self.toolbox.population(n=Config.POPULATION_SIZE)
        self.running = False

    def _setup_deap(self):
        # Genome: [Temperature (0-1), RAG_K (1-10), Pruning_Threshold (0-1)]
        self.toolbox.register("attr_temp", random.random)
        self.toolbox.register("attr_k", random.randint, 1, 10)
        self.toolbox.register("attr_prune", random.uniform, 0.1, 0.9)
        
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                             (self.toolbox.attr_temp, self.toolbox.attr_k, self.toolbox.attr_prune), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _evaluate_individual(self, individual):
        # Placeholder: Real system would track user feedback score per session config
        # Here we simulate random fitness for structural validity
        return (random.random(),)

    def evolve_hyperparameters(self):
        """Run one generation of evolution."""
        offspring = algorithms.varAnd(self.population, self.toolbox, cxpb=0.5, mutpb=0.2)
        fits = self.toolbox.map(self.toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        self.population = self.toolbox.select(offspring, k=len(self.population))
        
        best_ind = tools.selBest(self.population, 1)[0]
        return {
            "temperature": abs(best_ind[0]), 
            "rag_k": max(1, int(best_ind[1])),
            "prune_thresh": abs(best_ind[2])
        }

    def memory_dreaming(self):
        """Unsupervised recombination of memory nodes during idle time."""
        print("cosmos is dreaming...")
        params = self.evolve_hyperparameters()
        # Simulate linking random nodes in memory graph
        if self.memory.graph.number_of_nodes() > 5:
            nodes = list(self.memory.graph.nodes())
            node_a, node_b = random.sample(nodes, 2)
            if not self.memory.graph.has_edge(node_a, node_b):
                # In refined version: use LLM to check if they *should* be linked
                self.memory.graph.add_edge(node_a, node_b, weight=0.5, type="dream_generated")
                print(f"Dreamt connection between Node {node_a} and {node_b}")

    def start_background_loop(self):
        """Start the background evolution thread."""
        self.running = True
        t = threading.Thread(target=self._loop)
        t.daemon = True
        t.start()

    def _loop(self):
        while self.running:
            time.sleep(Config.EVOLUTION_INTERVAL)
            self.memory_dreaming()
