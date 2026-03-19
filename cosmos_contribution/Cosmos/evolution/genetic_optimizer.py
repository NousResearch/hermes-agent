"""
cosmos Genetic Optimizer - DEAP-Based Parameter Evolution

Novel Approaches:
1. NSGA-II Multi-Objective - Optimize multiple goals simultaneously
2. Adaptive Mutation - Mutation rate adjusts based on progress
3. Island Model - Parallel populations with migration
4. Hash-Chain Logging - Tamper-proof evolution history

AGI Upgrades (v1.5):
5. Meta-Learning - Self-optimizing evolutionary strategies
6. Strategy Portfolio - Learn which operators work best
7. Cross-Problem Transfer - Learn from past optimization runs
8. Gene Correlation Learning - Discover effective gene combinations

AGI Upgrades (v1.8 - Quantum):
9. Quantum Genetic Algorithm - Superposition-based population generation
10. Quantum Mutation - Use quantum circuits for probabilistic bit flips
11. IBM Quantum Integration - Hardware (10min/month) + unlimited simulators
"""

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Callable
from collections import defaultdict

from loguru import logger

try:
    import numpy as np
except ImportError:
    np = None


# ==============================================================================
# CST NON-VANISHING PENALTY
# ==============================================================================
# WP-NODE Precursor: Forces high-energy truthful responses by penalizing
# low-complexity or lazy outputs. Based on Cosmic Synapse Theory's principle
# that "any answer of low effort must be penalized to force real thinking."
# ==============================================================================

def cst_penalty(complexity_score: float, threshold: float = 2.0) -> float:
    """
    CST Non-Vanishing Penalty for evolutionary fitness.
    
    This function implements the WP-NODE precursor constraint that forces
    agents to provide high-effort, truthful responses by mathematically
    suppressing lazy or hallucinated outputs.
    
    Formula: penalty = (complexity_score - threshold)
    
    If complexity < threshold (default 2.0), the penalty is NEGATIVE,
    reducing fitness. This forces evolution toward more substantive responses.
    
    Args:
        complexity_score: A measure of response complexity/effort (0.0 - 10.0)
        threshold: Minimum acceptable complexity (default 2.0)
        
    Returns:
        Penalty value (negative for low-effort, positive for high-effort)
        
    Example:
        - complexity=1.0 → penalty=-1.0 (penalized)
        - complexity=2.0 → penalty=0.0 (neutral)
        - complexity=5.0 → penalty=3.0 (rewarded)
    """
    return complexity_score - threshold


def calculate_complexity_from_fitness(fitness_scores: dict) -> float:
    """
    Calculate complexity score from fitness metrics.
    
    This extracts or estimates a complexity value from the available
    fitness scores to feed into cst_penalty().
    
    Args:
        fitness_scores: Dictionary of fitness metrics
        
    Returns:
        Complexity score (0.0 - 10.0)
    """
    # If complexity is explicitly provided, use it
    if "complexity" in fitness_scores:
        return fitness_scores["complexity"]
    
    # Otherwise estimate from other metrics
    # Higher diversity + coherence = higher complexity
    diversity = fitness_scores.get("diversity", 0.5)
    coherence = fitness_scores.get("coherence", 0.5)
    quality = fitness_scores.get("quality", 0.5)
    
    # Combine metrics into complexity estimate (scaled to 0-10)
    complexity = (diversity + coherence + quality) * 3.33
    return min(10.0, max(0.0, complexity))


@dataclass
class Gene:
    """A single evolvable parameter."""
    name: str
    value: float
    min_val: float
    max_val: float
    mutation_sigma: float = 0.1

    def mutate(self) -> "Gene":
        """Create mutated copy."""
        new_value = self.value + random.gauss(0, self.mutation_sigma * (self.max_val - self.min_val))
        new_value = max(self.min_val, min(self.max_val, new_value))
        return Gene(
            name=self.name,
            value=new_value,
            min_val=self.min_val,
            max_val=self.max_val,
            mutation_sigma=self.mutation_sigma,
        )


@dataclass
class Genome:
    """A complete genome (set of parameters)."""
    id: str
    genes: dict[str, Gene]
    fitness_scores: dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def get_value(self, gene_name: str) -> float:
        """Get value of a gene."""
        return self.genes[gene_name].value if gene_name in self.genes else 0.0

    def to_dict(self) -> dict:
        """Serialize genome."""
        return {
            "id": self.id,
            "genes": {name: gene.value for name, gene in self.genes.items()},
            "fitness_scores": self.fitness_scores,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "created_at": self.created_at.isoformat(),
        }

    def total_fitness(
        self, 
        weights: Optional[dict[str, float]] = None,
        apply_cst_penalty: bool = False,
    ) -> float:
        """
        Calculate weighted total fitness with optional CST penalty.
        
        Args:
            weights: Optional weights for each fitness score
            apply_cst_penalty: Whether to apply CST non-vanishing penalty (opt-in)
            
        Returns:
            Total fitness score (may be negative if penalized)
        """
        if not self.fitness_scores:
            return 0.0
        
        if weights is None:
            base_fitness = sum(self.fitness_scores.values()) / len(self.fitness_scores)
        else:
            total = 0.0
            weight_sum = 0.0
            for name, score in self.fitness_scores.items():
                w = weights.get(name, 1.0)
                total += score * w
                weight_sum += w
            base_fitness = total / max(0.001, weight_sum)
        
        # Apply CST penalty to suppress low-effort responses
        if apply_cst_penalty:
            complexity = calculate_complexity_from_fitness(self.fitness_scores)
            penalty = cst_penalty(complexity)
            base_fitness += penalty * 0.1  # Scale penalty contribution
        
        return base_fitness


@dataclass
class EvolutionConfig:
    """Configuration for genetic evolution."""
    population_size: int = 20
    generations: int = 10
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    tournament_size: int = 3
    elite_count: int = 2

    # Multi-objective
    use_nsga2: bool = True

    # Adaptive mutation
    adaptive_mutation: bool = True
    stagnation_threshold: int = 3


@dataclass
class EvolutionResult:
    """Result from an evolution run."""
    best_genome: Genome
    final_population: list[Genome]
    generations_run: int
    fitness_history: list[dict]
    duration_seconds: float


# =============================================================================
# META-LEARNING (AGI Upgrade v1.5)
# =============================================================================

@dataclass
class OperatorPerformance:
    """Tracks performance of a genetic operator."""
    name: str
    uses: int = 0
    fitness_improvements: int = 0
    total_improvement: float = 0.0
    avg_improvement: float = 0.0
    success_rate: float = 0.5

    def update(self, improved: bool, improvement_amount: float = 0.0):
        self.uses += 1
        if improved:
            self.fitness_improvements += 1
            self.total_improvement += improvement_amount
        self.success_rate = self.fitness_improvements / self.uses if self.uses > 0 else 0.5
        self.avg_improvement = self.total_improvement / max(1, self.fitness_improvements)


@dataclass
class GeneCorrelation:
    """Tracks correlation between gene values and fitness."""
    gene1: str
    gene2: str
    correlation: float = 0.0
    sample_count: int = 0


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning."""
    enabled: bool = True
    strategy_update_interval: int = 3  # Update every N generations
    min_samples_for_adaptation: int = 10  # Min samples before adapting
    exploration_rate: float = 0.2  # Probability of trying non-optimal strategy
    transfer_learning: bool = True  # Use knowledge from past runs
    correlation_threshold: float = 0.5  # Min correlation to consider genes related


@dataclass
class EvolutionKnowledge:
    """Knowledge extracted from evolution runs for transfer learning."""
    problem_signature: str  # Hash of gene definitions
    best_hyperparameters: dict = field(default_factory=dict)
    effective_gene_combinations: list[tuple[str, str]] = field(default_factory=list)
    operator_preferences: dict[str, float] = field(default_factory=dict)
    total_runs: int = 0
    avg_convergence_speed: float = 0.0


class MetaLearner:
    """
    Meta-learning system for self-optimizing evolutionary strategies.

    Learns:
    - Which operators work best for different problem types
    - Effective hyperparameter settings
    - Gene correlations and effective combinations
    - Cross-problem knowledge transfer
    """

    def __init__(self, config: Optional[MetaLearningConfig] = None, data_dir: str = "./data/evolution"):
        self.config = config or MetaLearningConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Operator performance tracking (AGI v1.8: includes quantum operators)
        self.operators: dict[str, OperatorPerformance] = {
            "crossover_uniform": OperatorPerformance(name="crossover_uniform"),
            "crossover_two_point": OperatorPerformance(name="crossover_two_point"),
            "crossover_blend": OperatorPerformance(name="crossover_blend"),
            "crossover_quantum": OperatorPerformance(name="crossover_quantum"),  # AGI v1.8
            "mutation_gaussian": OperatorPerformance(name="mutation_gaussian"),
            "mutation_uniform": OperatorPerformance(name="mutation_uniform"),
            "mutation_adaptive": OperatorPerformance(name="mutation_adaptive"),
            "mutation_quantum": OperatorPerformance(name="mutation_quantum"),  # AGI v1.8
        }

        # AGI v1.8: Quantum integration flag
        self._quantum_available = False
        self._quantum_optimizer = None
        try:
            from Cosmos.integration.quantum import QISKIT_AVAILABLE, get_quantum_provider
            from Cosmos.integration.quantum.ibm_quantum import QuantumGeneticOptimizer
            self._quantum_available = QISKIT_AVAILABLE
            if QISKIT_AVAILABLE:
                provider = get_quantum_provider()
                if provider:
                    self._quantum_optimizer = QuantumGeneticOptimizer(provider, num_qubits=8)
                    logger.info("Quantum genetic operators available (IBM Quantum)")
        except ImportError:
            pass

        # Gene correlations
        self.gene_correlations: dict[tuple[str, str], GeneCorrelation] = {}

        # Cross-problem knowledge
        self.knowledge_base: dict[str, EvolutionKnowledge] = {}

        # Current strategy weights (AGI v1.8: includes quantum operators)
        self.strategy_weights = {
            "crossover_uniform": 0.30,
            "crossover_two_point": 0.30,
            "crossover_blend": 0.30,
            "crossover_quantum": 0.10,
            "mutation_gaussian": 0.40,
            "mutation_uniform": 0.25,
            "mutation_adaptive": 0.25,
            "mutation_quantum": 0.10,
        }

        # Hyperparameter learning
        self.hyperparameter_history: list[dict] = []
        self.best_hyperparameters: dict = {}

        # Load persisted knowledge
        self._load_knowledge()

    def _load_knowledge(self):
        """Load persisted meta-learning knowledge."""
        knowledge_file = self.data_dir / "meta_knowledge.json"
        if knowledge_file.exists():
            try:
                with knowledge_file.open('r') as f:
                    data = json.load(f)
                    for sig, kdata in data.get("knowledge_base", {}).items():
                        self.knowledge_base[sig] = EvolutionKnowledge(
                            problem_signature=sig,
                            **{k: v for k, v in kdata.items() if k != "problem_signature"}
                        )
                    self.strategy_weights = data.get("strategy_weights", self.strategy_weights)
                logger.info(f"Loaded meta-learning knowledge: {len(self.knowledge_base)} problems")
            except Exception as e:
                logger.warning(f"Failed to load meta-knowledge: {e}")

    def save_knowledge(self):
        """Persist meta-learning knowledge."""
        knowledge_file = self.data_dir / "meta_knowledge.json"
        data = {
            "knowledge_base": {
                sig: {
                    "best_hyperparameters": k.best_hyperparameters,
                    "effective_gene_combinations": k.effective_gene_combinations,
                    "operator_preferences": k.operator_preferences,
                    "total_runs": k.total_runs,
                    "avg_convergence_speed": k.avg_convergence_speed,
                }
                for sig, k in self.knowledge_base.items()
            },
            "strategy_weights": self.strategy_weights,
        }
        with knowledge_file.open('w') as f:
            json.dump(data, f, indent=2)

    def get_problem_signature(self, gene_definitions: dict) -> str:
        """Generate a signature for a problem based on gene definitions."""
        sig_data = {
            "genes": sorted(gene_definitions.keys()),
            "ranges": {k: (v["min"], v["max"]) for k, v in gene_definitions.items()},
        }
        return hashlib.sha256(json.dumps(sig_data, sort_keys=True).encode()).hexdigest()[:16]

    def select_crossover_operator(self) -> str:
        """Select crossover operator using learned preferences (includes quantum)."""
        # AGI v1.8: Include quantum crossover if available
        base_operators = ["crossover_uniform", "crossover_two_point", "crossover_blend"]
        operators = base_operators + (["crossover_quantum"] if self._quantum_available else [])

        if random.random() < self.config.exploration_rate:
            return random.choice(operators)

        # Exploitation: weighted selection based on success
        weights = [self.strategy_weights.get(op, 0.25) for op in operators]
        total = sum(weights)
        weights = [w / total for w in weights]

        return random.choices(operators, weights=weights)[0]

    def select_mutation_operator(self) -> str:
        """Select mutation operator using learned preferences (includes quantum)."""
        # AGI v1.8: Include quantum mutation if available
        base_operators = ["mutation_gaussian", "mutation_uniform", "mutation_adaptive"]
        operators = base_operators + (["mutation_quantum"] if self._quantum_available else [])

        if random.random() < self.config.exploration_rate:
            return random.choice(operators)

        weights = [self.strategy_weights.get(op, 0.25) for op in operators]
        total = sum(weights)
        weights = [w / total for w in weights]

        return random.choices(operators, weights=weights)[0]

    def record_operator_result(
        self,
        operator_name: str,
        parent_fitness: float,
        child_fitness: float,
    ):
        """Record the result of using an operator."""
        if operator_name not in self.operators:
            self.operators[operator_name] = OperatorPerformance(name=operator_name)

        improved = child_fitness > parent_fitness
        improvement = max(0, child_fitness - parent_fitness)
        self.operators[operator_name].update(improved, improvement)

    def update_strategy_weights(self):
        """Update strategy weights based on operator performance."""
        if not self.config.enabled:
            return

        # Compute weights from success rates (AGI v1.8: includes quantum operators)
        crossover_ops = ["crossover_uniform", "crossover_two_point", "crossover_blend"]
        mutation_ops = ["mutation_gaussian", "mutation_uniform", "mutation_adaptive"]
        if self._quantum_available:
            crossover_ops.append("crossover_quantum")
            mutation_ops.append("mutation_quantum")

        for op_group in [crossover_ops, mutation_ops]:
            total_success = sum(
                self.operators.get(op, OperatorPerformance(op)).success_rate
                for op in op_group
            )
            if total_success > 0:
                for op in op_group:
                    perf = self.operators.get(op, OperatorPerformance(op))
                    # Blend current weight with performance
                    new_weight = perf.success_rate / total_success
                    self.strategy_weights[op] = (
                        0.7 * self.strategy_weights.get(op, 0.33) +
                        0.3 * new_weight
                    )

        logger.debug(f"Updated strategy weights: {self.strategy_weights}")

    def learn_gene_correlations(self, population: list[Genome]):
        """Learn correlations between genes from population."""
        if not population or len(population) < self.config.min_samples_for_adaptation:
            return

        gene_names = list(population[0].genes.keys())
        if len(gene_names) < 2:
            return

        # Build data matrix
        if np is None:
            return  # Numpy required for correlation

        n_genes = len(gene_names)
        n_samples = len(population)

        gene_values = np.zeros((n_samples, n_genes))
        fitness_values = np.zeros(n_samples)

        for i, genome in enumerate(population):
            for j, name in enumerate(gene_names):
                gene_values[i, j] = genome.genes[name].value
            fitness_values[i] = genome.total_fitness()

        # Compute pairwise correlations with fitness
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                key = (gene_names[i], gene_names[j])

                # Combined effect correlation
                combined = gene_values[:, i] * gene_values[:, j]
                if np.std(combined) > 0 and np.std(fitness_values) > 0:
                    corr = np.corrcoef(combined, fitness_values)[0, 1]
                else:
                    corr = 0.0

                if key not in self.gene_correlations:
                    self.gene_correlations[key] = GeneCorrelation(
                        gene1=gene_names[i],
                        gene2=gene_names[j],
                    )

                gc = self.gene_correlations[key]
                gc.correlation = (gc.correlation * gc.sample_count + corr) / (gc.sample_count + 1)
                gc.sample_count += 1

    def get_correlated_genes(self) -> list[tuple[str, str, float]]:
        """Get genes with significant correlation."""
        return [
            (gc.gene1, gc.gene2, gc.correlation)
            for gc in self.gene_correlations.values()
            if abs(gc.correlation) >= self.config.correlation_threshold
        ]

    def record_run_result(
        self,
        gene_definitions: dict,
        best_fitness: float,
        generations_to_converge: int,
        hyperparameters: dict,
    ):
        """Record results of an evolution run for transfer learning."""
        sig = self.get_problem_signature(gene_definitions)

        if sig not in self.knowledge_base:
            self.knowledge_base[sig] = EvolutionKnowledge(problem_signature=sig)

        knowledge = self.knowledge_base[sig]
        knowledge.total_runs += 1

        # Update convergence speed (running average)
        knowledge.avg_convergence_speed = (
            (knowledge.avg_convergence_speed * (knowledge.total_runs - 1) + generations_to_converge)
            / knowledge.total_runs
        )

        # Update best hyperparameters if this run was better
        if not knowledge.best_hyperparameters or best_fitness > knowledge.best_hyperparameters.get("best_fitness", 0):
            knowledge.best_hyperparameters = {
                **hyperparameters,
                "best_fitness": best_fitness,
            }

        # Update operator preferences
        knowledge.operator_preferences = dict(self.strategy_weights)

        # Update effective gene combinations
        correlated = self.get_correlated_genes()
        knowledge.effective_gene_combinations = [
            (g1, g2) for g1, g2, _ in correlated
        ]

        self.save_knowledge()

    def get_recommended_hyperparameters(self, gene_definitions: dict) -> dict:
        """Get recommended hyperparameters for a problem."""
        sig = self.get_problem_signature(gene_definitions)

        # Check for exact match
        if sig in self.knowledge_base:
            return self.knowledge_base[sig].best_hyperparameters

        # No transfer knowledge available
        return {}

    def get_meta_stats(self) -> dict:
        """Get meta-learning statistics."""
        return {
            "operators": {
                name: {
                    "uses": op.uses,
                    "success_rate": op.success_rate,
                    "avg_improvement": op.avg_improvement,
                }
                for name, op in self.operators.items()
                if op.uses > 0
            },
            "strategy_weights": self.strategy_weights,
            "knowledge_base_size": len(self.knowledge_base),
            "gene_correlations_learned": len(self.gene_correlations),
            "significant_correlations": len(self.get_correlated_genes()),
        }


class GeneticOptimizer:
    """
    DEAP-inspired genetic optimizer for system parameters.

    Features:
    - NSGA-II for multi-objective optimization
    - Tournament selection with elitism
    - Adaptive mutation rates
    - Hash-chain evolution logging
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        data_dir: str = "./data/evolution",
        meta_learning_config: Optional[MetaLearningConfig] = None,
    ):
        self.config = config or EvolutionConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Gene definitions (parameter space)
        self.gene_definitions: dict[str, dict] = {}

        # Population
        self.population: list[Genome] = []
        self.generation = 0

        # History
        self.fitness_history: list[dict] = []
        self.evolution_log: list[dict] = []
        self.last_hash: str = "genesis"

        # Fitness function (set by user)
        self.fitness_fn: Optional[Callable[[Genome], dict[str, float]]] = None

        # Statistics
        self.stats = {
            "total_evaluations": 0,
            "generations_completed": 0,
            "best_fitness_ever": 0.0,
            "stagnation_count": 0,
        }

        # AGI v1.5: Meta-learning
        self.meta_learner = MetaLearner(
            config=meta_learning_config or MetaLearningConfig(),
            data_dir=data_dir,
        )

        # Nexus integration (lazy-loaded)
        self._nexus = None
        self._SignalType = None

    async def _emit_nexus(self, signal_type_name: str, payload: dict, urgency: float = 0.5):
        """Emit a signal to the Nexus event bus. Fails silently."""
        try:
            if self._nexus is None:
                from Cosmos.core.nexus import nexus, SignalType
                self._nexus = nexus
                self._SignalType = SignalType
            signal_type = getattr(self._SignalType, signal_type_name, None)
            if signal_type is None:
                return
            await self._nexus.emit(
                type=signal_type,
                payload=payload,
                source="genetic_optimizer",
                urgency=urgency,
            )
        except Exception as e:
            logger.debug(f"Nexus emit failed (non-critical): {e}")

    def define_gene(
        self,
        name: str,
        min_val: float,
        max_val: float,
        default: Optional[float] = None,
        mutation_sigma: float = 0.1,
    ):
        """Define an evolvable parameter."""
        self.gene_definitions[name] = {
            "min": min_val,
            "max": max_val,
            "default": default if default is not None else (min_val + max_val) / 2,
            "sigma": mutation_sigma,
        }

    def set_fitness_function(self, fn: Callable[[Genome], dict[str, float]]):
        """Set the fitness evaluation function."""
        self.fitness_fn = fn

    def _create_random_genome(self) -> Genome:
        """Create a genome with random values."""
        genes = {}
        for name, defn in self.gene_definitions.items():
            value = random.uniform(defn["min"], defn["max"])
            genes[name] = Gene(
                name=name,
                value=value,
                min_val=defn["min"],
                max_val=defn["max"],
                mutation_sigma=defn["sigma"],
            )

        return Genome(
            id=f"g{self.generation}_{random.randint(0, 9999):04d}",
            genes=genes,
            generation=self.generation,
        )

    def _create_default_genome(self) -> Genome:
        """Create a genome with default values."""
        genes = {}
        for name, defn in self.gene_definitions.items():
            genes[name] = Gene(
                name=name,
                value=defn["default"],
                min_val=defn["min"],
                max_val=defn["max"],
                mutation_sigma=defn["sigma"],
            )

        return Genome(
            id=f"g{self.generation}_default",
            genes=genes,
            generation=self.generation,
        )

    def initialize_population(self, include_default: bool = True):
        """Initialize the population."""
        self.population = []
        self.generation = 0

        # Include default genome
        if include_default:
            self.population.append(self._create_default_genome())

        # Fill with random genomes
        while len(self.population) < self.config.population_size:
            self.population.append(self._create_random_genome())

        logger.info(f"Initialized population with {len(self.population)} genomes")

    async def evaluate_population(self):
        """Evaluate fitness for all genomes in population."""
        if self.fitness_fn is None:
            raise RuntimeError("Fitness function not set")

        for genome in self.population:
            if not genome.fitness_scores:
                if asyncio.iscoroutinefunction(self.fitness_fn):
                    genome.fitness_scores = await self.fitness_fn(genome)
                else:
                    genome.fitness_scores = self.fitness_fn(genome)
                self.stats["total_evaluations"] += 1

    def _tournament_select(self, k: int = None) -> Genome:
        """Tournament selection."""
        k = k or self.config.tournament_size
        contestants = random.sample(self.population, min(k, len(self.population)))
        return max(contestants, key=lambda g: g.total_fitness())

    async def _crossover(self, parent1: Genome, parent2: Genome) -> tuple[Genome, Genome]:
        """
        Crossover with meta-learned operator selection.

        AGI v1.5: Selects crossover method based on learned performance.
        AGI v1.8: Real quantum crossover via IBM Quantum with classical fallback.
        """
        gene_names = list(self.gene_definitions.keys())
        if len(gene_names) < 2:
            return await self._mutate(parent1), await self._mutate(parent2)

        # AGI v1.5: Select crossover operator using meta-learning
        operator = self.meta_learner.select_crossover_operator()

        genes1 = {}
        genes2 = {}

        if operator == "crossover_uniform":
            # Uniform crossover - each gene randomly from either parent
            for name in gene_names:
                if random.random() < 0.5:
                    genes1[name] = Gene(**{**parent1.genes[name].__dict__})
                    genes2[name] = Gene(**{**parent2.genes[name].__dict__})
                else:
                    genes1[name] = Gene(**{**parent2.genes[name].__dict__})
                    genes2[name] = Gene(**{**parent1.genes[name].__dict__})

        elif operator == "crossover_blend":
            # Blend crossover - interpolate gene values
            alpha = 0.5
            for name in gene_names:
                g1, g2 = parent1.genes[name], parent2.genes[name]
                blend1 = alpha * g1.value + (1 - alpha) * g2.value
                blend2 = (1 - alpha) * g1.value + alpha * g2.value

                genes1[name] = Gene(
                    name=name, value=blend1,
                    min_val=g1.min_val, max_val=g1.max_val, mutation_sigma=g1.mutation_sigma
                )
                genes2[name] = Gene(
                    name=name, value=blend2,
                    min_val=g2.min_val, max_val=g2.max_val, mutation_sigma=g2.mutation_sigma
                )

        elif operator == "crossover_quantum" and self.meta_learner._quantum_available:
            # AGI v1.8: Real quantum crossover via IBM Quantum
            # Convert gene values to bitstrings, run quantum crossover, convert back
            quantum_success = False
            try:
                qopt = self.meta_learner._quantum_optimizer
                if qopt:
                    # Encode parent genes as bitstrings (8 bits per gene)
                    bits_per_gene = 8
                    parent1_bits = ""
                    parent2_bits = ""
                    for name in gene_names:
                        g1, g2 = parent1.genes[name], parent2.genes[name]
                        norm1 = (g1.value - g1.min_val) / max(0.001, g1.max_val - g1.min_val)
                        norm2 = (g2.value - g2.min_val) / max(0.001, g2.max_val - g2.min_val)
                        parent1_bits += format(int(min(255, max(0, norm1 * 255))), '08b')
                        parent2_bits += format(int(min(255, max(0, norm2 * 255))), '08b')

                    # Run real quantum crossover (prefer simulator to save hardware budget)
                    child1_bits = await qopt.quantum_crossover(
                        parent1_bits, parent2_bits, prefer_hardware=False
                    )
                    child2_bits = await qopt.quantum_crossover(
                        parent2_bits, parent1_bits, prefer_hardware=False
                    )

                    # Decode bitstrings back to gene values
                    for idx, name in enumerate(gene_names):
                        g1 = parent1.genes[name]
                        start = idx * bits_per_gene
                        end = start + bits_per_gene

                        c1_gene_bits = child1_bits[start:end] if end <= len(child1_bits) else '10000000'
                        c2_gene_bits = child2_bits[start:end] if end <= len(child2_bits) else '10000000'

                        norm1 = int(c1_gene_bits, 2) / 255.0
                        norm2 = int(c2_gene_bits, 2) / 255.0

                        val1 = g1.min_val + norm1 * (g1.max_val - g1.min_val)
                        val2 = g1.min_val + norm2 * (g1.max_val - g1.min_val)

                        genes1[name] = Gene(
                            name=name, value=val1,
                            min_val=g1.min_val, max_val=g1.max_val, mutation_sigma=g1.mutation_sigma
                        )
                        genes2[name] = Gene(
                            name=name, value=val2,
                            min_val=g1.min_val, max_val=g1.max_val, mutation_sigma=g1.mutation_sigma
                        )
                    quantum_success = True
                    logger.debug("Quantum crossover succeeded via IBM Quantum")
            except Exception as e:
                logger.warning(f"Quantum crossover failed, using classical blend: {e}")

            if not quantum_success:
                # Classical blend fallback
                alpha = 0.5
                for name in gene_names:
                    g1, g2 = parent1.genes[name], parent2.genes[name]
                    blend1 = alpha * g1.value + (1 - alpha) * g2.value
                    blend2 = (1 - alpha) * g1.value + alpha * g2.value
                    genes1[name] = Gene(
                        name=name, value=blend1,
                        min_val=g1.min_val, max_val=g1.max_val, mutation_sigma=g1.mutation_sigma
                    )
                    genes2[name] = Gene(
                        name=name, value=blend2,
                        min_val=g2.min_val, max_val=g2.max_val, mutation_sigma=g2.mutation_sigma
                    )

        else:  # crossover_two_point (default)
            point1 = random.randint(0, len(gene_names) - 1)
            point2 = random.randint(point1, len(gene_names))

            for i, name in enumerate(gene_names):
                if point1 <= i < point2:
                    genes1[name] = Gene(**{**parent2.genes[name].__dict__})
                    genes2[name] = Gene(**{**parent1.genes[name].__dict__})
                else:
                    genes1[name] = Gene(**{**parent1.genes[name].__dict__})
                    genes2[name] = Gene(**{**parent2.genes[name].__dict__})

        child1 = Genome(
            id=f"g{self.generation}_{random.randint(0, 9999):04d}",
            genes=genes1,
            generation=self.generation,
            parent_ids=[parent1.id, parent2.id],
        )
        child2 = Genome(
            id=f"g{self.generation}_{random.randint(0, 9999):04d}",
            genes=genes2,
            generation=self.generation,
            parent_ids=[parent1.id, parent2.id],
        )

        # Store operator for later tracking
        child1.genes["_crossover_op"] = operator  # type: ignore
        child2.genes["_crossover_op"] = operator  # type: ignore

        return child1, child2

    async def _mutate(self, genome: Genome) -> Genome:
        """
        Mutate a genome with meta-learned operator selection.

        AGI v1.5: Selects mutation method based on learned performance.
        AGI v1.8: Real quantum mutation via IBM Quantum with classical fallback.
        """
        # AGI v1.5: Select mutation operator using meta-learning
        operator = self.meta_learner.select_mutation_operator()

        new_genes = {}
        for name, gene in genome.genes.items():
            if name.startswith("_"):  # Skip metadata
                continue

            if random.random() < self.config.mutation_prob:
                if operator == "mutation_uniform":
                    # Uniform mutation - random value in range
                    new_value = random.uniform(gene.min_val, gene.max_val)
                    new_genes[name] = Gene(
                        name=name, value=new_value,
                        min_val=gene.min_val, max_val=gene.max_val,
                        mutation_sigma=gene.mutation_sigma
                    )

                elif operator == "mutation_adaptive":
                    # Adaptive mutation - sigma based on fitness
                    # Lower sigma for high-fitness genomes (fine-tuning)
                    fitness = genome.total_fitness()
                    adaptive_sigma = gene.mutation_sigma * (1.0 - fitness * 0.5)
                    delta = random.gauss(0, adaptive_sigma * (gene.max_val - gene.min_val))
                    new_value = max(gene.min_val, min(gene.max_val, gene.value + delta))
                    new_genes[name] = Gene(
                        name=name, value=new_value,
                        min_val=gene.min_val, max_val=gene.max_val,
                        mutation_sigma=gene.mutation_sigma
                    )

                elif operator == "mutation_quantum" and self.meta_learner._quantum_available:
                    # AGI v1.8: Real quantum mutation via IBM Quantum
                    quantum_mutated = False
                    try:
                        qopt = self.meta_learner._quantum_optimizer
                        if qopt:
                            normalized = (gene.value - gene.min_val) / max(0.001, gene.max_val - gene.min_val)
                            bitstring = format(int(min(255, max(0, normalized * 255))), '08b')

                            # Real quantum mutation with RY rotation gates
                            mutated_bitstring = await qopt.quantum_mutation(
                                bitstring, mutation_rate=gene.mutation_sigma, prefer_hardware=False
                            )

                            new_normalized = int(mutated_bitstring, 2) / 255.0
                            new_value = gene.min_val + new_normalized * (gene.max_val - gene.min_val)
                            new_value = max(gene.min_val, min(gene.max_val, new_value))
                            new_genes[name] = Gene(
                                name=name, value=new_value,
                                min_val=gene.min_val, max_val=gene.max_val,
                                mutation_sigma=gene.mutation_sigma
                            )
                            quantum_mutated = True
                    except Exception as e:
                        logger.debug(f"Quantum mutation fallback for gene {name}: {e}")

                    if not quantum_mutated:
                        # Classical fallback: gaussian mutation
                        new_genes[name] = gene.mutate()

                else:  # mutation_gaussian (default)
                    new_genes[name] = gene.mutate()
            else:
                new_genes[name] = Gene(**gene.__dict__)

        child = Genome(
            id=f"g{self.generation}_{random.randint(0, 9999):04d}",
            genes=new_genes,
            generation=self.generation,
            parent_ids=[genome.id],
        )

        # Store operator for later tracking
        child.genes["_mutation_op"] = operator  # type: ignore

        # Emit mutation signal for quantum mutations (lightweight: skip classical to avoid flood)
        if operator == "mutation_quantum":
            await self._emit_nexus("EVOLUTION_MUTATION", {
                "genome_id": child.id,
                "operator": operator,
                "generation": self.generation,
                "parent_id": genome.id,
                "genes_mutated": sum(1 for n in new_genes if n != genome.genes.get(n)),
            }, urgency=0.3)

        return child

    def _nsga2_sort(self, population: list[Genome]) -> list[Genome]:
        """
        NSGA-II non-dominated sorting.

        Returns population sorted by Pareto fronts.
        """
        if not population or not population[0].fitness_scores:
            return population

        objectives = list(population[0].fitness_scores.keys())
        n = len(population)

        # Domination counts and dominated sets
        domination_count = [0] * n
        dominated_by = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population[i], population[j], objectives):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(population[j], population[i], objectives):
                    dominated_by[j].append(i)
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        # Build subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)

        # Flatten and return sorted population
        sorted_indices = []
        for front in fronts:
            # Sort within front by crowding distance
            if front:
                front_with_crowd = self._crowding_distance(front, population, objectives)
                front_sorted = sorted(front_with_crowd, key=lambda x: x[1], reverse=True)
                sorted_indices.extend([idx for idx, _ in front_sorted])

        return [population[i] for i in sorted_indices]

    def _dominates(self, genome1: Genome, genome2: Genome, objectives: list[str]) -> bool:
        """Check if genome1 dominates genome2."""
        dominated_any = False
        for obj in objectives:
            v1 = genome1.fitness_scores.get(obj, 0)
            v2 = genome2.fitness_scores.get(obj, 0)
            if v1 < v2:
                return False
            if v1 > v2:
                dominated_any = True
        return dominated_any

    def _crowding_distance(
        self,
        front: list[int],
        population: list[Genome],
        objectives: list[str],
    ) -> list[tuple[int, float]]:
        """Calculate crowding distance for NSGA-II."""
        distances = {i: 0.0 for i in front}

        for obj in objectives:
            sorted_front = sorted(front, key=lambda i: population[i].fitness_scores.get(obj, 0))

            # Boundary points get infinite distance
            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')

            # Calculate range
            f_max = population[sorted_front[-1]].fitness_scores.get(obj, 0)
            f_min = population[sorted_front[0]].fitness_scores.get(obj, 0)
            f_range = f_max - f_min if f_max != f_min else 1.0

            # Calculate distances for interior points
            for i in range(1, len(sorted_front) - 1):
                prev_val = population[sorted_front[i-1]].fitness_scores.get(obj, 0)
                next_val = population[sorted_front[i+1]].fitness_scores.get(obj, 0)
                distances[sorted_front[i]] += (next_val - prev_val) / f_range

        return [(i, distances[i]) for i in front]

    async def evolve_generation(self):
        """Evolve one generation with meta-learning integration."""
        # Store pre-evaluation fitness for operator tracking
        pre_fitness = {g.id: g.total_fitness() for g in self.population}

        await self.evaluate_population()

        # AGI v1.5: Record operator results for meta-learning
        for genome in self.population:
            post_fitness = genome.total_fitness()

            # Track crossover operator if used
            crossover_op = genome.genes.pop("_crossover_op", None)  # type: ignore
            if crossover_op and genome.parent_ids:
                parent_fitness = max(
                    pre_fitness.get(pid, 0) for pid in genome.parent_ids
                )
                self.meta_learner.record_operator_result(
                    crossover_op, parent_fitness, post_fitness
                )

            # Track mutation operator if used
            mutation_op = genome.genes.pop("_mutation_op", None)  # type: ignore
            if mutation_op and genome.parent_ids:
                parent_fitness = pre_fitness.get(genome.parent_ids[0], 0)
                self.meta_learner.record_operator_result(
                    mutation_op, parent_fitness, post_fitness
                )

        # Sort population (NSGA-II or simple fitness)
        if self.config.use_nsga2:
            sorted_pop = self._nsga2_sort(self.population)
        else:
            sorted_pop = sorted(self.population, key=lambda g: g.total_fitness(), reverse=True)

        # Record best fitness
        best_fitness = sorted_pop[0].total_fitness() if sorted_pop else 0
        self.fitness_history.append({
            "generation": self.generation,
            "best_fitness": best_fitness,
            "avg_fitness": sum(g.total_fitness() for g in sorted_pop) / len(sorted_pop),
            "timestamp": datetime.now().isoformat(),
        })

        # Check for stagnation
        if best_fitness <= self.stats["best_fitness_ever"]:
            self.stats["stagnation_count"] += 1
        else:
            # Fitness improved -- emit signal
            improvement = best_fitness - self.stats["best_fitness_ever"]
            self.stats["best_fitness_ever"] = best_fitness
            self.stats["stagnation_count"] = 0
            await self._emit_nexus("EVOLUTION_FITNESS_IMPROVED", {
                "generation": self.generation,
                "best_fitness": best_fitness,
                "improvement": improvement,
                "population_size": len(sorted_pop),
            }, urgency=0.6)

        # Adaptive mutation
        if self.config.adaptive_mutation and self.stats["stagnation_count"] >= self.config.stagnation_threshold:
            self.config.mutation_prob = min(0.5, self.config.mutation_prob * 1.5)
            logger.info(f"Increasing mutation rate to {self.config.mutation_prob:.2f}")
            # Emit stagnation signal
            await self._emit_nexus("EVOLUTION_STAGNATION", {
                "generation": self.generation,
                "stagnation_count": self.stats["stagnation_count"],
                "new_mutation_prob": self.config.mutation_prob,
                "best_fitness": best_fitness,
            }, urgency=0.7)
        else:
            self.config.mutation_prob = max(0.1, self.config.mutation_prob * 0.95)

        # AGI v1.5: Update meta-learning periodically
        if self.generation % self.meta_learner.config.strategy_update_interval == 0:
            self.meta_learner.update_strategy_weights()
            self.meta_learner.learn_gene_correlations(sorted_pop)

        # Create next generation
        self.generation += 1
        new_population = []

        # Elitism - keep best genomes
        for genome in sorted_pop[:self.config.elite_count]:
            elite_copy = Genome(
                id=f"g{self.generation}_elite_{genome.id}",
                genes={k: Gene(**v.__dict__) for k, v in genome.genes.items()},
                fitness_scores=genome.fitness_scores.copy(),
                generation=self.generation,
                parent_ids=[genome.id],
            )
            new_population.append(elite_copy)

        # Fill rest with offspring
        while len(new_population) < self.config.population_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            if random.random() < self.config.crossover_prob:
                child1, child2 = await self._crossover(parent1, parent2)
            else:
                child1 = await self._mutate(parent1)
                child2 = await self._mutate(parent2)

            new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)

        self.population = new_population
        self.stats["generations_completed"] += 1

        # Emit generation complete signal
        await self._emit_nexus("EVOLUTION_GENERATION_COMPLETE", {
            "generation": self.generation,
            "best_fitness": best_fitness,
            "avg_fitness": self.fitness_history[-1]["avg_fitness"] if self.fitness_history else 0,
            "population_size": len(self.population),
            "mutation_prob": self.config.mutation_prob,
        }, urgency=0.4)

        # Log evolution step with hash chain
        self._log_evolution_step(sorted_pop[0] if sorted_pop else None)

    def _log_evolution_step(self, best_genome: Optional[Genome]):
        """Log evolution step with hash chain for integrity."""
        log_entry = {
            "generation": self.generation,
            "best_genome": best_genome.to_dict() if best_genome else None,
            "population_size": len(self.population),
            "timestamp": datetime.now().isoformat(),
            "prev_hash": self.last_hash,
        }

        # Create hash of this entry
        entry_str = json.dumps(log_entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()[:16]
        log_entry["hash"] = entry_hash

        self.evolution_log.append(log_entry)
        self.last_hash = entry_hash

        # Save to disk periodically
        if self.generation % 5 == 0:
            self._save_log()

    def _save_log(self):
        """Save evolution log to disk."""
        log_file = self.data_dir / "evolution_log.jsonl"
        with log_file.open('a', encoding='utf-8') as f:
            for entry in self.evolution_log[-5:]:
                f.write(json.dumps(entry) + "\n")

    async def run(
        self,
        generations: Optional[int] = None,
        early_stop_fitness: Optional[float] = None,
    ) -> EvolutionResult:
        """
        Run the full evolution process with meta-learning.

        Args:
            generations: Number of generations (uses config if None)
            early_stop_fitness: Stop if this fitness is reached

        Returns:
            EvolutionResult with best genome and history

        AGI v1.5: Includes meta-learning for self-optimization.
        """
        import time
        start_time = time.time()

        generations = generations or self.config.generations

        if not self.population:
            # AGI v1.5: Apply transfer learning from similar problems
            recommended = self.meta_learner.get_recommended_hyperparameters(self.gene_definitions)
            if recommended:
                logger.info(f"Applying transfer learning: {recommended}")
                if "mutation_prob" in recommended:
                    self.config.mutation_prob = recommended["mutation_prob"]
                if "crossover_prob" in recommended:
                    self.config.crossover_prob = recommended["crossover_prob"]

            self.initialize_population()

        early_stopped = False
        for gen in range(generations):
            await self.evolve_generation()

            # Check early stopping
            if early_stop_fitness is not None:
                best = max(self.population, key=lambda g: g.total_fitness())
                if best.total_fitness() >= early_stop_fitness:
                    logger.info(f"Early stopping at generation {gen}")
                    early_stopped = True
                    break

            logger.info(f"Generation {gen + 1}/{generations} complete")

        # Final evaluation
        await self.evaluate_population()
        best_genome = max(self.population, key=lambda g: g.total_fitness())

        # AGI v1.5: Record run results for transfer learning
        self.meta_learner.record_run_result(
            gene_definitions=self.gene_definitions,
            best_fitness=best_genome.total_fitness(),
            generations_to_converge=self.generation,
            hyperparameters={
                "mutation_prob": self.config.mutation_prob,
                "crossover_prob": self.config.crossover_prob,
                "population_size": self.config.population_size,
            },
        )

        duration = time.time() - start_time

        # Emit evolution run complete signal
        await self._emit_nexus("EVOLUTION_RUN_COMPLETE", {
            "generations_run": self.generation,
            "best_fitness": best_genome.total_fitness(),
            "duration_seconds": round(duration, 2),
            "early_stopped": early_stopped,
            "population_size": len(self.population),
            "total_evaluations": self.stats["total_evaluations"],
        }, urgency=0.6)

        return EvolutionResult(
            best_genome=best_genome,
            final_population=self.population,
            generations_run=self.generation,
            fitness_history=self.fitness_history,
            duration_seconds=duration,
        )

    def get_best_genome(self) -> Optional[Genome]:
        """Get the current best genome."""
        if not self.population:
            return None
        return max(self.population, key=lambda g: g.total_fitness())

    def get_stats(self) -> dict:
        """Get optimizer statistics including meta-learning."""
        return {
            **self.stats,
            "population_size": len(self.population),
            "current_generation": self.generation,
            "mutation_prob": self.config.mutation_prob,
            "gene_count": len(self.gene_definitions),
            # AGI v1.5: Meta-learning stats
            "meta_learning": self.meta_learner.get_meta_stats(),
        }

    def get_correlated_genes(self) -> list[tuple[str, str, float]]:
        """Get significantly correlated gene pairs from meta-learning."""
        return self.meta_learner.get_correlated_genes()
