"""
cosmos Evolution System Tests

Comprehensive tests for:
- Genetic optimization
- Fitness tracking
- LoRA evolution
- Behavior mutation
- Multi-objective optimization
"""

import asyncio
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestFitnessTracker:
    """Tests for fitness tracking."""

    def test_tracker_creation(self):
        """Test fitness tracker initialization."""
        from cosmos.evolution.fitness_tracker import FitnessTracker

        tracker = FitnessTracker()

        assert tracker is not None
        assert tracker.get_current_fitness() is not None

    def test_metric_recording(self):
        """Test recording fitness metrics."""
        from cosmos.evolution.fitness_tracker import FitnessTracker

        tracker = FitnessTracker()

        tracker.record("task_success", 0.9)
        tracker.record("task_success", 0.8)
        tracker.record("user_satisfaction", 0.85)

        current = tracker.get_current_fitness()
        assert "task_success" in current
        assert current["task_success"] > 0

    def test_weighted_fitness(self):
        """Test weighted fitness calculation."""
        from cosmos.evolution.fitness_tracker import FitnessTracker

        tracker = FitnessTracker()

        tracker.record("task_success", 0.9)
        tracker.record("efficiency", 0.7)
        tracker.record("user_satisfaction", 0.8)

        weighted = tracker.get_weighted_fitness()
        assert 0 <= weighted <= 1

    def test_task_outcome_recording(self):
        """Test recording complete task outcomes."""
        from cosmos.evolution.fitness_tracker import FitnessTracker

        tracker = FitnessTracker()

        tracker.record_task_outcome(
            success=True,
            tokens_used=500,
            time_seconds=2.5,
            user_feedback=0.9,
        )

        stats = tracker.get_stats()
        assert "current_fitness" in stats
        assert "sample_counts" in stats

    def test_trend_calculation(self):
        """Test fitness trend calculation."""
        from cosmos.evolution.fitness_tracker import FitnessTracker

        tracker = FitnessTracker()

        # Record improving metrics
        for i in range(10):
            tracker.record("task_success", 0.5 + i * 0.05)

        trend = tracker.get_trend("task_success")
        assert trend > 0  # Should be positive (improving)

    def test_leaderboard(self):
        """Test genome leaderboard."""
        from cosmos.evolution.fitness_tracker import FitnessTracker

        tracker = FitnessTracker()

        # Record fitness for different genomes
        tracker.record_genome_fitness("genome_1", 0.85)
        tracker.record_genome_fitness("genome_2", 0.92)
        tracker.record_genome_fitness("genome_3", 0.78)

        leaderboard = tracker.get_leaderboard(top_k=3)

        assert len(leaderboard) == 3
        assert leaderboard[0][1] >= leaderboard[1][1]  # Sorted by fitness


class TestGeneticOptimizer:
    """Tests for genetic optimization."""

    def test_optimizer_creation(self):
        """Test genetic optimizer initialization."""
        from cosmos.evolution.genetic_optimizer import GeneticOptimizer

        optimizer = GeneticOptimizer(
            population_size=20,
            generations=10,
        )

        assert optimizer.population_size == 20
        assert optimizer.generations == 10

    def test_genome_creation(self):
        """Test creating genomes."""
        from cosmos.evolution.genetic_optimizer import GeneticOptimizer, Genome

        optimizer = GeneticOptimizer()

        genome = optimizer.create_random_genome()

        assert genome is not None
        assert hasattr(genome, "genes")
        assert len(genome.genes) > 0

    def test_mutation(self):
        """Test genome mutation."""
        from cosmos.evolution.genetic_optimizer import GeneticOptimizer

        optimizer = GeneticOptimizer(mutation_rate=0.5)

        genome = optimizer.create_random_genome()
        original_genes = dict(genome.genes)

        mutated = optimizer.mutate(genome)

        # With high mutation rate, some genes should differ
        assert mutated is not None

    def test_crossover(self):
        """Test genome crossover."""
        from cosmos.evolution.genetic_optimizer import GeneticOptimizer

        optimizer = GeneticOptimizer()

        parent1 = optimizer.create_random_genome()
        parent2 = optimizer.create_random_genome()

        child1, child2 = optimizer.crossover(parent1, parent2)

        assert child1 is not None
        assert child2 is not None

    def test_selection(self):
        """Test selection mechanism."""
        from cosmos.evolution.genetic_optimizer import GeneticOptimizer

        optimizer = GeneticOptimizer(population_size=10)

        # Create population with varying fitness
        population = []
        for i in range(10):
            genome = optimizer.create_random_genome()
            genome.fitness = i / 10
            population.append(genome)

        selected = optimizer.select(population, k=5)

        assert len(selected) == 5
        # Higher fitness individuals should be more likely selected
        avg_fitness = sum(g.fitness for g in selected) / len(selected)
        assert avg_fitness > 0.3  # Should be above random average

    @pytest.mark.asyncio
    async def test_evolution_run(self):
        """Test running evolution."""
        from cosmos.evolution.genetic_optimizer import GeneticOptimizer

        optimizer = GeneticOptimizer(
            population_size=10,
            generations=5,
        )

        # Simple fitness function
        def fitness_fn(genome):
            return sum(genome.genes.values()) / len(genome.genes)

        result = await optimizer.run(
            fitness_function=fitness_fn,
            generations=3,
        )

        assert result is not None
        assert result.best_genome is not None
        assert result.generations_run == 3

    def test_nsga2_sorting(self):
        """Test NSGA-II non-dominated sorting."""
        from cosmos.evolution.genetic_optimizer import GeneticOptimizer

        optimizer = GeneticOptimizer()

        # Create population with multi-objective fitness
        population = []
        for i in range(10):
            genome = optimizer.create_random_genome()
            genome.objectives = {
                "task_success": i / 10,
                "efficiency": (10 - i) / 10,  # Trade-off
            }
            population.append(genome)

        fronts = optimizer.non_dominated_sort(population)

        assert len(fronts) > 0
        # First front should have Pareto-optimal solutions


class TestLoRAEvolver:
    """Tests for LoRA adapter evolution."""

    def test_evolver_creation(self):
        """Test LoRA evolver initialization."""
        from cosmos.evolution.lora_evolver import LoRAEvolver

        evolver = LoRAEvolver()
        assert evolver is not None

    @pytest.mark.asyncio
    async def test_adapter_creation(self, temp_data_dir):
        """Test creating a LoRA adapter."""
        from cosmos.evolution.lora_evolver import LoRAEvolver

        evolver = LoRAEvolver(output_dir=temp_data_dir)

        adapter = await evolver.create_adapter(
            base_model="test-model",
            training_data=[
                {"input": "Hello", "output": "Hi there!"},
            ],
            rank=8,
        )

        assert adapter is not None

    @pytest.mark.asyncio
    async def test_adapter_breeding(self, temp_data_dir):
        """Test breeding two adapters."""
        from cosmos.evolution.lora_evolver import LoRAEvolver

        evolver = LoRAEvolver(output_dir=temp_data_dir)

        # Create parent adapters (mock)
        parent1 = {"id": "adapter_1", "weights": [0.1, 0.2, 0.3]}
        parent2 = {"id": "adapter_2", "weights": [0.4, 0.5, 0.6]}

        child = await evolver.breed(parent1, parent2)

        assert child is not None

    def test_adapter_registry(self, temp_data_dir):
        """Test adapter registry operations."""
        from cosmos.evolution.lora_evolver import LoRAEvolver

        evolver = LoRAEvolver(output_dir=temp_data_dir)

        # Register adapters
        evolver.register_adapter("adapter_1", 0.85)
        evolver.register_adapter("adapter_2", 0.90)

        top = evolver.get_top_adapters(k=2)
        assert len(top) == 2
        assert top[0][1] >= top[1][1]


class TestBehaviorMutation:
    """Tests for swarm behavior evolution."""

    def test_mutator_creation(self):
        """Test behavior mutator initialization."""
        from cosmos.evolution.behavior_mutation import BehaviorMutator

        mutator = BehaviorMutator()
        assert mutator is not None
        assert mutator.generation == 0

    def test_behavior_params(self):
        """Test getting behavior parameters."""
        from cosmos.evolution.behavior_mutation import BehaviorMutator

        mutator = BehaviorMutator()
        params = mutator.get_behavior_params()

        assert "temperature" in params
        assert "verbosity" in params
        assert 0 <= params["temperature"] <= 1

    def test_team_config(self):
        """Test getting team configuration."""
        from cosmos.evolution.behavior_mutation import BehaviorMutator

        mutator = BehaviorMutator()
        config = mutator.get_team_config()

        assert "code_specialist_weight" in config
        assert "reasoning_specialist_weight" in config

    def test_generation_evolution(self):
        """Test evolving to next generation."""
        from cosmos.evolution.behavior_mutation import BehaviorMutator

        mutator = BehaviorMutator()
        initial_gen = mutator.generation
        initial_params = dict(mutator.get_behavior_params())

        mutator.evolve_generation()

        assert mutator.generation == initial_gen + 1
        # Parameters should have changed (probabilistically)

    def test_fitness_based_evolution(self):
        """Test evolution based on fitness feedback."""
        from cosmos.evolution.behavior_mutation import BehaviorMutator

        mutator = BehaviorMutator()

        # Record some fitness values
        mutator.record_fitness({
            "task_success": 0.8,
            "efficiency": 0.7,
        })

        mutator.record_fitness({
            "task_success": 0.85,
            "efficiency": 0.75,
        })

        # Evolution should favor successful configurations
        mutator.evolve_generation()

        assert mutator.generation > 0

    def test_genome_encoding(self):
        """Test encoding/decoding behavior genome."""
        from cosmos.evolution.behavior_mutation import BehaviorMutator

        mutator = BehaviorMutator()

        # Encode current state
        genome = mutator.encode_genome()

        assert isinstance(genome, dict)
        assert "behavior_params" in genome
        assert "team_config" in genome

        # Decode back
        mutator.decode_genome(genome)

        # Should restore same parameters
        assert mutator.get_behavior_params() == genome["behavior_params"]


class TestEvolutionIntegration:
    """Integration tests for the evolution system."""

    @pytest.mark.asyncio
    async def test_full_evolution_cycle(self, temp_data_dir):
        """Test complete evolution cycle."""
        from cosmos.evolution.fitness_tracker import FitnessTracker
        from cosmos.evolution.genetic_optimizer import GeneticOptimizer
        from cosmos.evolution.behavior_mutation import BehaviorMutator

        # Initialize components
        tracker = FitnessTracker()
        optimizer = GeneticOptimizer(population_size=10)
        mutator = BehaviorMutator()

        # Record some task outcomes
        for _ in range(10):
            tracker.record_task_outcome(
                success=True,
                tokens_used=500,
                time_seconds=2.0,
            )

        # Run genetic optimization
        def fitness_fn(genome):
            return sum(genome.genes.values()) / len(genome.genes)

        result = await optimizer.run(fitness_function=fitness_fn, generations=3)

        # Evolve behavior based on fitness
        mutator.record_fitness(tracker.get_current_fitness())
        mutator.evolve_generation()

        assert result.best_genome is not None
        assert mutator.generation > 0

    def test_hash_chain_logging(self, temp_data_dir):
        """Test tamper-proof evolution logging."""
        from cosmos.evolution.fitness_tracker import FitnessTracker

        tracker = FitnessTracker(log_dir=temp_data_dir)

        # Record events
        tracker.record("task_success", 0.8)
        tracker.record("task_success", 0.85)

        # Get log entries
        log = tracker.get_evolution_log()

        assert len(log) >= 2

        # Verify hash chain integrity
        for i in range(1, len(log)):
            assert log[i].get("prev_hash") == log[i-1].get("hash")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
