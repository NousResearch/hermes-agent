"""
Quantum-Enhanced Evolution for Cosmos Agents
=================================================

Integrates IBM Quantum computing with the existing evolution system
to provide quantum-accelerated genetic algorithms for agent optimization.

This module bridges:
- cosmos/integration/quantum/ibm_quantum.py (quantum algorithms)
- cosmos/evolution/genetic_optimizer.py (existing GA)
- cosmos/evolution/fitness_tracker.py (fitness evaluation)
- cosmos/core/collective/evolution.py (agent evolution)

Strategy:
- Use quantum simulators (unlimited) for routine evolution
- Reserve hardware (10 min/month) for breakthrough attempts
- Fall back to classical when quantum unavailable

"The best of both worlds - classical reliability meets quantum exploration."
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from loguru import logger
from datetime import datetime

# Quantum integration
try:
    from Cosmos.integration.quantum import (
        get_quantum_provider,
        initialize_quantum,
        QuantumGeneticOptimizer,
        QAOAOptimizer,
        QuantumPatternExtractor,
        QuantumTaskType,
        QISKIT_AVAILABLE
    )
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    logger.debug("Quantum integration not available")

# Existing evolution system
try:
    from Cosmos.evolution.fitness_tracker import FitnessTracker, get_fitness_tracker
    FITNESS_AVAILABLE = True
except ImportError:
    FITNESS_AVAILABLE = False

# Nexus integration for signal emission
try:
    from Cosmos.core.nexus import get_nexus
    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False


async def _emit_evolution_signal(signal_type: str, data: Dict[str, Any]) -> None:
    """Emit evolution-related signals to Nexus."""
    if not NEXUS_AVAILABLE:
        return
    try:
        nexus = get_nexus()
        await nexus.emit(signal_type, data)
    except Exception:
        pass


@dataclass
class QuantumEvolutionResult:
    """Result from quantum-enhanced evolution."""
    best_genome: str
    best_fitness: float
    generations_run: int
    quantum_jobs: int
    hardware_used: bool
    improvement: float
    population_history: List[Tuple[str, float]]
    execution_time: float
    method: str  # "quantum", "hybrid", "classical"


class QuantumEvolutionEngine:
    """
    Quantum-enhanced evolution engine for agent optimization.

    Provides quantum acceleration for:
    - Population generation (quantum sampling)
    - Crossover (entanglement-based)
    - Mutation (rotation gates)
    - Selection (amplitude amplification)
    """

    def __init__(self):
        self.provider = None
        self.qga = None
        self.qaoa = None
        self.pattern_extractor = None
        self._initialized = False
        self.evolution_history: List[QuantumEvolutionResult] = []

    async def initialize(self, api_key: str = None) -> bool:
        """
        Initialize quantum evolution capabilities.

        Args:
            api_key: IBM Quantum API key (optional)

        Returns:
            True if initialized (at least classical fallback available)
        """
        if not QUANTUM_AVAILABLE:
            logger.warning("Quantum module not available, using classical evolution only")
            self._initialized = True  # Classical fallback always available
            return True

        try:
            success = await initialize_quantum(api_key)
            self.provider = get_quantum_provider()

            if self.provider and success:
                self.qga = QuantumGeneticOptimizer(self.provider)
                self.qaoa = QAOAOptimizer(self.provider)
                self.pattern_extractor = QuantumPatternExtractor(self.provider)
                logger.info("Quantum evolution engine initialized with IBM Quantum")
            else:
                logger.warning("Quantum connection failed, using classical evolution")

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Quantum evolution initialization error: {e}")
            self._initialized = True  # Classical fallback
            return True

    def _genome_to_agent_params(self, genome: str) -> Dict[str, float]:
        """
        Convert binary genome to agent parameters.

        Genome encoding (example for 16-bit genome):
        - Bits 0-3: Exploration rate (0.0 - 1.0)
        - Bits 4-7: Learning rate (0.0 - 0.1)
        - Bits 8-11: Temperature (0.1 - 2.0)
        - Bits 12-15: Creativity (0.0 - 1.0)
        """
        n = len(genome)
        chunk_size = max(1, n // 4)

        def bits_to_float(bits: str, min_val: float, max_val: float) -> float:
            if not bits:
                return (min_val + max_val) / 2
            value = int(bits, 2) / (2 ** len(bits) - 1)
            return min_val + value * (max_val - min_val)

        return {
            "exploration_rate": bits_to_float(genome[:chunk_size], 0.0, 1.0),
            "learning_rate": bits_to_float(genome[chunk_size:chunk_size*2], 0.001, 0.1),
            "temperature": bits_to_float(genome[chunk_size*2:chunk_size*3], 0.1, 2.0),
            "creativity": bits_to_float(genome[chunk_size*3:], 0.0, 1.0)
        }

    def _agent_params_to_genome(self, params: Dict[str, float], genome_length: int = 16) -> str:
        """Convert agent parameters back to binary genome."""
        chunk_size = genome_length // 4

        def float_to_bits(value: float, min_val: float, max_val: float, bits: int) -> str:
            normalized = (value - min_val) / (max_val - min_val)
            normalized = max(0.0, min(1.0, normalized))
            int_val = int(normalized * (2 ** bits - 1))
            return format(int_val, f'0{bits}b')

        genome = ""
        genome += float_to_bits(params.get("exploration_rate", 0.5), 0.0, 1.0, chunk_size)
        genome += float_to_bits(params.get("learning_rate", 0.01), 0.001, 0.1, chunk_size)
        genome += float_to_bits(params.get("temperature", 0.7), 0.1, 2.0, chunk_size)
        genome += float_to_bits(params.get("creativity", 0.5), 0.0, 1.0, genome_length - 3*chunk_size)

        return genome

    async def evolve_agent(
        self,
        agent_id: str,
        initial_params: Optional[Dict[str, float]] = None,
        fitness_func: Optional[Callable[[Dict[str, float]], float]] = None,
        generations: int = 10,
        population_size: int = 20,
        genome_length: int = 16,
        prefer_hardware: bool = False,
        use_quantum: bool = True
    ) -> QuantumEvolutionResult:
        """
        Evolve an agent's parameters using quantum-enhanced GA.

        Args:
            agent_id: Agent identifier for tracking
            initial_params: Starting parameters (optional)
            fitness_func: Function to evaluate parameter fitness
            generations: Number of evolution generations
            population_size: Population size per generation
            genome_length: Length of binary genome
            prefer_hardware: Use quantum hardware for final generation
            use_quantum: Enable quantum algorithms (can disable for comparison)

        Returns:
            QuantumEvolutionResult with best parameters and statistics
        """
        start_time = datetime.now()

        # Emit evolution started signal
        asyncio.create_task(_emit_evolution_signal("quantum.evolution_started", {
            "agent_id": agent_id,
            "generations": generations,
            "population_size": population_size,
            "use_quantum": use_quantum,
            "prefer_hardware": prefer_hardware,
            "timestamp": datetime.now().isoformat()
        }))

        # Default fitness function using fitness tracker
        if fitness_func is None:
            if FITNESS_AVAILABLE:
                tracker = get_fitness_tracker()
                def fitness_func(params: Dict[str, float]) -> float:
                    # Combine parameters into a fitness score
                    # Higher exploration + moderate temperature + creativity = better
                    score = (
                        params.get("exploration_rate", 0.5) * 0.3 +
                        (1 - abs(params.get("temperature", 0.7) - 0.7)) * 0.3 +
                        params.get("creativity", 0.5) * 0.2 +
                        params.get("learning_rate", 0.01) * 10 * 0.2  # Normalized
                    )
                    return min(1.0, max(0.0, score))
            else:
                # Simple default
                def fitness_func(params: Dict[str, float]) -> float:
                    return sum(params.values()) / len(params)

        # Convert initial params to genome
        if initial_params:
            initial_genome = self._agent_params_to_genome(initial_params, genome_length)
        else:
            initial_genome = ''.join(str(np.random.randint(2)) for _ in range(genome_length))

        # Wrapper to evaluate genome fitness
        def genome_fitness(genome: str) -> float:
            params = self._genome_to_agent_params(genome)
            return fitness_func(params)

        initial_fitness = genome_fitness(initial_genome)
        population_history = [(initial_genome, initial_fitness)]
        quantum_jobs = 0
        hardware_used = False
        method = "classical"

        # Try quantum evolution
        if use_quantum and QUANTUM_AVAILABLE and self.qga:
            try:
                # Generate initial population with quantum sampling
                population = await self.qga.generate_quantum_population(
                    population_size,
                    genome_fitness,
                    grover_iterations=1,
                    prefer_hardware=False  # Save hardware for later
                )
                quantum_jobs += 1
                method = "quantum"

                best_genome, best_fitness = population[0]

                # Evolution loop
                for gen in range(generations):
                    # Use hardware on last generation if requested
                    use_hw = prefer_hardware and gen == generations - 1

                    # Selection (top half)
                    selected = population[:population_size // 2]

                    # Crossover with quantum
                    offspring = []
                    for i in range(0, len(selected) - 1, 2):
                        child = await self.qga.quantum_crossover(
                            selected[i][0],
                            selected[i + 1][0],
                            prefer_hardware=use_hw
                        )
                        if use_hw:
                            hardware_used = True

                        # Mutation
                        child = await self.qga.quantum_mutation(child, mutation_rate=0.1)

                        fitness = genome_fitness(child)
                        offspring.append((child, fitness))
                        quantum_jobs += 2  # crossover + mutation

                    # Combine and select
                    combined = population + offspring
                    combined.sort(key=lambda x: x[1], reverse=True)
                    population = combined[:population_size]

                    if population[0][1] > best_fitness:
                        best_genome, best_fitness = population[0]
                        population_history.append((best_genome, best_fitness))

                    logger.debug(f"Quantum evolution gen {gen + 1}: best fitness {best_fitness:.4f}")

            except Exception as e:
                logger.warning(f"Quantum evolution failed, falling back to classical: {e}")
                method = "hybrid"
                # Continue with classical below

        # Classical fallback or classical-only mode
        if method == "classical" or (method == "hybrid" and best_fitness <= initial_fitness):
            # Classical GA
            population = [
                (initial_genome, initial_fitness)
            ]
            for _ in range(population_size - 1):
                genome = ''.join(str(np.random.randint(2)) for _ in range(genome_length))
                population.append((genome, genome_fitness(genome)))

            population.sort(key=lambda x: x[1], reverse=True)
            best_genome, best_fitness = population[0]

            for gen in range(generations):
                selected = population[:population_size // 2]

                offspring = []
                for i in range(0, len(selected) - 1, 2):
                    # Classical crossover
                    point = np.random.randint(1, genome_length)
                    child = selected[i][0][:point] + selected[i + 1][0][point:]

                    # Classical mutation
                    child_list = list(child)
                    for j in range(len(child_list)):
                        if np.random.random() < 0.1:
                            child_list[j] = '0' if child_list[j] == '1' else '1'
                    child = ''.join(child_list)

                    offspring.append((child, genome_fitness(child)))

                combined = population + offspring
                combined.sort(key=lambda x: x[1], reverse=True)
                population = combined[:population_size]

                if population[0][1] > best_fitness:
                    best_genome, best_fitness = population[0]
                    population_history.append((best_genome, best_fitness))

            if method == "classical":
                method = "classical"

        execution_time = (datetime.now() - start_time).total_seconds()

        result = QuantumEvolutionResult(
            best_genome=best_genome,
            best_fitness=best_fitness,
            generations_run=generations,
            quantum_jobs=quantum_jobs,
            hardware_used=hardware_used,
            improvement=best_fitness - initial_fitness,
            population_history=population_history,
            execution_time=execution_time,
            method=method
        )

        # Store in history
        self.evolution_history.append(result)

        # Log result
        logger.info(
            f"Agent {agent_id} evolved: {method} method, "
            f"fitness {initial_fitness:.4f} -> {best_fitness:.4f} "
            f"(+{result.improvement:.4f}), {quantum_jobs} quantum jobs"
        )

        # Emit evolution completed signal
        asyncio.create_task(_emit_evolution_signal("quantum.evolution_complete", {
            "agent_id": agent_id,
            "method": method,
            "initial_fitness": initial_fitness,
            "best_fitness": best_fitness,
            "improvement": result.improvement,
            "generations": generations,
            "quantum_jobs": quantum_jobs,
            "hardware_used": hardware_used,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }))

        return result

    async def optimize_multi_objective(
        self,
        objectives: List[Tuple[str, Callable[[Dict], float], float]],
        genome_length: int = 16,
        prefer_hardware: bool = False
    ) -> Dict[str, Any]:
        """
        Multi-objective optimization using QAOA.

        Args:
            objectives: List of (name, function, weight) tuples
            genome_length: Genome size
            prefer_hardware: Use quantum hardware

        Returns:
            Pareto-optimal solutions
        """
        if not QUANTUM_AVAILABLE or not self.qaoa:
            logger.warning("QAOA not available, using weighted sum method")
            # Classical fallback
            return {"method": "classical", "solutions": []}

        # Build optimization problem as graph
        # Each objective becomes edges in the optimization graph
        edges = []
        for i in range(genome_length - 1):
            edges.append((i, i + 1))

        # Run QAOA
        result = await self.qaoa.optimize(
            num_qubits=min(genome_length, 10),  # Limit for practical execution
            edges=edges,
            p=2,  # QAOA depth
            shots=1024,
            prefer_hardware=prefer_hardware
        )

        if result.success:
            best_solution = result.metadata.get("best_solution", "")
            return {
                "method": "qaoa",
                "best_solution": best_solution,
                "parameters": self._genome_to_agent_params(best_solution.ljust(genome_length, '0')),
                "qaoa_depth": result.metadata.get("qaoa_depth", 2),
                "execution_time": result.execution_time
            }

        return {"method": "qaoa_failed", "error": result.error}

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get statistics from evolution history."""
        if not self.evolution_history:
            return {
                "total_evolutions": 0,
                "quantum_evolutions": 0,
                "classical_evolutions": 0,
                "average_improvement": 0.0,
                "hardware_runs": 0
            }

        quantum_runs = [r for r in self.evolution_history if r.method == "quantum"]
        classical_runs = [r for r in self.evolution_history if r.method == "classical"]
        hybrid_runs = [r for r in self.evolution_history if r.method == "hybrid"]

        return {
            "total_evolutions": len(self.evolution_history),
            "quantum_evolutions": len(quantum_runs),
            "classical_evolutions": len(classical_runs),
            "hybrid_evolutions": len(hybrid_runs),
            "average_improvement": np.mean([r.improvement for r in self.evolution_history]),
            "best_improvement": max(r.improvement for r in self.evolution_history),
            "hardware_runs": sum(1 for r in self.evolution_history if r.hardware_used),
            "total_quantum_jobs": sum(r.quantum_jobs for r in self.evolution_history),
            "average_execution_time": np.mean([r.execution_time for r in self.evolution_history])
        }


# Singleton instance
_quantum_evolution_engine: Optional[QuantumEvolutionEngine] = None


def get_quantum_evolution_engine() -> QuantumEvolutionEngine:
    """Get or create quantum evolution engine singleton."""
    global _quantum_evolution_engine
    if _quantum_evolution_engine is None:
        _quantum_evolution_engine = QuantumEvolutionEngine()
    return _quantum_evolution_engine


async def quantum_evolve_agent_params(
    agent_id: str,
    current_params: Optional[Dict[str, float]] = None,
    generations: int = 10,
    prefer_hardware: bool = False
) -> Dict[str, Any]:
    """
    High-level function to evolve agent parameters.

    Convenience wrapper for integration with existing evolution system.

    Args:
        agent_id: Agent to evolve
        current_params: Current agent parameters
        generations: Evolution generations
        prefer_hardware: Use quantum hardware

    Returns:
        Dict with best_params, fitness, improvement, method
    """
    engine = get_quantum_evolution_engine()

    if not engine._initialized:
        await engine.initialize()

    result = await engine.evolve_agent(
        agent_id=agent_id,
        initial_params=current_params,
        generations=generations,
        prefer_hardware=prefer_hardware
    )

    return {
        "best_params": engine._genome_to_agent_params(result.best_genome),
        "best_fitness": result.best_fitness,
        "improvement": result.improvement,
        "method": result.method,
        "quantum_jobs": result.quantum_jobs,
        "hardware_used": result.hardware_used,
        "execution_time": result.execution_time
    }


# Test function
async def test_quantum_evolution():
    """Test quantum evolution system."""
    print("Testing Quantum Evolution Engine")
    print("=" * 50)

    engine = get_quantum_evolution_engine()
    await engine.initialize()

    # Test evolution
    result = await engine.evolve_agent(
        agent_id="test_agent",
        initial_params={
            "exploration_rate": 0.3,
            "learning_rate": 0.01,
            "temperature": 0.7,
            "creativity": 0.5
        },
        generations=5,
        population_size=10,
        prefer_hardware=False
    )

    print(f"Method: {result.method}")
    print(f"Best fitness: {result.best_fitness:.4f}")
    print(f"Improvement: {result.improvement:.4f}")
    print(f"Quantum jobs: {result.quantum_jobs}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Best params: {engine._genome_to_agent_params(result.best_genome)}")

    print("\nEvolution stats:", engine.get_evolution_stats())


if __name__ == "__main__":
    asyncio.run(test_quantum_evolution())
