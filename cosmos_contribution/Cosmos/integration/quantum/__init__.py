"""
Cosmos Quantum Computing Integration
=========================================

Provides quantum-enhanced capabilities via IBM Quantum Experience.
Implementation follows IBM Quantum best practices (2026 docs).

Free Tier Strategy:
- 10 minutes/month hardware for high-value tasks (evolution, optimization)
- Unlimited simulators for development and testing

Execution Modes (per IBM docs):
- JOB: Single primitive request, no context
- BATCH: Multiple independent jobs in parallel (best for parallel experiments)
- SESSION: Exclusive QPU access for iterative workflows (VQE, optimization loops)

Error Mitigation (per IBM docs):
- Level 0: No mitigation
- Level 1: TREX readout error correction (default)
- Level 2: TREX + ZNE + gate twirling (~3x overhead)

Quick Start:
    from Cosmos.integration.quantum import (
        initialize_quantum, quantum_evolve_agent,
        QuantumOptions, ResilienceLevel, ExecutionMode
    )

    # Initialize (use env var IBM_QUANTUM_API_KEY or pass key)
    await initialize_quantum()

    # Evolve an agent genome
    best_genome, fitness = await quantum_evolve_agent(
        agent_genome="10110010",
        fitness_func=lambda x: sum(int(b) for b in x) / len(x),
        generations=5
    )

    # Configure advanced options
    options = QuantumOptions(
        execution_mode=ExecutionMode.SESSION,
        resilience_level=ResilienceLevel.MEDIUM,
        dynamical_decoupling=True,
        enable_twirling=True
    )

Components:
- IBMQuantumProvider: Core connection and execution management
- QuantumGeneticOptimizer: QGA for agent evolution
- QAOAOptimizer: Combinatorial optimization (QAOA algorithm)
- QuantumPatternExtractor: Memory pattern discovery
- QuantumOptions: Configuration for error mitigation and execution modes
"""

from .ibm_quantum import (
    # Core provider
    IBMQuantumProvider,
    get_quantum_provider,
    initialize_quantum,

    # Enums
    QuantumBackend,
    QuantumTaskType,
    ExecutionMode,
    ResilienceLevel,

    # Data classes
    QuantumUsageStats,
    QuantumJobResult,
    QuantumOptions,

    # Algorithms
    QuantumGeneticOptimizer,
    QAOAOptimizer,
    QuantumPatternExtractor,

    # High-level functions
    quantum_evolve_agent,

    # Availability flag
    QISKIT_AVAILABLE,
)

__all__ = [
    "IBMQuantumProvider",
    "get_quantum_provider",
    "initialize_quantum",
    "QuantumBackend",
    "QuantumTaskType",
    "ExecutionMode",
    "ResilienceLevel",
    "QuantumUsageStats",
    "QuantumJobResult",
    "QuantumOptions",
    "QuantumGeneticOptimizer",
    "QAOAOptimizer",
    "QuantumPatternExtractor",
    "quantum_evolve_agent",
    "QISKIT_AVAILABLE",
]
