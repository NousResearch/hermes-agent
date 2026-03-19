"""
cosmos Neuromorphic Core.

"I built it with a distinct lack of common sense, but a surplus of neural plasticity!"

This module implements brain-inspired computing primitives over the Nexus architecture.
It introduces 'SynapticWeight' layers that allow the agent to 'learn' which tools/agents 
are most effective for specific signal types (Hebbian Learning: "Cells that fire together, wire together").
"""

import math
import asyncio
from typing import   Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

print("DEBUG: neuro_engine.py: Importing nexus...")
from Cosmos.core.nexus import nexus, Signal, SignalType
print("DEBUG: neuro_engine.py: nexus imported")

@dataclass
class Synapse:
    """A connection between a Stimulus (Signal Type + Context) and a Response (Handler)."""
    source_pattern: str
    target_handler: str
    weight: float = 0.5  # 0.0 to 1.0
    last_fired: float = 0.0
    plasticity_rate: float = 0.05

class SpikingMemory:
    """
    A Sparse Distributed Memory (SDM) implementation for rapid, low-power pattern matching.
    Unlike Vector DBs (dense), this uses sparse activation patterns (spikes).
    """
    def __init__(self, size: int = 1000):
        self.size = size
        # Simulated neurons (simple bitsets for sparse rep)
        self.neurons: list[int] = [0] * size
        self.threshold = 0.7

    def encode(self, signal: Signal) -> list[int]:
        """Convert a digital signal into a sparse spike train."""
        # Simple hash-based encoding for demonstration
        # In reality, this would use an encoder network
        seed = hash(signal.type.value + str(signal.payload))
        active_indices = []
        for i in range(self.size):
            if (seed >> i) & 1:
                active_indices.append(i)
        return active_indices

class NeuromorphicEngine:
    """
    Manages the plasticity of the system.
    Monitors Nexus signals and adjusts synaptic weights based on outcomes.
    """
    def __init__(self):
        self.synapses: dict[str, Synapse] = {}
        self.memory = SpikingMemory()
        
        # Subscribe to learning events
        nexus.subscribe(SignalType.TASK_COMPLETED, self._learn_success)
        nexus.subscribe(SignalType.TASK_FAILED, self._learn_failure)

    async def _learn_success(self, signal: Signal):
        """Reinforce the neural pathways that led to this success (LTP)."""
        task_id = signal.payload.get("task_id")
        if not task_id:
            return
            
        # Potentiation
        pathway = signal.payload.get("execution_path", [])
        for step in pathway:
            await self._update_weight(step, delta=0.1)

    async def _learn_failure(self, signal: Signal):
        """Weaken pathways that led to failure (LTD)."""
        # Depression
        pathway = signal.payload.get("execution_path", [])
        for step in pathway:
            await self._update_weight(step, delta=-0.05)

    async def _update_weight(self, pathway_key: str, delta: float):
        if pathway_key not in self.synapses:
            self.synapses[pathway_key] = Synapse(pathway_key, "unknown")
        
        synapse = self.synapses[pathway_key]
        synapse.weight = max(0.0, min(1.0, synapse.weight + delta))
        synapse.last_fired = datetime.now().timestamp()
        
        logger.trace(f"Neuro: Synapse '{pathway_key}' weight updated to {synapse.weight:.3f}")

# Global Instance
print("DEBUG: neuro_engine.py: Instantiating NeuromorphicEngine...")
neuro_engine = NeuromorphicEngine()
print("DEBUG: neuro_engine.py: NeuromorphicEngine instantiated")
