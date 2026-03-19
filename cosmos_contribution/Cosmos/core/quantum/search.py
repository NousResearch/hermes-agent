"""
cosmos Quantum-Inspired Search (Schrödinger's Query)
--------------------------------------------------------

"I finished the search before I even started it!"

This module implements a Quantum-Inspired Evolutionary Algorithm (QIEA) for reasoning path exploration.
Instead of binary states (explored/unexplored), reasoning nodes exist in a superposition of states
represented by a Q-bit (alpha, beta).

Concepts:
- **Q-bit Representation**: Probabilistic representation of reasoning validity.
- **Interference**: Destructive/Constructive interference between reasoning paths.
- **Collapse**: Observation triggers state collapse to the most probable solution.
"""

import math
import random
import asyncio
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import numpy as np

from loguru import logger

class QuantumState(Enum):
    SUPERPOSITION = 0
    COLLAPSED_TRUE = 1
    COLLAPSED_FALSE = 2

@dataclass
class QBit:
    """
    A quantum bit representing the probability of a reasoning step being valid.
    |psi> = alpha|0> + beta|1>
    where |alpha|^2 + |beta|^2 = 1
    """
    alpha: float = 1/math.sqrt(2) # Probability of being False (0)
    beta: float = 1/math.sqrt(2)  # Probability of being True (1)
    
    label: str = ""
    content: str = ""
    
    def measure(self) -> bool:
        """Collapse the Q-bit state based on probabilities."""
        if random.random() < (self.beta ** 2):
            return True
        return False

    def rotate(self, theta: float):
        """Apply a rotation gate (update belief based on heuristic)."""
        # Rotation matrix:
        # [ cos(theta) -sin(theta) ]
        # [ sin(theta)  cos(theta) ]
        new_alpha = self.alpha * math.cos(theta) - self.beta * math.sin(theta)
        new_beta = self.alpha * math.sin(theta) + self.beta * math.cos(theta)
        
        # Normalize to ensure valid quantum state
        norm = math.sqrt(new_alpha**2 + new_beta**2)
        self.alpha = new_alpha / norm
        self.beta = new_beta / norm

@dataclass
class EntanglementGroup:
    """Group of Q-bits that are entangled (correlated)."""
    qbits: list[str] # IDs of interacting qbits
    correlation_strength: float = 0.5 

class SchrodingerSearch:
    """
    Quantum-Inspired Search Engine for Reasoning Paths.
    """
    def __init__(self, observation_threshold: float = 0.85):
        self.qbits: dict[str, QBit] = {}
        self.entanglements: list[EntanglementGroup] = []
        self.observation_threshold = observation_threshold
        self.history: list[str] = []

    def add_reasoning_node(self, id: str, content: str, initial_confidence: float = 0.5):
        """
        Add a reasoning node in a superposition state.
        
        Args:
            id: Unique identifier
            content: The reasoning step info
            initial_confidence: A heuristic estimate (0.0 to 1.0)
        """
        # Convert classical confidence to quantum amplitude
        # If confidence is 1.0, beta should be 1.0. If 0.0, alpha 1.0.
        # We map 0..1 to rotation angle 0..pi/2
        theta = initial_confidence * (math.pi / 2)
        alpha = math.cos(theta)
        beta = math.sin(theta)
        
        # Invert alpha/beta mapping because usually alpha is coeff of |0> (False)
        # So high confidence -> high beta (coeff of |1>)
        # The calculation above gives:
        # conf=1.0 -> theta=pi/2 -> cos=0, sin=1 -> alpha=0, beta=1 -> |1> (Correct)
        # conf=0.0 -> theta=0 -> cos=1, sin=0 -> alpha=1, beta=0 -> |0> (Correct)
        
        self.qbits[id] = QBit(alpha=alpha, beta=beta, label=id, content=content)
        logger.debug(f"Added Q-node {id}: |{alpha:.2f}|0> + |{beta:.2f}|1>")

    def entangle_nodes(self, id_a: str, id_b: str, strength: float = 0.5):
        """
        Entangle two nodes. If one collapses to True, the other is influenced.
        Used for logical dependencies (if A is True, B is likely True).
        """
        if id_a in self.qbits and id_b in self.qbits:
            self.entanglements.append(EntanglementGroup([id_a, id_b], strength))

    async def interfere(self, heuristic_fn: Callable[[str], float]):
        """
        Apply 'interference' to the system. 
        This is the search step where we update amplitudes based on heuristics.
        Similar to Grover's diffusion operator but for semantic search.
        """
        logger.info("Applying Quantum Interference Wave...")
        
        # 1. Oracle Phase: Mark promising states
        for id, qbit in self.qbits.items():
            # External heuristic acts as the 'Oracle' identifying good states
            # We don't measure yet, just update phase/amplitude
            heuristic_val = heuristic_fn(qbit.content)
            
            # Rotation angle: Positive heuristic rotates towards |1>, negative towards |0>
            # Map -1..1 score to -pi/4..pi/4 rotation
            rotation = heuristic_val * (math.pi / 4)
            qbit.rotate(rotation)

        # 2. Diffusion Phase: Spread probability through entanglement
        for group in self.entanglements:
            # Simplified entanglement: Average probability mass transfer
            # If qbit A is high prob, it boosts qbit B
            q_a = self.qbits[group.qbits[0]]
            q_b = self.qbits[group.qbits[1]]
            
            # Constructive interference
            avg_beta = (q_a.beta + q_b.beta) / 2
            
            # Pull both towards the average, scaled by strength
            q_a.beta += (avg_beta - q_a.beta) * group.correlation_strength
            q_b.beta += (avg_beta - q_b.beta) * group.correlation_strength
            
            # Renormalize
            q_a.alpha = math.sqrt(1 - min(q_a.beta**2, 1.0))
            q_b.alpha = math.sqrt(1 - min(q_b.beta**2, 1.0))

    def collapse(self) -> list[tuple[str, str, float]]:
        """
        Observe the system. Collapse wavefunctions and return valid paths.
        Returns list of (id, content, confidence).
        """
        valid_paths = []
        logger.info("Collapsing Wavefunction...")
        
        for id, qbit in self.qbits.items():
            is_valid = qbit.measure()
            confidence = qbit.beta ** 2  # Prob of being True
            
            if is_valid:
                valid_paths.append((id, qbit.content, confidence))
                
        # Sort by confidence
        valid_paths.sort(key=lambda x: x[2], reverse=True)
        return valid_paths

    def get_superposition_status(self) -> str:
        """Visualize the current superposition."""
        status = []
        for id, qbit in self.qbits.items():
            prob = qbit.beta ** 2
            bar = "▓" * int(prob * 10) + "░" * (10 - int(prob * 10))
            status.append(f"{id}: [{bar}] {prob:.1%}")
        return "\n".join(status)

async def demo_quantum_search():
    """Demonstrate the module."""
    qs = SchrodingerSearch()
    
    # Init superposition of possible bug causes
    qs.add_reasoning_node("cause_A", "Database connection timeout", 0.3)
    qs.add_reasoning_node("cause_B", "API Rate limit exceeded", 0.3)
    qs.add_reasoning_node("cause_C", "Memory leak in worker", 0.3)
    
    print("Initial State:")
    print(qs.get_superposition_status())
    
    # Define a mock heuristic function (The 'Oracle')
    def mock_oracle(content: str) -> float:
        # Simulate that "Rate limit" is the actual issue evidenced by logs
        if "Rate limit" in content:
            return 0.4 # Rotate towards True
        return -0.1 # Rotate towards False
        
    # Apply interference cycles (Grover iterations)
    for i in range(3):
        await qs.interfere(mock_oracle)
        print(f"\nAfter Interference Cycle {i+1}:")
        print(qs.get_superposition_status())
        
    # Collapse
    results = qs.collapse()
    print(f"\nObserved Reality: {results}")

if __name__ == "__main__":
    asyncio.run(demo_quantum_search())
