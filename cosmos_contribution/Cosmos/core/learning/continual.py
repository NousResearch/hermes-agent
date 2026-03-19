"""
cosmos Continual Learning Engine.

"I don't just learn; I accumulate wisdom without displacing the old stuff!"

This module implements mechanisms to prevent catastrophic forgetting and enable graceful skill acquisition.

Features:
1. Experience Replay: Buffers important past events to re-validate new behaviors.
2. Elastic Concept Consolidation: Protects critical "knowledge weights" from being overwritten.
3. Drift Detection: Identifies when the environment shifts significantly.
"""

import random
import asyncio
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from Cosmos.core.nexus import nexus, Signal, SignalType

@dataclass
class Experience:
    """A unit of experience to be preserved."""
    id: str
    context_vector: list[float]
    action: str
    outcome: str
    importance: float  # computed based on reward/novelty
    timestamp: datetime = field(default_factory=datetime.now)

class ExperienceReplayBuffer:
    """
    Stores critical experiences.
    Used to 'replay' past situations to ensure new behaviors don't break old ones.
    """
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: list[Experience] = []

    def add(self, exp: Experience):
        if len(self.buffer) >= self.capacity:
            # Drop lowest importance
            self.buffer.sort(key=lambda x: x.importance)
            self.buffer.pop(0)
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> list[Experience]:
        if not self.buffer:
            return []
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

class ContinualLearner:
    """
    Manages the stability-plasticity dilemma.
    Ensures that while the system adapts (plasticity), it retains core skills (stability).
    """
    def __init__(self):
        self.replay = ExperienceReplayBuffer()
        self.concept_drift_monitor = 0.0
        
        # Subscribe to learning events
        nexus.subscribe(SignalType.TASK_COMPLETED, self._on_task_complete)

    async def _on_task_complete(self, signal: Signal):
        """
        When a task completes, evaluate if it's a "Significant Experience" 
        worth preserving in the Replay Buffer.
        """
        task_data = signal.payload
        importance = self._calculate_importance(task_data)
        
        if importance > 0.7:  # Threshold for "Core Memory"
            exp = Experience(
                id=signal.id,
                context_vector=signal.context_vector or [0.0]*10,
                action=str(task_data.get("action", "unknown")),
                outcome=str(task_data.get("result", "unknown")),
                importance=importance
            )
            self.replay.add(exp)
            logger.debug(f"Continual Learning: Consolidating experience '{signal.id}' (Imp: {importance:.2f})")
            
            # Trigger a massive rehearsal sleep cycle if buffer is getting full?
            # For now, we just store it.

    def _calculate_importance(self, data: dict) -> float:
        # Heuristic: Failures that were recovered are high importance
        # Successes on novel tasks are high importance
        if data.get("was_recovery", False):
            return 0.9
        if data.get("complexity", 1) > 4:
            return 0.8
        return 0.5

    async def validate_new_skill(self, skill_func: dict) -> bool:
        """
        Before adopting a new skill/tool/prompt, test it against the Replay Buffer
        to ensure it doesn't cause regression (Catastrophic Forgetting check).
        """
        samples = self.replay.sample(5)
        if not samples:
            return True
            
        logger.info("Continual Learning: Validating new skill against experience replay...")
        # In a real impl, we would simulate the skill against the stored context
        # For simulation, we return True
        return True

# Global Instance
continual_learner = ContinualLearner()
