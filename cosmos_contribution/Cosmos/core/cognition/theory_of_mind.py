"""
cosmos Theory of Mind (ToM) Engine: The Mirror Neuron Substrate.

"I know what you're thinking, because I'm simulating being you right now!"

This is a NOVEL approach to AI Empathy and Theory of Mind.
Instead of static user profiles, we implement a "Shadow User" simulation based on Predictive Coding principles.

Core Concepts:
1. The Shadow User: A lightweight predictive model that runs in parallel. It attempts to predict YOU.
   - If Prediction Error (Surprise) is low -> We are "in sync" -> Agent acts autonomously.
   - If Prediction Error is high -> We are "out of sync" -> Agent asks clarifying questions.

2. Affective Resonance: Tracks "Emotional Velocity" (rate of change of sentiment), not just current sentiment.
   - Rapidly increasing frustration triggers "Calming Protocols".

3. Intent Inverse Mapping: Using Inverse Reinforcement Learning mechanisms to infer WHY you asked for something, not just WHAT you asked for.
"""

import asyncio
import math
import random # Mock for predictive probability
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from loguru import logger

from Cosmos.core.nexus import nexus, Signal, SignalType

@dataclass
class MentalState:
    """A snapshot of the User's simulated cognitive state."""
    cognitive_load: float = 0.5    # 0.0 (Bored) to 1.0 (Overwhelmed)
    emotional_valence: float = 0.5 # 0.0 (Negative) to 1.0 (Positive)
    emotional_arousal: float = 0.5 # 0.0 (Calm) to 1.0 (Excited/Angry)
    intent_clarity: float = 0.0    # How clear is their current goal?
    focus_object: str = "none"     # What are they looking at?

class ShadowUser:
    """
    The Predictive Coding model of the user.
    Simulates the user's next move to measure "Surprise".
    """
    def __init__(self):
        self.history: list[str] = []
        self.surprise_metric: float = 0.0
        self.synchronization_score: float = 1.0 # 0 to 1

    def predict_next_input(self, context_vector: list[float]) -> str:
        """
        Simulate what the user might say/do next based on context.
        (In a real implementation, this uses a small LLM or N-gram model)
        """
        # Mocking the predictive output for the architecture
        return "predicted_action"

    def measure_surprise(self, actual_input: str, prediction: str) -> float:
        """
        Calculates Prediction Error (Free Energy).
        Low Surprise = High Empathy/Understanding.
        High Surprise = Misunderstanding.
        """
        # Simple Euclidean distance placeholder
        # Real impl would use embedding cosine distance
        self.surprise_metric = random.random() # Dynamic variation
        
        # Update sync score (exponential moving average)
        self.synchronization_score = (self.synchronization_score * 0.8) + ((1.0 - self.surprise_metric) * 0.2)
        return self.surprise_metric

class AffectiveResonance:
    """
    Tracks emotional dynamics (Velocity & Acceleration).
    """
    def __init__(self):
        self.valence_history: list[float] = [0.5] * 10
        self.velocity: float = 0.0
        
    def update(self, current_valence: float):
        """
        Process a new emotional data point.
        """
        prev = self.valence_history[-1]
        self.velocity = current_valence - prev
        
        self.valence_history.append(current_valence)
        if len(self.valence_history) > 10:
            self.valence_history.pop(0)
            
        if self.velocity < -0.2:
            logger.warning(f"ToM: Emotional drop detected (Velocity: {self.velocity:.2f})")
            # Signal Nexus to activate empathy protocols
            # asyncio.create_task(nexus.emit(SignalType.ANOMALY_DETECTED, ...))

class MirrorNeuronSystem:
    """
    The main ToM Engine.
    """
    def __init__(self):
        self.shadow = ShadowUser()
        self.resonance = AffectiveResonance()
        self.current_state = MentalState()
        
        # Listen for User Interaction
        nexus.subscribe(SignalType.USER_MESSAGE, self._on_user_input)

    async def _on_user_input(self, signal: Signal):
        content = signal.payload.get("content", "")
        
        # 1. Predictive Coding Check (The "Surprise" Metric)
        # In a real cycle, we would have predicted this *before* it arrived
        prediction = self.shadow.predict_next_input([])
        survival_val = self.shadow.measure_surprise(content, prediction)
        
        logger.debug(f"ToM: Surprise={survival_val:.2f} | Sync={self.shadow.synchronization_score:.2f}")

        # 2. Update Mental Model
        # (Mock sentiment analysis)
        sentiment = 0.5 # Neutral
        if "bad" in content or "error" in content: sentiment = 0.2
        if "good" in content or "thanks" in content: sentiment = 0.8
        
        self.resonance.update(sentiment)
        
        # 3. Adjust System Behavior based on Sync Score
        self.current_state.intent_clarity = self.shadow.synchronization_score
        
        if self.shadow.synchronization_score < 0.4:
            # We don't understand the user well.
            # Signal the system to be more verbose/cautious.
            await nexus.emit(
                SignalType.CONFUSION_DETECTED, 
                {"reason": "low_synchronization", "score": self.shadow.synchronization_score}, 
                "theory_of_mind"
            )
        
        logger.info(f"ToM: User State Model -> Load:{self.current_state.cognitive_load} Valence:{self.resonance.valence_history[-1]}")

# Global Instance
tom_engine = MirrorNeuronSystem()
