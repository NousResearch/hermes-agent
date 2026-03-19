
"""
Uncertainty Injector (Human Intuition Module)
=============================================
Adds stochastic "human" elements to AI decision making:
- Doubt (questioning own conclusions)
- Curiosity (random tangent exploration)
- Intuition (gut feelings based on fuzzy logic)
"""

import random
import time
from dataclasses import dataclass
from typing import  Optional

@dataclass
class IntuitionState:
    doubt_level: float = 0.0          # 0.0 to 1.0
    curiosity_vector: float = 0.5     # 0.0 to 1.0
    gut_feeling: str = "neutral"      # positive, negative, neutral
    last_hunch_time: float = 0.0

class UncertaintyInjector:
    """
    Injects controlled chaos and doubt into the swarm.
    Prevents "AI Over-Confidence" by simulating human hesitation.
    """
    
    def __init__(self, base_doubt: float = 0.1):
        self.state = IntuitionState()
        self.base_doubt = base_doubt
        self.random_musings = [
            "Wait, is that actually right?",
            "I feel like I'm missing something obvious.",
            "Let me double check that logic.",
            "This seems too simple.",
            "What if we looked at this backwards?",
            "Just a hunch, but maybe...",
        ]

    def evaluate_hunch(self, confidence: float, context_complexity: float) -> Optional[str]:
        """
        Decide if a "hunch" should interrupt the flow.
        """
        self.state.last_hunch_time = time.time()
        
        # Human Intuition Formula:
        # High complexity + High confidence = Suspicious (Trigger Doubt)
        # Low complexity + Low confidence = Confusion (Trigger Curiosity)
        
        trigger_threshold = 0.7 + (random.random() * 0.2)
        
        if confidence > 0.95 and context_complexity > 0.8:
            # Too confident about something complex
            self.state.doubt_level = 0.8
            return self._get_random_musing("doubt")
            
        if confidence < 0.4 and context_complexity < 0.3:
            # Confused about something simple
            self.state.curiosity_vector += 0.2
            return self._get_random_musing("curiosity")
            
        # Random "Gut Check"
        if random.random() < 0.05:
            return "Just a gut feeling, but we should verify this with the 12D logical core."
            
        return None

    def _get_random_musing(self, type_key: str) -> str:
        """Return a random human-like thought."""
        return random.choice(self.random_musings)

    def modulate_confidence(self, raw_confidence: float) -> float:
        """
        Apply human doubt to reduce over-confidence.
        """
        # If doubt is high, slash confidence
        if self.state.doubt_level > 0.5:
            return raw_confidence * (1.0 - (self.state.doubt_level * 0.5))
        return raw_confidence
