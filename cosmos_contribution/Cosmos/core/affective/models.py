from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime

class EmotionCategory(Enum):
    NEUTRAL = "neutral"
    FOCUS = "focus"
    FLOW = "flow"
    FRUSTRATION = "frustration"
    EXHAUSTION = "exhaustion"
    ANXIETY = "anxiety"
    EXCITEMENT = "excitement"

@dataclass
class AffectiveState:
    timestamp: datetime = field(default_factory=datetime.now)
    valence: float = 0.0  # -1.0 to 1.0 (Negative to Positive)
    arousal: float = 0.0  # 0.0 to 1.0 (Calm to Excited)
    dominance: float = 0.0 # 0.0 to 1.0 (Submissive to Dominant) - Optional PAD model
    primary_emotion: EmotionCategory = EmotionCategory.NEUTRAL
    confidence: float = 1.0
    source: str = "manual" # e.g., "bio_interface", "text_analysis", "manual"

@dataclass
class SystemAction:
    action_id: str
    priority_delta: float = 0.0 # Change in priority
    description: str = ""
    parameters: dict[str, str] = field(default_factory=dict)

class SystemPriority(Enum):
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1
    DORMANT = 0
