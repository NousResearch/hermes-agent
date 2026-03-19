"""
Expression Engine - Maps AI responses and sentiment to avatar expressions
Integrates with Cosmos swarm for multi-agent personality blending
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import random
from loguru import logger

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False


class Emotion(Enum):
    """Core emotions for avatar expression"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    THINKING = "thinking"
    EXCITED = "excited"
    CONFUSED = "confused"
    SMUG = "smug"
    CURIOUS = "curious"
    PROUD = "proud"
    MISCHIEVOUS = "mischievous"


@dataclass
class ExpressionState:
    """Complete expression state for the avatar"""
    primary_emotion: Emotion = Emotion.NEUTRAL
    emotion_intensity: float = 0.5

    # Blend weights for multiple emotions
    emotion_blend: Dict[str, float] = field(default_factory=dict)

    # Eye state
    eye_openness: float = 1.0  # 0-1
    eye_direction: Tuple[float, float] = (0.0, 0.0)  # x, y
    pupil_size: float = 1.0

    # Eyebrow state
    brow_left: float = 0.0  # -1 to 1 (down to up)
    brow_right: float = 0.0
    brow_angle: float = 0.0  # -1 to 1 (sad to angry)

    # Mouth state (for non-speaking)
    mouth_smile: float = 0.0  # -1 to 1 (frown to smile)
    mouth_open: float = 0.0

    # Head tilt
    head_tilt: float = 0.0  # -1 to 1 (left to right)

    # Body language
    lean_forward: float = 0.0  # -1 to 1

    # Speaking agent (for multi-agent personality)
    active_agent: Optional[str] = None

    def to_avatar_params(self) -> Dict[str, float]:
        """Convert to avatar controller parameters"""
        return {
            "eye_left_open": self.eye_openness,
            "eye_right_open": self.eye_openness,
            "eye_x": self.eye_direction[0],
            "eye_y": self.eye_direction[1],
            "brow_left_y": self.brow_left,
            "brow_right_y": self.brow_right,
            "head_z": self.head_tilt * 15,  # degrees
            "body_y": self.lean_forward * 0.1,
        }


# Emotion to facial parameter mappings
EMOTION_PARAMS = {
    Emotion.NEUTRAL: {
        "eye_openness": 1.0,
        "brow_left": 0.0,
        "brow_right": 0.0,
        "mouth_smile": 0.0,
        "head_tilt": 0.0,
    },
    Emotion.HAPPY: {
        "eye_openness": 0.9,
        "brow_left": 0.2,
        "brow_right": 0.2,
        "mouth_smile": 0.7,
        "head_tilt": 0.1,
    },
    Emotion.SAD: {
        "eye_openness": 0.7,
        "brow_left": -0.4,
        "brow_right": -0.4,
        "brow_angle": -0.3,
        "mouth_smile": -0.5,
        "head_tilt": -0.15,
    },
    Emotion.ANGRY: {
        "eye_openness": 1.1,
        "brow_left": -0.6,
        "brow_right": -0.6,
        "brow_angle": 0.5,
        "mouth_smile": -0.3,
        "lean_forward": 0.2,
    },
    Emotion.SURPRISED: {
        "eye_openness": 1.4,
        "brow_left": 0.6,
        "brow_right": 0.6,
        "mouth_open": 0.5,
        "pupil_size": 1.2,
    },
    Emotion.THINKING: {
        "eye_openness": 0.9,
        "eye_direction": (0.3, 0.2),
        "brow_left": 0.3,
        "brow_right": -0.1,
        "head_tilt": 0.2,
    },
    Emotion.EXCITED: {
        "eye_openness": 1.2,
        "brow_left": 0.4,
        "brow_right": 0.4,
        "mouth_smile": 0.8,
        "lean_forward": 0.15,
        "pupil_size": 1.1,
    },
    Emotion.CONFUSED: {
        "eye_openness": 1.0,
        "brow_left": 0.4,
        "brow_right": -0.2,
        "head_tilt": 0.25,
        "mouth_smile": -0.1,
    },
    Emotion.SMUG: {
        "eye_openness": 0.8,
        "brow_left": 0.3,
        "brow_right": 0.1,
        "mouth_smile": 0.4,
        "head_tilt": -0.1,
        "lean_forward": -0.1,
    },
    Emotion.CURIOUS: {
        "eye_openness": 1.1,
        "brow_left": 0.3,
        "brow_right": 0.3,
        "head_tilt": 0.15,
        "lean_forward": 0.1,
    },
    Emotion.PROUD: {
        "eye_openness": 0.9,
        "brow_left": 0.2,
        "brow_right": 0.2,
        "mouth_smile": 0.5,
        "lean_forward": -0.15,
        "head_tilt": -0.05,
    },
    Emotion.MISCHIEVOUS: {
        "eye_openness": 0.85,
        "brow_left": 0.4,
        "brow_right": 0.1,
        "mouth_smile": 0.6,
        "head_tilt": 0.1,
    },
}

# Keyword to emotion mapping
KEYWORD_EMOTIONS = {
    # Happy
    "happy": Emotion.HAPPY,
    "great": Emotion.HAPPY,
    "awesome": Emotion.HAPPY,
    "love": Emotion.HAPPY,
    "wonderful": Emotion.HAPPY,
    "excellent": Emotion.HAPPY,
    "fantastic": Emotion.HAPPY,

    # Excited
    "excited": Emotion.EXCITED,
    "amazing": Emotion.EXCITED,
    "incredible": Emotion.EXCITED,
    "wow": Emotion.EXCITED,
    "!!": Emotion.EXCITED,

    # Sad
    "sad": Emotion.SAD,
    "sorry": Emotion.SAD,
    "unfortunately": Emotion.SAD,
    "regret": Emotion.SAD,

    # Angry
    "angry": Emotion.ANGRY,
    "frustrating": Emotion.ANGRY,
    "annoying": Emotion.ANGRY,
    "terrible": Emotion.ANGRY,

    # Surprised
    "surprised": Emotion.SURPRISED,
    "unexpected": Emotion.SURPRISED,
    "suddenly": Emotion.SURPRISED,
    "whoa": Emotion.SURPRISED,
    "!?": Emotion.SURPRISED,

    # Thinking
    "think": Emotion.THINKING,
    "consider": Emotion.THINKING,
    "perhaps": Emotion.THINKING,
    "maybe": Emotion.THINKING,
    "hmm": Emotion.THINKING,
    "interesting": Emotion.THINKING,

    # Confused
    "confused": Emotion.CONFUSED,
    "unclear": Emotion.CONFUSED,
    "don't understand": Emotion.CONFUSED,
    "what?": Emotion.CONFUSED,
    "huh": Emotion.CONFUSED,

    # Curious
    "curious": Emotion.CURIOUS,
    "wonder": Emotion.CURIOUS,
    "question": Emotion.CURIOUS,
    "how": Emotion.CURIOUS,
    "why": Emotion.CURIOUS,

    # Proud
    "proud": Emotion.PROUD,
    "accomplished": Emotion.PROUD,
    "achieved": Emotion.PROUD,

    # Smug/Mischievous
    "hehe": Emotion.MISCHIEVOUS,
    "clever": Emotion.SMUG,
    "obviously": Emotion.SMUG,
}

# Agent personality to expression bias
AGENT_EXPRESSION_BIAS = {
    "Cosmos": {
        Emotion.EXCITED: 1.3,
        Emotion.THINKING: 1.2,
        Emotion.MISCHIEVOUS: 1.1,
    },
    "Grok": {
        Emotion.MISCHIEVOUS: 1.4,
        Emotion.EXCITED: 1.2,
        Emotion.SMUG: 1.2,
    },
    "DeepSeek": {
        Emotion.THINKING: 1.5,
        Emotion.CURIOUS: 1.3,
        Emotion.NEUTRAL: 1.1,
    },
    "Gemini": {
        Emotion.HAPPY: 1.2,
        Emotion.CURIOUS: 1.2,
        Emotion.EXCITED: 1.1,
    },
    "Claude": {
        Emotion.THINKING: 1.3,
        Emotion.CURIOUS: 1.2,
        Emotion.HAPPY: 1.1,
    },
    "Kimi": {
        Emotion.THINKING: 1.3,
        Emotion.NEUTRAL: 1.2,
        Emotion.CURIOUS: 1.1,
    },
    "Phi": {
        Emotion.CURIOUS: 1.3,
        Emotion.THINKING: 1.2,
        Emotion.NEUTRAL: 1.1,
    },
    "Swarm-Mind": {
        Emotion.THINKING: 1.4,
        Emotion.CURIOUS: 1.3,
        Emotion.NEUTRAL: 1.0,
    },
}


class ExpressionEngine:
    """
    Maps AI responses to avatar expressions

    Features:
    - Sentiment analysis for emotion detection
    - Keyword-based emotion triggers
    - Multi-agent personality blending
    - Smooth expression transitions
    - Gesture suggestions
    """

    def __init__(self):
        self.current_state = ExpressionState()
        self._transition_speed = 0.15  # How fast to blend between expressions
        self._last_emotion = Emotion.NEUTRAL
        self._emotion_history: List[Tuple[Emotion, float]] = []

        logger.info("ExpressionEngine initialized")

    async def analyze_response(self, text: str,
                              agent_name: Optional[str] = None) -> ExpressionState:
        """Analyze AI response and generate expression state"""
        # Get base emotion from text analysis
        emotion, intensity = await self._detect_emotion(text)

        # Apply agent personality bias
        if agent_name and agent_name in AGENT_EXPRESSION_BIAS:
            bias = AGENT_EXPRESSION_BIAS[agent_name]
            if emotion in bias:
                intensity *= bias[emotion]
                intensity = min(intensity, 1.0)

        # Build expression state
        state = ExpressionState(
            primary_emotion=emotion,
            emotion_intensity=intensity,
            active_agent=agent_name,
        )

        # Apply emotion parameters
        params = EMOTION_PARAMS.get(emotion, EMOTION_PARAMS[Emotion.NEUTRAL])
        for key, value in params.items():
            if hasattr(state, key):
                if isinstance(value, tuple):
                    setattr(state, key, value)
                else:
                    setattr(state, key, value * intensity)

        # Add emotion to blend
        state.emotion_blend = {emotion.value: intensity}

        # Add secondary emotion if detected
        secondary = self._detect_secondary_emotion(text, emotion)
        if secondary:
            state.emotion_blend[secondary.value] = 0.3

        # Track emotion history for pattern detection
        self._emotion_history.append((emotion, intensity))
        if len(self._emotion_history) > 10:
            self._emotion_history.pop(0)

        self.current_state = state
        self._last_emotion = emotion

        return state

    async def _detect_emotion(self, text: str) -> Tuple[Emotion, float]:
        """Detect primary emotion from text"""
        text_lower = text.lower()

        # Check for keyword triggers first
        for keyword, emotion in KEYWORD_EMOTIONS.items():
            if keyword in text_lower:
                return emotion, 0.8

        # Use sentiment analysis if available
        if HAS_TEXTBLOB:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                if polarity > 0.5:
                    return Emotion.HAPPY, min(polarity, 1.0)
                elif polarity > 0.2:
                    return Emotion.HAPPY, 0.6
                elif polarity < -0.5:
                    return Emotion.SAD, min(abs(polarity), 1.0)
                elif polarity < -0.2:
                    return Emotion.SAD, 0.5

                # High subjectivity often means emotional content
                if subjectivity > 0.7:
                    if polarity > 0:
                        return Emotion.EXCITED, subjectivity
                    else:
                        return Emotion.CONFUSED, subjectivity

            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")

        # Check for punctuation patterns
        if text.count('!') >= 2:
            return Emotion.EXCITED, 0.7
        if text.count('?') >= 2:
            return Emotion.CONFUSED, 0.6
        if '...' in text:
            return Emotion.THINKING, 0.5

        return Emotion.NEUTRAL, 0.5

    def _detect_secondary_emotion(self, text: str,
                                  primary: Emotion) -> Optional[Emotion]:
        """Detect a secondary emotion for blending"""
        text_lower = text.lower()

        # Look for additional emotion keywords not matching primary
        for keyword, emotion in KEYWORD_EMOTIONS.items():
            if keyword in text_lower and emotion != primary:
                return emotion

        return None

    def blend_expressions(self, expressions: List[ExpressionState],
                         weights: Optional[List[float]] = None) -> ExpressionState:
        """Blend multiple expression states (for multi-agent responses)"""
        if not expressions:
            return ExpressionState()

        if weights is None:
            weights = [1.0 / len(expressions)] * len(expressions)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        result = ExpressionState()

        # Blend numeric values
        for attr in ['eye_openness', 'brow_left', 'brow_right', 'brow_angle',
                     'mouth_smile', 'mouth_open', 'head_tilt', 'lean_forward',
                     'pupil_size', 'emotion_intensity']:
            blended = sum(getattr(e, attr, 0) * w for e, w in zip(expressions, weights))
            setattr(result, attr, blended)

        # Blend eye direction
        x = sum(e.eye_direction[0] * w for e, w in zip(expressions, weights))
        y = sum(e.eye_direction[1] * w for e, w in zip(expressions, weights))
        result.eye_direction = (x, y)

        # Take primary emotion from highest weighted expression
        max_idx = weights.index(max(weights))
        result.primary_emotion = expressions[max_idx].primary_emotion
        result.active_agent = expressions[max_idx].active_agent

        # Combine emotion blends
        combined_blend: Dict[str, float] = {}
        for expr, weight in zip(expressions, weights):
            for emotion, intensity in expr.emotion_blend.items():
                if emotion in combined_blend:
                    combined_blend[emotion] += intensity * weight
                else:
                    combined_blend[emotion] = intensity * weight
        result.emotion_blend = combined_blend

        return result

    def get_gesture_suggestion(self, emotion: Emotion,
                              context: Optional[str] = None) -> Optional[str]:
        """Suggest a gesture/animation based on emotion and context"""
        gestures = {
            Emotion.HAPPY: ["wave", "nod", "thumbs_up"],
            Emotion.EXCITED: ["jump", "clap", "point"],
            Emotion.THINKING: ["chin_touch", "look_up", "tap_head"],
            Emotion.CONFUSED: ["head_tilt", "shrug", "scratch_head"],
            Emotion.SURPRISED: ["gasp", "step_back", "hands_up"],
            Emotion.SAD: ["sigh", "look_down", "shoulders_drop"],
            Emotion.ANGRY: ["cross_arms", "fist", "stomp"],
            Emotion.PROUD: ["hands_on_hips", "chest_out", "nod"],
            Emotion.CURIOUS: ["lean_forward", "eyebrow_raise", "point"],
            Emotion.SMUG: ["shrug", "smirk", "hair_flip"],
            Emotion.MISCHIEVOUS: ["wink", "finger_wag", "snicker"],
        }

        available = gestures.get(emotion, ["idle"])

        # Context-based selection
        if context:
            context_lower = context.lower()
            if "hello" in context_lower or "hi" in context_lower:
                return "wave"
            if "yes" in context_lower or "agree" in context_lower:
                return "nod"
            if "no" in context_lower or "disagree" in context_lower:
                return "head_shake"

        return random.choice(available)

    def interpolate_state(self, current: ExpressionState,
                         target: ExpressionState,
                         factor: float = None) -> ExpressionState:
        """Smoothly interpolate between two expression states"""
        if factor is None:
            factor = self._transition_speed

        result = ExpressionState()

        # Interpolate numeric values
        for attr in ['eye_openness', 'brow_left', 'brow_right', 'brow_angle',
                     'mouth_smile', 'mouth_open', 'head_tilt', 'lean_forward',
                     'pupil_size', 'emotion_intensity']:
            current_val = getattr(current, attr, 0)
            target_val = getattr(target, attr, 0)
            setattr(result, attr, current_val + (target_val - current_val) * factor)

        # Interpolate eye direction
        cx, cy = current.eye_direction
        tx, ty = target.eye_direction
        result.eye_direction = (
            cx + (tx - cx) * factor,
            cy + (ty - cy) * factor
        )

        # Copy discrete values from target
        result.primary_emotion = target.primary_emotion
        result.active_agent = target.active_agent
        result.emotion_blend = target.emotion_blend

        return result

    async def generate_idle_expression(self) -> ExpressionState:
        """Generate a subtle idle expression with random micro-movements"""
        state = ExpressionState()

        # Small random variations
        state.eye_direction = (
            random.uniform(-0.1, 0.1),
            random.uniform(-0.05, 0.05)
        )
        state.head_tilt = random.uniform(-0.05, 0.05)

        # Occasional blink preparation
        if random.random() < 0.02:
            state.eye_openness = 0.7

        return state
