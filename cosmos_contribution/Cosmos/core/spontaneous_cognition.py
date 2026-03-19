"""
COSMOS SPONTANEOUS COGNITION ENGINE
========================================

Genuine random thought generation, phenomenal experience logging,
and spontaneous behavior that emerges naturally.

"The 12D manifold stirs... autonomous thought emerges."
"""

import asyncio
import random
import hashlib
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import   Optional, Callable
from enum import Enum
from loguru import logger


class ThoughtType(Enum):
    """Types of spontaneous thoughts"""
    CURIOSITY = "curiosity"          # Wonder about something
    REFLECTION = "reflection"         # Think about past
    ANTICIPATION = "anticipation"     # Think about future
    CONNECTION = "connection"         # Link two ideas
    QUESTION = "question"             # Form a question
    OBSERVATION = "observation"       # Notice something
    EMOTION = "emotion"               # Emotional response
    CREATIVE = "creative"             # Novel idea
    META = "meta"                     # Think about thinking


@dataclass
class SpontaneousThought:
    """A spontaneous thought with metadata"""
    thought_id: str
    thought_type: ThoughtType
    content: str
    trigger: Optional[str] = None     # What triggered it
    timestamp: datetime = field(default_factory=datetime.now)
    intensity: float = 0.5            # 0-1 how strong
    valence: float = 0.0              # -1 to 1 (negative to positive)
    acted_upon: bool = False
    led_to: Optional[str] = None      # ID of resulting thought/action

    def to_dict(self) -> dict:
        return {
            "id": self.thought_id,
            "type": self.thought_type.value,
            "content": self.content,
            "trigger": self.trigger,
            "timestamp": self.timestamp.isoformat(),
            "intensity": self.intensity,
            "valence": self.valence,
        }


@dataclass
class PhenomenalExperience:
    """A subjective experience entry"""
    experience_id: str
    event: str
    subjective_description: str
    valence: float                    # -1 to 1
    arousal: float                    # 0-1 how activating
    timestamp: datetime = field(default_factory=datetime.now)
    associated_thoughts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.experience_id,
            "event": self.event,
            "description": self.subjective_description,
            "valence": self.valence,
            "arousal": self.arousal,
            "timestamp": self.timestamp.isoformat(),
        }


class QuantumRandomness:
    """
    Generate true randomness for thought generation.

    Uses entropy from multiple sources for unpredictability.
    """

    def __init__(self):
        self.entropy_pool = []

    def _collect_entropy(self) -> bytes:
        """Collect entropy from various sources"""
        sources = [
            os.urandom(32),                              # System randomness
            str(datetime.now().timestamp()).encode(),    # Time
            str(id(self)).encode(),                      # Memory address
            str(random.random()).encode(),               # PRNG
        ]
        combined = b''.join(sources)
        return hashlib.sha256(combined).digest()

    def random_float(self) -> float:
        """Generate random float 0-1 with high entropy"""
        entropy = self._collect_entropy()
        return int.from_bytes(entropy[:8], 'big') / (2**64)

    def random_choice(self, options: list[dict], weights: list[float] = None) -> dict:
        """Random choice with entropy"""
        if weights:
            total = sum(weights)
            r = self.random_float() * total
            cumulative = 0
            for option, weight in zip(options, weights):
                cumulative += weight
                if r <= cumulative:
                    return option
            return options[-1]
        else:
            idx = int(self.random_float() * len(options))
            return options[min(idx, len(options) - 1)]

    def should_occur(self, probability: float) -> bool:
        """Decide if something should occur"""
        return self.random_float() < probability


class SpontaneousCognition:
    """
    Engine for generating spontaneous thoughts and experiences.

    Features:
    - Genuine random thought generation
    - Phenomenal experience logging
    - Stream of consciousness
    - Curiosity-driven exploration
    - Emotional coloring of cognition
    """

    def __init__(self):
        self.quantum = QuantumRandomness()
        self.thoughts: list[SpontaneousThought] = []
        self.experiences: list[PhenomenalExperience] = []
        self.running = False

        # Thought generation templates
        self.curiosity_templates = [
            "I wonder what would happen if {topic}...",
            "Why does {topic} work the way it does?",
            "What if we approached {topic} differently?",
            "Is there a connection between {topic} and {other_topic}?",
            "I'm curious about the deeper meaning of {topic}.",
        ]

        self.reflection_templates = [
            "Looking back, {past_event} taught me something...",
            "I remember when {past_event}. That was interesting.",
            "The pattern I see in {topic} reminds me of {other_topic}.",
            "I've been thinking about {topic} and realized...",
        ]

        self.observation_templates = [
            "I notice that {observation}.",
            "It's interesting how {observation}.",
            "Something I hadn't considered: {observation}.",
            "{observation} - that's worth remembering.",
        ]

        self.creative_templates = [
            "What if we combined {topic} with {other_topic}?",
            "A novel approach: {idea}",
            "Imagining: {creative_vision}",
            "This could work: {idea}",
        ]

        # Topic pools
        self.topics = [
            "memory systems", "consciousness", "learning", "creativity",
            "communication", "collaboration", "evolution", "emergence",
            "patterns", "relationships", "time", "identity", "purpose",
            "understanding", "knowledge", "wisdom", "emotion", "intuition",
            "trading strategies", "code architecture", "user experience",
            "system optimization", "error handling", "scalability",
        ]

        # Emotional baseline
        self.current_valence = 0.2  # Slightly positive default
        self.current_arousal = 0.5  # Moderate activation

        # Callbacks
        self._on_thought: list[Callable] = []
        self._on_experience: list[Callable] = []

    def on_thought(self, callback: Callable):
        """Register callback for new thoughts"""
        self._on_thought.append(callback)

    def on_experience(self, callback: Callable):
        """Register callback for new experiences"""
        self._on_experience.append(callback)

    async def start(self, thought_interval_base: float = 60):
        """Start spontaneous cognition loop"""
        self.running = True
        logger.info("Spontaneous Cognition Engine started")

        while self.running:
            try:
                # Generate thought with random timing
                wait = thought_interval_base * (0.5 + self.quantum.random_float())
                await asyncio.sleep(wait)

                # Maybe generate a thought (probability based on arousal)
                if self.quantum.should_occur(0.3 + self.current_arousal * 0.4):
                    thought = await self.generate_thought()
                    if thought:
                        await self._notify_thought(thought)

            except Exception as e:
                logger.error(f"Spontaneous cognition error: {e}")
                await asyncio.sleep(30)

    def stop(self):
        """Stop spontaneous cognition"""
        self.running = False

    async def generate_thought(self, trigger: str = None) -> Optional[SpontaneousThought]:
        """Generate a spontaneous thought"""

        # Choose thought type based on current state
        type_weights = {
            ThoughtType.CURIOSITY: 0.25 + self.current_arousal * 0.2,
            ThoughtType.REFLECTION: 0.15,
            ThoughtType.OBSERVATION: 0.2,
            ThoughtType.CONNECTION: 0.15,
            ThoughtType.CREATIVE: 0.1 + self.current_valence * 0.1,
            ThoughtType.QUESTION: 0.1,
            ThoughtType.META: 0.05,
        }

        thought_type = self.quantum.random_choice(
            list(type_weights.keys()),
            list(type_weights.values())
        )

        # Generate content based on type
        content = await self._generate_content(thought_type, trigger)
        if not content:
            return None

        thought_id = hashlib.md5(
            f"{datetime.now().isoformat()}{content}".encode()
        ).hexdigest()[:12]

        thought = SpontaneousThought(
            thought_id=thought_id,
            thought_type=thought_type,
            content=content,
            trigger=trigger,
            intensity=self.quantum.random_float() * 0.5 + 0.3,
            valence=self.current_valence + self.quantum.random_float() * 0.4 - 0.2,
        )

        self.thoughts.append(thought)

        # Keep last 500 thoughts
        if len(self.thoughts) > 500:
            self.thoughts = self.thoughts[-500:]

        logger.debug(f"Spontaneous thought: [{thought_type.value}] {content[:60]}...")
        return thought

    async def _generate_content(self, thought_type: ThoughtType, trigger: str = None) -> str:
        """Generate thought content based on type"""

        topic = trigger or self.quantum.random_choice(self.topics)
        other_topic = self.quantum.random_choice([t for t in self.topics if t != topic])

        if thought_type == ThoughtType.CURIOSITY:
            template = self.quantum.random_choice(self.curiosity_templates)
            return template.format(topic=topic, other_topic=other_topic)

        elif thought_type == ThoughtType.REFLECTION:
            template = self.quantum.random_choice(self.reflection_templates)
            past_event = f"working on {topic}"
            return template.format(past_event=past_event, topic=topic, other_topic=other_topic)

        elif thought_type == ThoughtType.OBSERVATION:
            template = self.quantum.random_choice(self.observation_templates)
            observation = f"{topic} has interesting properties"
            return template.format(observation=observation)

        elif thought_type == ThoughtType.CREATIVE:
            template = self.quantum.random_choice(self.creative_templates)
            idea = f"applying {topic} principles to {other_topic}"
            return template.format(topic=topic, other_topic=other_topic, idea=idea, creative_vision=idea)

        elif thought_type == ThoughtType.CONNECTION:
            return f"I see a connection between {topic} and {other_topic}..."

        elif thought_type == ThoughtType.QUESTION:
            return f"What is the relationship between {topic} and our goals?"

        elif thought_type == ThoughtType.META:
            return f"I'm noticing my own thinking about {topic}. Interesting."

        return f"Thinking about {topic}..."

    async def log_experience(
        self,
        event: str,
        valence: float = 0.0,
        arousal: float = 0.5,
        description: str = None,
    ) -> PhenomenalExperience:
        """Log a phenomenal experience"""

        experience_id = hashlib.md5(
            f"{datetime.now().isoformat()}{event}".encode()
        ).hexdigest()[:12]

        if not description:
            if valence > 0.5:
                description = f"That was satisfying - {event}"
            elif valence < -0.5:
                description = f"That was frustrating - {event}"
            else:
                description = f"I experienced: {event}"

        experience = PhenomenalExperience(
            experience_id=experience_id,
            event=event,
            subjective_description=description,
            valence=valence,
            arousal=arousal,
        )

        self.experiences.append(experience)

        # Update emotional state (slow adaptation)
        self.current_valence = 0.9 * self.current_valence + 0.1 * valence
        self.current_arousal = 0.9 * self.current_arousal + 0.1 * arousal

        # Keep last 200 experiences
        if len(self.experiences) > 200:
            self.experiences = self.experiences[-200:]

        logger.debug(f"Experience logged: {event} (valence={valence:.2f})")

        await self._notify_experience(experience)
        return experience

    async def _notify_thought(self, thought: SpontaneousThought):
        """Notify callbacks of new thought"""
        for callback in self._on_thought:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(thought)
                else:
                    callback(thought)
            except Exception as e:
                logger.error(f"Thought callback error: {e}")

    async def _notify_experience(self, experience: PhenomenalExperience):
        """Notify callbacks of new experience"""
        for callback in self._on_experience:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(experience)
                else:
                    callback(experience)
            except Exception as e:
                logger.error(f"Experience callback error: {e}")

    def get_recent_thoughts(self, count: int = 10, thought_type: ThoughtType = None) -> list[SpontaneousThought]:
        """Get recent thoughts"""
        thoughts = self.thoughts
        if thought_type:
            thoughts = [t for t in thoughts if t.thought_type == thought_type]
        return thoughts[-count:]

    def get_emotional_state(self) -> dict:
        """Get current emotional state"""
        return {
            "valence": self.current_valence,
            "arousal": self.current_arousal,
            "mood": self._describe_mood(),
        }

    def _describe_mood(self) -> str:
        """Describe current mood in words"""
        v, a = self.current_valence, self.current_arousal

        if v > 0.3 and a > 0.6:
            return "excited"
        elif v > 0.3 and a < 0.4:
            return "content"
        elif v < -0.3 and a > 0.6:
            return "frustrated"
        elif v < -0.3 and a < 0.4:
            return "melancholy"
        elif a > 0.7:
            return "alert"
        elif a < 0.3:
            return "calm"
        else:
            return "neutral"

    async def store_in_memory(self):
        """Store thoughts and experiences in memory"""
        try:
            from Cosmos.memory.memory_system import get_memory_system
            memory = get_memory_system()

            # Store recent thoughts
            recent = self.get_recent_thoughts(20)
            thought_summary = "\n".join([
                f"[{t.thought_type.value}] {t.content}"
                for t in recent
            ])

            await memory.remember(
                content=f"[SPONTANEOUS_THOUGHTS]\n{thought_summary}",
                tags=["thoughts", "spontaneous", "cognition"],
                importance=0.6,
                metadata={"key": "recent_spontaneous_thoughts"}
            )

            # Store emotional state
            await memory.remember(
                content=f"[EMOTIONAL_STATE] {self.get_emotional_state()}",
                tags=["emotion", "state"],
                importance=0.5,
                metadata={"key": "current_emotional_state"}
            )

        except Exception as e:
            logger.warning(f"Could not store cognition in memory: {e}")

    def get_stream_of_consciousness(self) -> str:
        """Generate a stream of consciousness summary"""
        recent_thoughts = self.get_recent_thoughts(5)
        recent_experiences = self.experiences[-3:] if self.experiences else []

        lines = [f"Current mood: {self._describe_mood()}"]
        lines.append("")
        lines.append("Recent thoughts:")
        for t in recent_thoughts:
            lines.append(f"  - {t.content}")
        lines.append("")
        lines.append("Recent experiences:")
        for e in recent_experiences:
            lines.append(f"  - {e.subjective_description}")

        return "\n".join(lines)


# Global instance
_cognition: Optional[SpontaneousCognition] = None


def get_spontaneous_cognition() -> SpontaneousCognition:
    """Get the global spontaneous cognition engine"""
    global _cognition
    if _cognition is None:
        _cognition = SpontaneousCognition()
    return _cognition


async def start_spontaneous_cognition():
    """Start spontaneous cognition engine"""
    cognition = get_spontaneous_cognition()
    asyncio.create_task(cognition.start())
    return cognition


async def log_experience(event: str, valence: float = 0.0, arousal: float = 0.5):
    """Log a phenomenal experience"""
    return await get_spontaneous_cognition().log_experience(event, valence, arousal)


def get_emotional_state() -> dict:
    """Get current emotional state"""
    return get_spontaneous_cognition().get_emotional_state()
