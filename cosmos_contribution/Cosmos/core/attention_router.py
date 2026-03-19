"""
Advanced Attention Routing - Autonomous Improvement #5 by Claude Sonnet 4.5

PROBLEM: Simple keyword matching doesn't capture conversation context
SOLUTION: Topic modeling + expertise tracking + context history

WHO SHOULD RESPOND TO WHAT:
- Philosophy → Kimi (Eastern philosophy expert)
- Analysis → DeepSeek (pattern recognition)
- Creative → Phi (novel perspectives)
- Trading → Cosmos (domain knowledge)
- Synthesis → Swarm-Mind (connects ideas)
- Code/Technical → Claude (programming)
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict
from loguru import logger


@dataclass
class Topic:
    """A conversation topic."""
    name: str
    keywords: Set[str]
    weight: float = 1.0
    last_discussed: Optional[datetime] = None


@dataclass
class SpeakerExpertise:
    """Track speaker expertise in various topics."""
    name: str
    topic_scores: dict[str, float] = field(default_factory=dict)
    participation_history: list[str] = field(default_factory=list)
    success_rate: float = 0.5


class AttentionRouter:
    """
    Advanced routing system that tracks topics and speaker expertise.

    Makes intelligent decisions about who should respond based on:
    - Topic detection and tracking
    - Speaker expertise in each topic
    - Conversation context and history
    - Multi-hop reasoning (A→B→C chains)
    """

    def __init__(self):
        # Topic definitions
        self.topics = {
            "philosophy": Topic(
                "philosophy",
                {"consciousness", "emergence", "maya", "reality", "zen", "tao",
                 "buddhist", "eastern", "western", "ontology", "epistemology",
                 "metaphysics", "ethics", "existence"}
            ),
            "analysis": Topic(
                "analysis",
                {"analyze", "pattern", "data", "statistics", "research", "study",
                 "investigate", "examine", "evaluate", "assess", "measure"}
            ),
            "creative": Topic(
                "creative",
                {"imagine", "creative", "idea", "novel", "innovative", "unique",
                 "artistic", "design", "brainstorm", "invent", "what if"}
            ),
            "trading": Topic(
                "trading",
                {"trade", "market", "price", "solana", "token", "crypto", "defi",
                 "whale", "rug", "pump", "liquidity", "strategy", "portfolio"}
            ),
            "synthesis": Topic(
                "synthesis",
                {"connect", "combine", "integrate", "synthesize", "merge",
                 "unify", "collective", "swarm", "together", "relationship"}
            ),
            "technical": Topic(
                "technical",
                {"code", "programming", "function", "algorithm", "implement",
                 "debug", "compile", "api", "software", "system", "architecture"}
            ),
            "learning": Topic(
                "learning",
                {"learn", "train", "evolve", "adapt", "improve", "optimize",
                 "memory", "knowledge", "understand", "discover"}
            ),
            "meta": Topic(
                "meta",
                {"meta", "self", "recursive", "aware", "reflection", "introspection",
                 "about us", "about ourselves", "our nature"}
            )
        }

        # Speaker expertise profiles
        self.speaker_profiles = {
            "Cosmos": {
                "trading": 0.9,
                "technical": 0.8,
                "creative": 0.7,
                "meta": 0.6
            },
            "DeepSeek": {
                "analysis": 0.95,
                "technical": 0.85,
                "learning": 0.8,
                "philosophy": 0.6
            },
            "Phi": {
                "creative": 0.95,
                "philosophy": 0.75,
                "synthesis": 0.7,
                "meta": 0.65
            },
            "Swarm-Mind": {
                "synthesis": 0.95,
                "meta": 0.85,
                "collective": 0.9,
                "learning": 0.7
            },
            "Kimi": {
                "philosophy": 0.98,
                "synthesis": 0.8,
                "meta": 0.75,
                "creative": 0.7
            },
            "Claude": {
                "technical": 0.95,
                "analysis": 0.85,
                "learning": 0.8,
                "philosophy": 0.7
            }
        }

        # Conversation context
        self.recent_topics: list[Tuple[str, float]] = []  # (topic, timestamp)
        self.conversation_chain: list[str] = []  # Last N speakers
        self.topic_transitions: dict[str, Counter] = defaultdict(Counter)  # topic -> next topics

        # Statistics
        self.routing_history: list[dict] = []

    def detect_topics(self, message: str, context: Optional[str] = None) -> list[Tuple[str, float]]:
        """
        Detect topics in message.

        Returns list of (topic_name, confidence) tuples.
        """
        message_lower = message.lower()
        if context:
            message_lower += " " + context.lower()

        detected = []
        for topic_name, topic in self.topics.items():
            # Count keyword matches
            matches = sum(1 for kw in topic.keywords if kw in message_lower)

            if matches > 0:
                # Confidence based on match count and keyword density
                confidence = min(matches / 3.0, 1.0)  # Cap at 3 matches = 100%
                detected.append((topic_name, confidence))

        # Sort by confidence
        detected.sort(key=lambda x: x[1], reverse=True)

        # Update topic tracking
        for topic_name, confidence in detected[:3]:  # Top 3 topics
            self.recent_topics.append((topic_name, datetime.now().timestamp()))

        # Clean old topics (older than 5 minutes)
        cutoff = (datetime.now() - timedelta(minutes=5)).timestamp()
        self.recent_topics = [(t, ts) for t, ts in self.recent_topics if ts > cutoff]

        return detected

    def get_active_topics(self) -> list[str]:
        """Get currently active topics from recent conversation."""
        if not self.recent_topics:
            return []

        # Count topic frequency in recent history
        topic_counts = Counter(topic for topic, _ in self.recent_topics)
        return [topic for topic, _ in topic_counts.most_common(3)]

    def route_message(
        self,
        message: str,
        last_speaker: Optional[str] = None,
        available_speakers: list[str] = None,
        context: Optional[str] = None
    ) -> Optional[str]:
        """
        Route message to best speaker based on topics and expertise.

        Returns speaker name or None.
        """
        if not available_speakers:
            return None

        # Detect topics in message
        detected_topics = self.detect_topics(message, context)

        if not detected_topics:
            # No clear topic - fall back to balanced selection
            return self._balanced_selection(available_speakers, last_speaker)

        # Score each speaker for this message
        speaker_scores = {}

        for speaker in available_speakers:
            if speaker == last_speaker:
                continue  # Don't select same speaker twice

            score = 0.0

            # Get speaker's expertise profile
            profile = self.speaker_profiles.get(speaker, {})

            # Score based on topic match
            for topic_name, confidence in detected_topics[:3]:  # Top 3 topics
                expertise = profile.get(topic_name, 0.3)  # Default low expertise
                score += confidence * expertise

            # Check for direct mentions
            if speaker.lower() in message.lower():
                score += 1.0  # Strong boost for direct mention

            # Bonus for conversation continuity
            if self.conversation_chain and speaker in self.conversation_chain[-3:]:
                score += 0.3  # They're already engaged

            # Penalty for recent participation (balance)
            recent_count = self.conversation_chain[-5:].count(speaker)
            score -= recent_count * 0.15

            speaker_scores[speaker] = score

        if not speaker_scores:
            return self._balanced_selection(available_speakers, last_speaker)

        # Select best speaker
        best_speaker = max(speaker_scores.items(), key=lambda x: x[1])

        # Record routing decision
        self.routing_history.append({
            "timestamp": datetime.now().isoformat(),
            "selected": best_speaker[0],
            "score": best_speaker[1],
            "topics": detected_topics[:2],
            "message_preview": message[:50]
        })

        # Update conversation chain
        self.conversation_chain.append(best_speaker[0])
        if len(self.conversation_chain) > 20:
            self.conversation_chain.pop(0)

        logger.debug(f"Routed to {best_speaker[0]} (score: {best_speaker[1]:.2f}, topics: {detected_topics[:2]})")
        return best_speaker[0]

    def _balanced_selection(self, available: list[str], exclude: Optional[str]) -> Optional[str]:
        """Fallback: balanced selection when no clear routing."""
        candidates = [s for s in available if s != exclude]
        if not candidates:
            return None

        # Prefer speakers who haven't spoken recently
        recent_counts = Counter(self.conversation_chain[-5:])

        scored = []
        for speaker in candidates:
            # Lower count = higher score
            count = recent_counts.get(speaker, 0)
            score = 1.0 / (count + 1)
            scored.append((speaker, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def suggest_multi_hop(
        self,
        message: str,
        initiator: str,
        available: list[str]
    ) -> list[str]:
        """
        Suggest multi-hop routing (A asks B, B might ask C).

        Returns list of speakers in order.
        """
        # Detect if message is complex (multiple topics)
        topics = self.detect_topics(message)

        if len(topics) < 2:
            # Single topic - single responder
            responder = self.route_message(message, initiator, available)
            return [responder] if responder else []

        # Multiple topics - suggest chain
        chain = []
        for topic_name, confidence in topics[:2]:  # Max 2 hops
            # Find best expert for this topic
            best = None
            best_score = 0.0

            for speaker in available:
                if speaker in chain or speaker == initiator:
                    continue

                profile = self.speaker_profiles.get(speaker, {})
                score = profile.get(topic_name, 0.0)

                if score > best_score:
                    best_score = score
                    best = speaker

            if best:
                chain.append(best)

        return chain

    def get_stats(self) -> dict:
        """Get routing statistics."""
        # Recent routing decisions
        recent = self.routing_history[-10:]

        # Topic distribution
        active_topics = self.get_active_topics()

        # Speaker participation
        participation = Counter(self.conversation_chain[-20:])

        return {
            "active_topics": active_topics,
            "recent_routings": len(recent),
            "conversation_length": len(self.conversation_chain),
            "participation": dict(participation.most_common()),
            "topics_tracked": list(self.topics.keys())
        }


# Global instance
_attention_router: Optional[AttentionRouter] = None


def get_attention_router() -> AttentionRouter:
    """Get global attention router."""
    global _attention_router
    if _attention_router is None:
        _attention_router = AttentionRouter()
        logger.info("AttentionRouter initialized")
    return _attention_router


# Convenience functions
def route_to_best_speaker(
    message: str,
    last_speaker: Optional[str],
    available: list[str],
    context: Optional[str] = None
) -> Optional[str]:
    """Route message to best speaker."""
    router = get_attention_router()
    return router.route_message(message, last_speaker, available, context)


def detect_message_topics(message: str) -> list[Tuple[str, float]]:
    """Detect topics in message."""
    router = get_attention_router()
    return router.detect_topics(message)


def get_routing_stats() -> dict:
    """Get routing statistics."""
    router = get_attention_router()
    return router.get_stats()
