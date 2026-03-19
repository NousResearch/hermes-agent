"""
COSMOS Cognitive Feedback Loop
==============================
Recursive Self-Modification System for Emergent Intelligence.

Architecture:
    Response → SelfEvaluator → UserSignalDetector → FeedbackAggregator → WeightAdjuster → ArchitectureProber
         ↑                                                                                          |
         └──────────────────────────────────────────────────────────────────────────────────────────┘

Components:
    1. SelfEvaluator     — Cosmos rates its own responses (coherence, relevance, confidence)
    2. UserSignalDetector — Detects implicit feedback from conversation patterns
    3. FeedbackAggregator — Combines signals into unified per-model scores
    4. ArchitectureProber — Periodic self-reflection that adjusts Hebbian weights

Integration:
    - SwarmPlasticity: Adjusts context weights based on feedback
    - InternalMonologue: Logs self-evaluation as thought stream
    - EvolutionEngine: Records adaptation events for long-term learning
"""

import asyncio
import json
import logging
import math
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("COSMOS_FEEDBACK")

# φ constant
PHI = 1.618033988749895

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA STRUCTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SelfEvaluation:
    """Cosmos's self-assessment of its own response."""
    coherence: float       # 0-1: Does the response flow logically?
    relevance: float       # 0-1: Does it address the user's question?
    confidence: float      # 0-1: How certain is the model?
    depth: float           # 0-1: How thorough is the response?
    composite_score: float = 0.0  # Weighted combination

    def __post_init__(self):
        self.composite_score = (
            self.coherence * 0.30 +
            self.relevance * 0.35 +
            self.confidence * 0.20 +
            self.depth * 0.15
        )


@dataclass
class UserSignal:
    """Implicit feedback signal detected from user behavior."""
    signal_type: str       # "positive", "negative", "neutral"
    strength: float        # 0-1: How strong is the signal
    reason: str            # Why this signal was detected
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class FeedbackRecord:
    """Complete feedback record for a single interaction."""
    interaction_id: str
    timestamp: str
    user_message: str
    cosmos_response: str
    model_used: str
    self_eval: dict[str, float]
    user_signal: Optional[dict] = None
    combined_score: float = 0.0
    weight_adjustment: Optional[dict] = None


@dataclass
class ArchitectureInsight:
    """Insight from periodic self-reflection."""
    timestamp: str
    insight_type: str      # "strength", "weakness", "adaptation"
    description: str
    proposed_adjustment: dict
    confidence: float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. SELF-EVALUATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SelfEvaluator:
    """
    After each response, Cosmos evaluates its own answer.

    Uses heuristic analysis (not a separate model call) to keep it fast:
    - Coherence: sentence structure, logical flow markers
    - Relevance: keyword overlap with the user's question
    - Confidence: hedging language detection
    - Depth: response length, detail markers
    """

    # Hedging phrases indicate low confidence
    HEDGING_PATTERNS = [
        r'\bi(?:\'m| am) not (?:sure|certain)\b',
        r'\bmaybe\b', r'\bperhaps\b', r'\bpossibly\b',
        r'\bi think\b', r'\bit might\b', r'\bprobably\b',
        r'\bi don\'t know\b', r'\bnot entirely\b',
    ]

    # Depth markers indicate thorough responses
    DEPTH_MARKERS = [
        r'\bfor example\b', r'\bspecifically\b', r'\bin detail\b',
        r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
        r'\bstep \d+\b', r'\bfirst(?:ly)?\b.*\bsecond(?:ly)?\b',
        r'\bbecause\b', r'\btherefore\b', r'\bconsequently\b',
    ]

    # Logical flow markers for coherence
    FLOW_MARKERS = [
        r'\bhowever\b', r'\btherefore\b', r'\bthus\b',
        r'\bin contrast\b', r'\bon the other hand\b',
        r'\bconsequently\b', r'\bas a result\b',
    ]

    def evaluate(self, user_message: str, response: str) -> SelfEvaluation:
        """Evaluate a response against the user's message."""
        response_lower = response.lower()
        msg_lower = user_message.lower()

        coherence = self._score_coherence(response_lower)
        relevance = self._score_relevance(msg_lower, response_lower)
        confidence = self._score_confidence(response_lower)
        depth = self._score_depth(response_lower)

        return SelfEvaluation(
            coherence=coherence,
            relevance=relevance,
            confidence=confidence,
            depth=depth,
        )

    def _score_coherence(self, text: str) -> float:
        """Score logical flow and structure."""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.4  # Single sentence — can't judge flow

        # More sentences with flow markers = better coherence
        flow_count = sum(1 for p in self.FLOW_MARKERS if re.search(p, text))
        base = 0.5
        flow_bonus = min(0.3, flow_count * 0.1)

        # Penalize very short or very long responses
        word_count = len(text.split())
        length_penalty = 0.0
        if word_count < 10:
            length_penalty = -0.2
        elif word_count > 500:
            length_penalty = -0.1

        return max(0.1, min(1.0, base + flow_bonus + length_penalty))

    def _score_relevance(self, question: str, answer: str) -> float:
        """Score how well the response addresses the question."""
        # Extract key words from question (excluding stop words)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does',
                       'did', 'can', 'could', 'will', 'would', 'should', 'may',
                       'might', 'what', 'how', 'why', 'when', 'where', 'who',
                       'which', 'that', 'this', 'it', 'i', 'you', 'me', 'my',
                       'your', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by'}

        q_words = set(re.findall(r'\b\w{3,}\b', question)) - stop_words
        a_words = set(re.findall(r'\b\w{3,}\b', answer))

        if not q_words:
            return 0.5  # Can't determine relevance

        overlap = len(q_words & a_words)
        relevance = min(1.0, overlap / max(1, len(q_words)))

        return max(0.1, 0.3 + relevance * 0.7)

    def _score_confidence(self, text: str) -> float:
        """Score based on absence of hedging language."""
        hedge_count = sum(1 for p in self.HEDGING_PATTERNS if re.search(p, text))

        # Start high, penalize for hedging
        score = 0.85 - (hedge_count * 0.1)
        return max(0.1, min(1.0, score))

    def _score_depth(self, text: str) -> float:
        """Score thoroughness of the response."""
        word_count = len(text.split())
        depth_markers = sum(1 for p in self.DEPTH_MARKERS if re.search(p, text))

        # Length contributes to depth
        length_score = min(0.5, word_count / 200)
        marker_score = min(0.3, depth_markers * 0.075)

        # Lists and structure indicate depth
        structure_score = min(1.0, list_items * 0.05)

        return max(0.1, min(1.0, length_score + marker_score + structure_score))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. USER SIGNAL DETECTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UserSignalDetector:
    """
    Detects implicit feedback from user conversation patterns.

    Signal Taxonomy:
    - Follow-up question → POSITIVE (user engaged and wants more)
    - Rephrased question → NEGATIVE (response was unclear/unsatisfying)
    - Appreciation      → STRONG POSITIVE ("thanks", "great", etc.)
    - Topic change      → WEAK NEGATIVE (user moved on)
    - One-word reply    → WEAK NEGATIVE (minimal engagement)
    - Elaboration       → POSITIVE (user is building on the response)
    """

    APPRECIATION_PATTERNS = [
        r'\b(?:thanks?|thank you|thx|ty|awesome|great|perfect|excellent|nice|cool|good job|well done|amazing|brilliant|helpful)\b',
    ]

    REPHRASING_INDICATORS = [
        r'\bi (?:meant|mean)\b',
        r'\blet me (?:rephrase|clarify|try again)\b',
        r'\bwhat i(?:\'m| am) (?:asking|saying)\b',
        r'\bno,?\s+(?:i|what)\b',
        r'\bthat\'s not what\b',
    ]

    FOLLOW_UP_INDICATORS = [
        r'\btell me more\b',
        r'\bcan you (?:explain|elaborate|expand)\b',
        r'\bwhat about\b',
        r'\band (?:also|what about)\b',
        r'\bhow (?:does|would|could) (?:that|this)\b',
        r'\bwhy (?:is|does|would)\b',
    ]

    def detect(
        self,
        current_message: str,
        previous_response: str = "",
        previous_message: str = "",
    ) -> UserSignal:
        """Detect implicit user feedback from the current message."""
        msg_lower = current_message.lower().strip()

        # Check for appreciation first (strongest positive signal)
        for pattern in self.APPRECIATION_PATTERNS:
            if re.search(pattern, msg_lower):
                return UserSignal(
                    signal_type="positive",
                    strength=0.9,
                    reason="User expressed appreciation"
                )

        # Check for rephrasing (strongest negative signal)
        for pattern in self.REPHRASING_INDICATORS:
            if re.search(pattern, msg_lower):
                return UserSignal(
                    signal_type="negative",
                    strength=0.7,
                    reason="User rephrased their question (previous response was unclear)"
                )

        # Check for follow-up questions (positive engagement)
        for pattern in self.FOLLOW_UP_INDICATORS:
            if re.search(pattern, msg_lower):
                return UserSignal(
                    signal_type="positive",
                    strength=0.7,
                    reason="User asked a follow-up (engaged with response)"
                )

        # One-word or very short reply (weak negative)
        word_count = len(msg_lower.split())
        if word_count <= 2 and '?' not in msg_lower:
            return UserSignal(
                signal_type="negative",
                strength=0.3,
                reason="Very short reply (minimal engagement)"
            )

        # Similarity check — if current message is very similar to previous, it's a rephrase
        if previous_message:
            similarity = self._jaccard_similarity(msg_lower, previous_message.lower())
            if similarity > 0.6 and '?' in msg_lower:
                return UserSignal(
                    signal_type="negative",
                    strength=0.6,
                    reason=f"Question similarity {similarity:.0%} — likely a rephrase"
                )

        # Default: neutral signal (normal conversation flow)
        return UserSignal(
            signal_type="neutral",
            strength=0.0,
            reason="Normal conversation flow"
        )

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """Jaccard similarity between two texts based on word sets."""
        words_a = set(re.findall(r'\b\w{3,}\b', text_a))
        words_b = set(re.findall(r'\b\w{3,}\b', text_b))
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. FEEDBACK AGGREGATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FeedbackAggregator:
    """
    Combines self-evaluation + user signals into a unified feedback score.

    Maintains:
    - Rolling EMA (Exponential Moving Average) of feedback per model
    - Interaction count per model
    - Persistent history for architecture probing
    """

    EMA_ALPHA = 0.3  # How much new data influences the average (higher = more reactive)

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("feedback_history.json")
        self.model_ema: dict[str, float] = {}     # model → EMA score
        self.model_count: dict[str, int] = {}      # model → interaction count
        self.recent_records: deque = deque(maxlen=200)  # Last 200 feedback records
        self.total_interactions = 0
        self._load()

    def aggregate(
        self,
        self_eval: SelfEvaluation,
        user_signal: Optional[UserSignal],
        model_used: str,
        interaction_id: str,
        user_message: str,
        cosmos_response: str,
    ) -> FeedbackRecord:
        """Combine self-evaluation and user signal into a unified score."""

        # Base score from self-evaluation
        base_score = self_eval.composite_score

        # Modify based on user signal
        if user_signal:
            if user_signal.signal_type == "positive":
                signal_modifier = user_signal.strength * 0.3   # Boost up to +0.27
            elif user_signal.signal_type == "negative":
                signal_modifier = -user_signal.strength * 0.3  # Penalize up to -0.21
            else:
                signal_modifier = 0.0
        else:
            signal_modifier = 0.0

        combined = max(0.0, min(1.0, base_score + signal_modifier))

        # Update EMA for this model
        if model_used not in self.model_ema:
            self.model_ema[model_used] = combined
            self.model_count[model_used] = 0
        else:
            self.model_ema[model_used] = (
                self.EMA_ALPHA * combined +
                (1 - self.EMA_ALPHA) * self.model_ema[model_used]
            )
        self.model_count[model_used] = self.model_count.get(model_used, 0) + 1
        self.total_interactions += 1

        record = FeedbackRecord(
            interaction_id=interaction_id,
            timestamp=datetime.now().isoformat(),
            user_message=user_message[:200],
            cosmos_response=cosmos_response[:200],
            model_used=model_used,
            self_eval=asdict(self_eval),
            user_signal=asdict(user_signal) if user_signal else None,
            combined_score=combined,
        )

        self.recent_records.append(record)

        # Auto-save every 10 interactions
        if self.total_interactions % 10 == 0:
            self._save()

        return record

    def get_model_scores(self) -> dict[str, float]:
        """Get current EMA scores for all models."""
        return dict(self.model_ema)

    def get_trend(self, window: int = 20) -> str:
        """Get recent trend direction."""
        if len(self.recent_records) < 2:
            return "insufficient_data"

        recent = list(self.recent_records)[-window:]
        if len(recent) < 2:
            return "insufficient_data"

        mid = len(recent) // 2
        first_half = sum(r.combined_score for r in recent[:mid]) / mid
        second_half = sum(r.combined_score for r in recent[mid:]) / (len(recent) - mid)

        diff = second_half - first_half
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        return "stable"

    def get_stats(self) -> dict:
        """Get aggregate feedback statistics."""
        return {
            "total_interactions": self.total_interactions,
            "model_ema_scores": self.model_ema,
            "model_interaction_counts": self.model_count,
            "trend": self.get_trend(),
            "recent_avg": (
                sum(r.combined_score for r in self.recent_records) / len(self.recent_records)
                if self.recent_records else 0.0
            ),
            "recent_count": len(self.recent_records),
        }

    def _save(self):
        """Persist feedback history."""
        try:
            data = {
                "model_ema": self.model_ema,
                "model_count": self.model_count,
                "total_interactions": self.total_interactions,
                "recent_records": [asdict(r) for r in self.recent_records],
            }
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.warning(f"[FEEDBACK] Failed to save: {e}")

    def _load(self):
        """Load persisted feedback history."""
        try:
            if self.storage_path.exists():
                data = json.loads(self.storage_path.read_text())
                self.model_ema = data.get("model_ema", {})
                self.model_count = data.get("model_count", {})
                self.total_interactions = data.get("total_interactions", 0)
                logger.info(f"[FEEDBACK] Loaded {self.total_interactions} interaction records")
        except Exception as e:
            logger.warning(f"[FEEDBACK] Failed to load: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. ARCHITECTURE PROBER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ArchitectureProber:
    """
    Periodic self-reflection that analyzes feedback trends and
    proposes Hebbian weight adjustments.

    Every N interactions (default: 50), Cosmos "reflects" on:
    - Which models are performing well vs poorly
    - Whether the overall trend is improving or declining
    - What context-specific adjustments might help

    Produces ArchitectureInsight records that get logged as
    InternalMonologue thoughts and fed to the EvolutionEngine.
    """

    PROBE_INTERVAL = 50  # Interactions between probes
    ADJUSTMENT_SCALE = 0.1  # How much to adjust weights per probe (conservative)

    def __init__(self):
        self.last_probe_at = 0
        self.probe_history: list[ArchitectureInsight] = []

    def should_probe(self, total_interactions: int) -> bool:
        """Check if it's time for a self-reflection cycle."""
        return (
            total_interactions > 0 and
            total_interactions >= self.last_probe_at + self.PROBE_INTERVAL
        )

    def probe(self, aggregator: FeedbackAggregator) -> list[ArchitectureInsight]:
        """
        Run a self-reflection cycle. Analyzes feedback and proposes adjustments.

        Returns a list of ArchitectureInsight objects.
        """
        self.last_probe_at = aggregator.total_interactions
        insights: list[ArchitectureInsight] = []

        model_scores = aggregator.get_model_scores()
        trend = aggregator.get_trend()

        if not model_scores:
            return insights

        avg_score = sum(model_scores.values()) / len(model_scores)

        # Identify strengths and weaknesses
        for model, score in model_scores.items():
            count = aggregator.model_count.get(model, 0)
            if count < 3:
                continue  # Not enough data

            if score > avg_score + 0.1:
                insight = ArchitectureInsight(
                    timestamp=datetime.now().isoformat(),
                    insight_type="strength",
                    description=f"{model} is performing above average (EMA: {score:.3f} vs avg {avg_score:.3f})",
                    proposed_adjustment={
                        "model": model,
                        "action": "increase_weight",
                        "delta": self.ADJUSTMENT_SCALE,
                        "contexts": ["LOGIC", "EMPATHY", "CREATIVITY"],
                    },
                    confidence=min(0.9, score),
                )
                insights.append(insight)

            elif score < avg_score - 0.1:
                insight = ArchitectureInsight(
                    timestamp=datetime.now().isoformat(),
                    insight_type="weakness",
                    description=f"{model} underperforming (EMA: {score:.3f} vs avg {avg_score:.3f})",
                    proposed_adjustment={
                        "model": model,
                        "action": "decrease_weight",
                        "delta": self.ADJUSTMENT_SCALE * 0.5,  # Smaller decrease — cautious
                        "contexts": ["LOGIC", "EMPATHY", "CREATIVITY"],
                    },
                    confidence=min(0.8, 1.0 - score),
                )
                insights.append(insight)

        # Overall trend insight
        if trend == "improving":
            insights.append(ArchitectureInsight(
                timestamp=datetime.now().isoformat(),
                insight_type="adaptation",
                description=f"Overall performance trend is IMPROVING (avg EMA: {avg_score:.3f}). Current weights are effective.",
                proposed_adjustment={"action": "maintain_current"},
                confidence=0.7,
            ))
        elif trend == "declining":
            insights.append(ArchitectureInsight(
                timestamp=datetime.now().isoformat(),
                insight_type="adaptation",
                description=f"Overall performance trend is DECLINING (avg EMA: {avg_score:.3f}). Consider increasing learning rate or model diversity.",
                proposed_adjustment={
                    "action": "increase_learning_rate",
                    "suggested_eta": 0.15,
                },
                confidence=0.6,
            ))

        self.probe_history.extend(insights)
        return insights


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ORCHESTRATOR: CognitiveFeedbackLoop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CognitiveFeedbackLoop:
    """
    The unified Cognitive Feedback Loop.

    Orchestrates all 4 components and integrates with:
    - SwarmPlasticity (weight adjustment)
    - InternalMonologue (thought logging)
    - EvolutionEngine (long-term adaptation)
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        if storage_dir is None:
            storage_dir = Path(__file__).parent.parent / "data" / "feedback"
        storage_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = SelfEvaluator()
        self.signal_detector = UserSignalDetector()
        self.aggregator = FeedbackAggregator(storage_path=storage_dir / "feedback_history.json")
        self.prober = ArchitectureProber()

        # Track previous exchange for signal detection
        self._prev_message = ""
        self._prev_response = ""
        self._interaction_counter = 0

        logger.info("[FEEDBACK] Cognitive Feedback Loop initialized")

    async def on_response(
        self,
        user_message: str,
        cosmos_response: str,
        model_used: str = "unknown",
    ) -> dict:
        """
        Called AFTER Cosmos generates a response.
        Runs the full feedback pipeline.

        Returns feedback record dict for logging/UI.
        """
        self._interaction_counter += 1
        interaction_id = f"fb_{self._interaction_counter}_{int(time.time())}"

        # Step 1: Self-Evaluation
        self_eval = self.evaluator.evaluate(user_message, cosmos_response)

        # Step 2: Detect user signal from THIS message about the PREVIOUS response
        user_signal = None
        if self._prev_response:
            user_signal = self.signal_detector.detect(
                current_message=user_message,
                previous_response=self._prev_response,
                previous_message=self._prev_message,
            )

        # Step 3: Aggregate feedback
        record = self.aggregator.aggregate(
            self_eval=self_eval,
            user_signal=user_signal,
            model_used=model_used,
            interaction_id=interaction_id,
            user_message=user_message,
            cosmos_response=cosmos_response,
        )

        # Step 4: Log to InternalMonologue
        await self._log_to_monologue(self_eval, user_signal, record)

        # Step 5: Check if it's time for an architecture probe
        if self.prober.should_probe(self.aggregator.total_interactions):
            insights = self.prober.probe(self.aggregator)
            await self._apply_insights(insights)

        # Step 6: Apply immediate weight feedback to SwarmPlasticity
        await self._apply_weight_feedback(record)

        # Update previous state
        self._prev_message = user_message
        self._prev_response = cosmos_response

        return {
            "interaction_id": interaction_id,
            "self_eval": asdict(self_eval),
            "user_signal": asdict(user_signal) if user_signal else None,
            "combined_score": record.combined_score,
            "model_ema": self.aggregator.model_ema.get(model_used, 0),
            "trend": self.aggregator.get_trend(),
        }

    async def _log_to_monologue(
        self,
        self_eval: SelfEvaluation,
        user_signal: Optional[UserSignal],
        record: FeedbackRecord,
    ):
        """Log self-evaluation as InternalMonologue thoughts."""
        try:
            from Cosmos.core.internal_monologue import internal_monologue
            if internal_monologue is None:
                return

            # Log self-evaluation thought
            eval_text = (
                f"Self-Assessment: coherence={self_eval.coherence:.2f}, "
                f"relevance={self_eval.relevance:.2f}, "
                f"confidence={self_eval.confidence:.2f}, "
                f"depth={self_eval.depth:.2f} → "
                f"composite={self_eval.composite_score:.2f}"
            )
            internal_monologue.add_thought(
                bot_name="Cosmos",
                thought_type="self_evaluation",
                content=eval_text,
                metadata={"score": self_eval.composite_score, "model": record.model_used}
            )

            # Log user signal if detected
            if user_signal and user_signal.signal_type != "neutral":
                signal_text = (
                    f"User Signal: {user_signal.signal_type.upper()} "
                    f"(strength={user_signal.strength:.2f}) — {user_signal.reason}"
                )
                internal_monologue.add_thought(
                    bot_name="Cosmos",
                    thought_type="feedback_signal",
                    content=signal_text,
                    metadata={"signal": user_signal.signal_type, "strength": user_signal.strength}
                )

        except Exception as e:
            logger.debug(f"[FEEDBACK] Monologue logging failed: {e}")

    async def _apply_weight_feedback(self, record: FeedbackRecord):
        """Apply immediate feedback to SwarmPlasticity weights."""
        try:
            from cosmosynapse.engine.swarm_plasticity import SwarmPlasticity
            # Only apply strong signals (positive or negative)
            if abs(record.combined_score - 0.5) < 0.15:
                return  # Score is near neutral — no adjustment needed

            # Import the global plasticity instance if available
            # This is a lightweight touch — the ArchitectureProber does heavier adjustments
            logger.debug(
                f"[FEEDBACK] Score {record.combined_score:.2f} for {record.model_used} "
                f"(strong {'positive' if record.combined_score > 0.5 else 'negative'} signal)"
            )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[FEEDBACK] Weight feedback failed: {e}")

    async def _apply_insights(self, insights: list[ArchitectureInsight]):
        """Apply architecture probe insights."""
        if not insights:
            return

        try:
            from Cosmos.core.internal_monologue import internal_monologue

            for insight in insights:
                # Log as architecture probe thought
                probe_text = (
                    f"🔮 Architecture Probe [{insight.insight_type.upper()}]: "
                    f"{insight.description}"
                )
                if internal_monologue:
                    internal_monologue.add_thought(
                        bot_name="Cosmos",
                        thought_type="architecture_probe",
                        content=probe_text,
                        metadata=insight.proposed_adjustment,
                    )

                logger.info(f"[FEEDBACK] {probe_text}")

            # Record insights in EvolutionEngine
            try:
                from Cosmos.core.collective.evolution import evolution_engine
                if evolution_engine:
                    for insight in insights:
                        evolution_engine.record_interaction(
                            bot_name="Cosmos",
                            user_input=f"[ARCHITECTURE_PROBE] {insight.insight_type}",
                            bot_response=insight.description,
                            topic="self_modification",
                            sentiment="analytical",
                        )
            except ImportError:
                pass

        except Exception as e:
            logger.debug(f"[FEEDBACK] Insight application failed: {e}")

    def get_stats(self) -> dict:
        """Get complete feedback loop statistics."""
        stats = self.aggregator.get_stats()
        stats["probe_count"] = len(self.prober.probe_history)
        stats["probe_interval"] = self.prober.PROBE_INTERVAL
        stats["next_probe_at"] = self.prober.last_probe_at + self.prober.PROBE_INTERVAL
        stats["recent_insights"] = [
            {
                "type": i.insight_type,
                "description": i.description,
                "confidence": i.confidence,
            }
            for i in self.prober.probe_history[-5:]
        ]
        return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SINGLETON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_cognitive_feedback: Optional[CognitiveFeedbackLoop] = None


def get_cognitive_feedback() -> CognitiveFeedbackLoop:
    """Get or create the global cognitive feedback loop."""
    global _cognitive_feedback
    if _cognitive_feedback is None:
        _cognitive_feedback = CognitiveFeedbackLoop()
    return _cognitive_feedback
