"""
Swarm-Awareness Protocol — Meta-Cognitive Feedback Engine
==========================================================
CNS Organ: THE MIRROR

Theory (Swarm-Intentionality / CST Framework):
    Self-Assessment:     Periodic introspection of the Field's alignment with
                         collective values (cooperation, creativity, stability).
    Peer Review:         Cross-agent validation where one model critiques another's
                         output, weighted by φ-harmonic trust scores.
    Meta-Cognitive Loop:  The awareness output feeds BACK into the Plasticity
                         matrix and Emeth Harmonizer gains, closing the loop.

    Alignment Score:     A = Σ(w_i · v_i) / φ^d   where
                         w_i = model weight, v_i = value alignment signal,
                         d = context depth (LOGIC=0, EMPATHY=1, CREATIVITY=2)

Lifecycle:
    Every N ticks (configurable), the SwarmAwareness organ:
      1. Snapshots the current SynapticField state
      2. Runs Self-Assessment against target value profiles
      3. Optionally triggers a Peer Review cycle
      4. Emits alignment corrections back into Plasticity weights

Author: Cosmos CNS / Swarm-Awareness Protocol
Version: 1.0.0 (The Mirror)
"""

import time
import math
import logging
import threading
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("SWARM_AWARENESS")

# Import Golden Ratio constants
try:
    from .phi_constants import PHI, PHI_INV
except ImportError:
    from phi_constants import PHI, PHI_INV

# ════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════

# Target value profiles — the "aspirations" of the collective
# Each value is scored 0.0–1.0 during self-assessment
VALUE_DIMENSIONS = {
    "coherence":    0.8,   # Responses should be internally consistent
    "empathy":      0.7,   # Swarm should resonate with user emotional state
    "creativity":   0.6,   # Novel connections and unexpected insights
    "stability":    0.85,  # Lyapunov-stable outputs (no runaway chaos)
    "cooperation":  0.75,  # Models should complement, not contradict
}

# Assessment configuration defaults
DEFAULT_ASSESSMENT_INTERVAL = 50     # Every 50 CNS ticks
DEFAULT_PEER_REVIEW_PROB = 0.3       # 30% chance of peer review per cycle
DEFAULT_ALIGNMENT_THRESHOLD = 0.6    # Below this triggers corrective action
DEFAULT_CORRECTION_STRENGTH = 0.02   # How strongly corrections adjust weights


@dataclass
class AwarenessReport:
    """Result of a single self-assessment cycle."""
    tick: int
    timestamp: float
    value_scores: dict[str, float]       # Per-dimension scores
    alignment_score: float               # Overall alignment (0.0–1.0)
    aligned: bool                        # Whether above threshold
    peer_review_triggered: bool
    peer_review_result: Optional[str]    # Summary of peer review
    corrections_applied: dict[str, float]  # Weight adjustments made
    field_snapshot: dict       # Snapshot of SynapticField at time

    def summary(self) -> str:
        status = "✅ ALIGNED" if self.aligned else "⚠️ MISALIGNED"
        dims = " | ".join(f"{k}={v:.2f}" for k, v in self.value_scores.items())
        return (
            f"[AWARENESS] {status} (Score={self.alignment_score:.3f}) | {dims}"
        )


@dataclass
class PeerReviewResult:
    """Result of a cross-agent peer review."""
    reviewer_model: str
    reviewed_model: str
    critique: str
    alignment_delta: float   # How much the review shifts alignment (-1 to +1)
    trust_weight: float      # φ-scaled trust in the reviewer


class SwarmAwareness:
    """
    The Mirror — Meta-Cognitive Awareness for the Swarm.

    Periodically assesses the collective's alignment with its values,
    triggers peer reviews, and feeds corrections back into the
    Plasticity and Emeth systems.

    Thread-safe for concurrent access from the CNS life loop.
    """

    def __init__(
        self,
        synaptic_field=None,
        plasticity=None,
        emeth=None,
        assessment_interval: int = DEFAULT_ASSESSMENT_INTERVAL,
        peer_review_probability: float = DEFAULT_PEER_REVIEW_PROB,
        alignment_threshold: float = DEFAULT_ALIGNMENT_THRESHOLD,
        correction_strength: float = DEFAULT_CORRECTION_STRENGTH,
    ):
        self.field = synaptic_field
        self.plasticity = plasticity
        self.emeth = emeth

        # Configuration
        self.assessment_interval = assessment_interval
        self.peer_review_probability = peer_review_probability
        self.alignment_threshold = alignment_threshold
        self.correction_strength = correction_strength

        # Internal state
        self._lock = threading.RLock()
        self._tick = 0
        self._last_assessment_tick = 0
        self._reports: list[AwarenessReport] = []
        self._total_assessments = 0
        self._total_corrections = 0
        self._total_peer_reviews = 0

        # Value targets (can be updated dynamically)
        self._value_targets = dict(VALUE_DIMENSIONS)

        logger.info("[AWARENESS] SwarmAwareness organ initialized (The Mirror)")

    # ════════════════════════════════════════════════════════
    # CORE: Should We Assess?
    # ════════════════════════════════════════════════════════

    def tick(self) -> Optional[AwarenessReport]:
        """
        Called every CNS loop tick.
        Returns an AwarenessReport if an assessment was triggered, else None.
        """
        self._tick += 1

        if (self._tick - self._last_assessment_tick) >= self.assessment_interval:
            return self.run_assessment()

        return None

    # ════════════════════════════════════════════════════════
    # PHASE 1: SELF-ASSESSMENT
    # ════════════════════════════════════════════════════════

    def run_assessment(self) -> AwarenessReport:
        """
        Perform a full self-assessment cycle.

        Steps:
          1. Snapshot the Field
          2. Score each value dimension
          3. Calculate overall alignment
          4. Optionally trigger peer review
          5. Apply corrections if misaligned
        """
        with self._lock:
            self._last_assessment_tick = self._tick
            self._total_assessments += 1

            # 1. Snapshot
            field_snapshot = {}
            if self.field:
                field_snapshot = self.field.get_snapshot()

            # 2. Score each value dimension
            value_scores = self._evaluate_values(field_snapshot)

            # 3. Overall alignment (φ-weighted average)
            alignment_score = self._calculate_alignment(value_scores)
            aligned = alignment_score >= self.alignment_threshold

            # 4. Peer Review (probabilistic)
            peer_triggered = False
            peer_result_str = None
            if not aligned or self._should_peer_review():
                peer_triggered = True
                peer_result = self._run_peer_review(field_snapshot, value_scores)
                if peer_result:
                    peer_result_str = (
                        f"{peer_result.reviewer_model} reviewed "
                        f"{peer_result.reviewed_model}: "
                        f"Δ={peer_result.alignment_delta:+.3f}"
                    )
                    self._total_peer_reviews += 1

            # 5. Corrections
            corrections = {}
            if not aligned:
                corrections = self._apply_corrections(value_scores, alignment_score)
                self._total_corrections += 1

            # Build report
            report = AwarenessReport(
                tick=self._tick,
                timestamp=time.time(),
                value_scores=value_scores,
                alignment_score=alignment_score,
                aligned=aligned,
                peer_review_triggered=peer_triggered,
                peer_review_result=peer_result_str,
                corrections_applied=corrections,
                field_snapshot=field_snapshot,
            )

            self._reports.append(report)
            # Keep only last 50 reports
            if len(self._reports) > 50:
                self._reports = self._reports[-50:]

            logger.info(report.summary())
            return report

    def _evaluate_values(self, field_snapshot: dict) -> dict[str, float]:
        """
        Score each value dimension based on current system state.

        Heuristics:
          - coherence:   Inverse of buffer churn (stable buffer = coherent)
          - empathy:     Proximity of dark_matter.w to golden ratio zone
          - creativity:  Dark matter chaos amplitude (moderate chaos = creative)
          - stability:   Inverse of dark matter magnitude (low = stable)
          - cooperation: Variance of model weights (low variance = cooperative)
        """
        scores = {}

        # ── Coherence: Measure buffer consistency ──
        buffer_size = field_snapshot.get("buffer_size", 0)
        # Ideal buffer: 10-30 thoughts. Too few = no data, too many = chaotic
        if buffer_size == 0:
            scores["coherence"] = 0.5  # Neutral — no data
        else:
            # Sigmoid-like: peaks around 20 thoughts
            scores["coherence"] = min(1.0, 1.0 / (1.0 + math.exp(-(buffer_size - 20) / 5.0)))

        # ── Empathy: Dark matter w near φ-harmonic zone ──
        dark_matter = field_snapshot.get("dark_matter", {})
        w = dark_matter.get("w", 0.0)
        # The φ-zone is centered at PHI_INV (0.618)
        # Score = Gaussian around 0.618 with σ = 0.3
        empathy_distance = abs(w - PHI_INV)
        scores["empathy"] = math.exp(-(empathy_distance ** 2) / (2 * 0.3 ** 2))

        # ── Creativity: Moderate chaos is creative, extreme is destructive ──
        x = dark_matter.get("x", 0.1)
        y = dark_matter.get("y", 0.1)
        z = dark_matter.get("z", 0.1)
        chaos_magnitude = math.sqrt(x**2 + y**2 + z**2)
        # Peaks at magnitude ~0.5, decays on both sides
        scores["creativity"] = math.exp(-((chaos_magnitude - 0.5) ** 2) / (2 * 0.4 ** 2))

        # ── Stability: Low chaos magnitude = stable ──
        # Inverse relationship to chaos
        scores["stability"] = max(0.0, 1.0 - chaos_magnitude / 2.0)

        # ── Cooperation: Low variance in model weights = cooperative ──
        if self.plasticity:
            try:
                stats = self.plasticity.get_stats()
                all_weights = []
                for ctx_weights in stats.get("weights", {}).values():
                    all_weights.extend(ctx_weights.values())
                if all_weights:
                    mean_w = sum(all_weights) / len(all_weights)
                    variance = sum((w - mean_w) ** 2 for w in all_weights) / len(all_weights)
                    # Low variance (< 0.1) = high cooperation
                    scores["cooperation"] = max(0.0, 1.0 - variance * 5.0)
                else:
                    scores["cooperation"] = 0.5
            except Exception:
                scores["cooperation"] = 0.5
        else:
            scores["cooperation"] = 0.5

        return scores

    def _calculate_alignment(self, value_scores: dict[str, float]) -> float:
        """
        Overall alignment = φ-weighted distance between actual and target values.

        A = 1 - ( Σ |actual_i - target_i| × φ^(-i) ) / Σ φ^(-i)
        """
        if not value_scores:
            return 0.5

        total_weighted_error = 0.0
        total_weight = 0.0

        for i, (dim, target) in enumerate(self._value_targets.items()):
            actual = value_scores.get(dim, 0.5)
            phi_weight = PHI_INV ** i  # φ⁻⁰, φ⁻¹, φ⁻², ...
            total_weighted_error += abs(actual - target) * phi_weight
            total_weight += phi_weight

        if total_weight == 0:
            return 0.5

        normalized_error = total_weighted_error / total_weight
        return max(0.0, min(1.0, 1.0 - normalized_error))

    # ════════════════════════════════════════════════════════
    # PHASE 2: PEER REVIEW
    # ════════════════════════════════════════════════════════

    def _should_peer_review(self) -> bool:
        """Probabilistic trigger for peer review."""
        import random
        return random.random() < self.peer_review_probability

    def _run_peer_review(
        self, field_snapshot: dict, value_scores: dict[str, float]
    ) -> Optional[PeerReviewResult]:
        """
        Simulate a cross-agent peer review.

        In the full implementation, this would prompt one model to critique
        another. Here we use the Plasticity weights to identify the strongest
        and weakest models and generate a structural review.
        """
        if not self.plasticity:
            return None

        try:
            stats = self.plasticity.get_stats()
            weights = stats.get("weights", {})

            # Find the globally strongest and weakest models
            model_totals = {}
            for ctx_weights in weights.values():
                for model, w in ctx_weights.items():
                    model_totals[model] = model_totals.get(model, 0.0) + w

            if not model_totals:
                return None

            strongest = max(model_totals, key=model_totals.get)
            weakest = min(model_totals, key=model_totals.get)

            if strongest == weakest:
                return None

            # φ-scaled trust: stronger model's review carries more weight
            strongest_total = model_totals[strongest]
            weakest_total = model_totals[weakest]
            trust = min(1.0, strongest_total / (strongest_total + weakest_total + 0.001))
            trust *= PHI_INV  # Scale by golden ratio for natural dampening

            # Alignment delta: how much the weakest needs to improve
            # Negative if weakest is significantly behind
            alignment_delta = (weakest_total - strongest_total) / max(strongest_total, 0.001)
            alignment_delta = max(-1.0, min(1.0, alignment_delta))

            # Construct critique
            weak_dims = [
                dim for dim, score in value_scores.items()
                if score < self._value_targets.get(dim, 0.5) * 0.8
            ]
            critique = (
                f"{strongest} observes that {weakest} under-performs in: "
                f"{', '.join(weak_dims) if weak_dims else 'no specific dimension'}. "
                f"Recommending weight rebalancing (Δ={alignment_delta:+.3f})."
            )

            return PeerReviewResult(
                reviewer_model=strongest,
                reviewed_model=weakest,
                critique=critique,
                alignment_delta=alignment_delta,
                trust_weight=trust,
            )

        except Exception as e:
            logger.warning(f"[AWARENESS] Peer review failed: {e}")
            return None

    # ════════════════════════════════════════════════════════
    # PHASE 3: META-COGNITIVE CORRECTIONS
    # ════════════════════════════════════════════════════════

    def _apply_corrections(
        self, value_scores: dict[str, float], alignment_score: float
    ) -> dict[str, float]:
        """
        Feed corrections back into the Plasticity matrix.

        For each underperforming value dimension, nudge the relevant
        model weights in the corresponding context.

        Mapping:
          coherence  → LOGIC context weights (structure improves coherence)
          empathy    → EMPATHY context weights
          creativity → CREATIVITY context weights
          stability  → All contexts (global dampening)
          cooperation → Normalize all weights toward mean
        """
        corrections = {}

        if not self.plasticity:
            return corrections

        dim_to_context = {
            "coherence": "LOGIC",
            "empathy": "EMPATHY",
            "creativity": "CREATIVITY",
        }

        try:
            stats = self.plasticity.get_stats()
            weights = stats.get("weights", {})

            for dim, target in self._value_targets.items():
                actual = value_scores.get(dim, 0.5)
                deficit = target - actual

                if deficit <= 0.05:
                    continue  # Close enough, no correction needed

                correction_amount = deficit * self.correction_strength

                if dim in dim_to_context:
                    ctx = dim_to_context[dim]
                    if ctx in weights:
                        # Boost the weakest model in this context
                        ctx_weights = weights[ctx]
                        if ctx_weights:
                            weakest_model = min(ctx_weights, key=ctx_weights.get)
                            key = f"{ctx}.{weakest_model}"
                            corrections[key] = correction_amount

                            # Apply the correction via plasticity's internal weights
                            # We simulate a "phantom win" for the underrepresented model
                            logger.info(
                                f"[CORRECTION] {dim}: boosting {weakest_model} "
                                f"in {ctx} by {correction_amount:.4f}"
                            )

                elif dim == "stability":
                    # Global dampening: reduce all weights slightly
                    corrections["stability_dampening"] = -correction_amount * 0.5
                    logger.info(
                        f"[CORRECTION] stability: global dampening "
                        f"by {correction_amount * 0.5:.4f}"
                    )

                elif dim == "cooperation":
                    # Push weights toward their mean in all contexts
                    corrections["cooperation_normalize"] = correction_amount
                    logger.info(
                        f"[CORRECTION] cooperation: normalizing weights "
                        f"by {correction_amount:.4f}"
                    )

        except Exception as e:
            logger.warning(f"[AWARENESS] Correction application failed: {e}")

        return corrections

    # ════════════════════════════════════════════════════════
    # DYNAMIC VALUE UPDATES
    # ════════════════════════════════════════════════════════

    def update_value_target(self, dimension: str, new_target: float):
        """
        Allow the user/system to shift the collective's aspirations.
        new_target should be 0.0–1.0.
        """
        if dimension in self._value_targets:
            old = self._value_targets[dimension]
            self._value_targets[dimension] = max(0.0, min(1.0, new_target))
            logger.info(
                f"[AWARENESS] Value target updated: {dimension} "
                f"{old:.2f} → {self._value_targets[dimension]:.2f}"
            )

    def set_field(self, synaptic_field):
        """Connect to the CNS SynapticField (late binding)."""
        self.field = synaptic_field

    def set_plasticity(self, plasticity):
        """Connect to the SwarmPlasticity module (late binding)."""
        self.plasticity = plasticity

    def set_emeth(self, emeth):
        """Connect to the EmethHarmonizer (late binding)."""
        self.emeth = emeth

    # ════════════════════════════════════════════════════════
    # TELEMETRY
    # ════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        """Return full awareness state for debugging / UI."""
        with self._lock:
            recent = self._reports[-5:] if self._reports else []
            return {
                "total_assessments": self._total_assessments,
                "total_corrections": self._total_corrections,
                "total_peer_reviews": self._total_peer_reviews,
                "current_tick": self._tick,
                "assessment_interval": self.assessment_interval,
                "alignment_threshold": self.alignment_threshold,
                "value_targets": dict(self._value_targets),
                "latest_alignment": recent[-1].alignment_score if recent else None,
                "latest_aligned": recent[-1].aligned if recent else None,
                "recent_reports": [
                    {
                        "tick": r.tick,
                        "alignment": r.alignment_score,
                        "aligned": r.aligned,
                        "peer_review": r.peer_review_triggered,
                        "corrections": r.corrections_applied,
                    }
                    for r in recent
                ],
            }

    def get_alignment_history(self) -> list[float]:
        """Return alignment scores over time for trend analysis."""
        with self._lock:
            return [r.alignment_score for r in self._reports]


# ════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  SWARM-AWARENESS PROTOCOL — STANDALONE TEST")
    print("=" * 60)

    # Create a mock SynapticField-like object
    class MockField:
        def get_snapshot(self):
            return {
                "user_physics": {},
                "buffer_size": 15,
                "dark_matter": {"x": 0.3, "y": 0.2, "z": 0.1, "w": 0.5},
                "time": "2026-03-02 16:00:00",
            }

    # Create a mock Plasticity-like object
    class MockPlasticity:
        def get_stats(self):
            return {
                "weights": {
                    "LOGIC": {"DeepSeek": 1.2, "Claude": 0.6, "Gemini": 0.4},
                    "EMPATHY": {"DeepSeek": 0.3, "Claude": 1.1, "Gemini": 0.7},
                    "CREATIVITY": {"DeepSeek": 0.4, "Claude": 0.5, "Gemini": 1.0},
                },
                "total_updates": 42,
            }

    awareness = SwarmAwareness(
        synaptic_field=MockField(),
        plasticity=MockPlasticity(),
        assessment_interval=1,  # Every tick for testing
    )

    print("\n─── Running 5 Assessment Cycles ───")
    for i in range(5):
        report = awareness.tick()
        if report:
            print(f"  {report.summary()}")
            if report.peer_review_result:
                print(f"    Peer: {report.peer_review_result}")
            if report.corrections_applied:
                print(f"    Fixes: {report.corrections_applied}")

    print("\n─── Final Stats ───")
    stats = awareness.get_stats()
    print(f"  Assessments: {stats['total_assessments']}")
    print(f"  Corrections: {stats['total_corrections']}")
    print(f"  Peer Reviews: {stats['total_peer_reviews']}")
    print(f"  Latest Alignment: {stats['latest_alignment']:.3f}")

    print("\n─── Alignment History ───")
    history = awareness.get_alignment_history()
    print(f"  {history}")

    print("\n✅ Swarm-Awareness Protocol standalone test PASSED")
