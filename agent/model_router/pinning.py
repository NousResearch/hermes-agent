"""Session pinning, flip-flop guard, and cache-breakeven economics.

Ports of pi-smart-router pinning/session-pinner.ts (core hold/break logic),
pinning/flip-flop-guard.ts, and pinning/cache-breakeven.ts.

A session pin preserves provider prefix-cache economics: once a session is
routed to a model, subsequent turns stay on it unless an explicit break
condition fires (compaction, context overflow, loop escalation, or a
voluntary switch that clears the cache-breakeven / score-margin gate).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import ModelProfile, SessionPin

# ─── Flip-flop guard (port of flip-flop-guard.ts) ─────────────────────────────

FLIP_FLOP_PIN_THRESHOLD = 3


@dataclass(frozen=True)
class FlipFlopObservation:
    tier_flip_detected: bool
    consecutive_tier_flips: int
    tier_pinned: Optional[str]


class FlipFlopGuard:
    """Tracks consecutive per-turn tier flips within a session.

    When the shadow routing tier flips ``threshold`` times in a row, the
    session tier is pinned for the remainder of the process lifetime to stop
    adversarial paraphrase / suffix oscillation. In-memory (matches the Pi
    implementation); the durable pin lives in the state store.
    """

    def __init__(self, threshold: int = FLIP_FLOP_PIN_THRESHOLD):
        self._threshold = max(1, int(threshold))
        self._sessions = {}

    def is_tier_pinned(self, session_id: str) -> Optional[str]:
        state = self._sessions.get(session_id)
        return state[2] if state else None

    def observe_tier(self, session_id: str, observed_tier: str) -> FlipFlopObservation:
        if not session_id:
            return FlipFlopObservation(False, 0, None)
        current = self._sessions.get(session_id)
        if current is None:
            self._sessions[session_id] = (observed_tier, 0, None)
            return FlipFlopObservation(False, 0, None)

        last_tier, consecutive, tier_pinned = current
        if tier_pinned is not None:
            return FlipFlopObservation(False, consecutive, tier_pinned)
        if last_tier == observed_tier:
            self._sessions[session_id] = (observed_tier, 0, None)
            return FlipFlopObservation(False, 0, None)

        consecutive += 1
        if consecutive >= self._threshold:
            self._sessions[session_id] = (observed_tier, consecutive, observed_tier)
            return FlipFlopObservation(True, consecutive, observed_tier)
        self._sessions[session_id] = (observed_tier, consecutive, None)
        return FlipFlopObservation(True, consecutive, None)

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# ─── Cache breakeven economics (port of cache-breakeven.ts) ──────────────────

DEFAULT_PREFIX_CACHE_WEIGHT = 0.2
DEFAULT_PREFIX_CACHE_DISCOUNT = 0.9
_TOKENS_PER_M = 1_000_000


@dataclass(frozen=True)
class CacheBreakevenResult:
    should_switch: bool
    marginal_savings: float
    future_cache_value: float
    cache_reprime_cost: float
    total_benefit: float
    reason: str


def compute_future_cache_value(
    warm_prefix_tokens: float,
    pinned_cost_per_1m: float,
    prefix_cache_weight: float = DEFAULT_PREFIX_CACHE_WEIGHT,
    prefix_cache_discount: float = DEFAULT_PREFIX_CACHE_DISCOUNT,
) -> float:
    """Estimate the retained value of a warm prefix cache."""
    if warm_prefix_tokens <= 0 or pinned_cost_per_1m < 0:
        return 0.0
    prefix_cost = (warm_prefix_tokens / _TOKENS_PER_M) * pinned_cost_per_1m
    return prefix_cost * prefix_cache_discount * prefix_cache_weight


def compute_cache_reprime_cost(warm_prefix_tokens: float, candidate_cost_per_1m: float) -> float:
    """Cost to re-transmit a cold prefix on the candidate after a switch."""
    if warm_prefix_tokens <= 0 or candidate_cost_per_1m < 0:
        return 0.0
    return (warm_prefix_tokens / _TOKENS_PER_M) * candidate_cost_per_1m


def compute_marginal_switch_savings(
    pinned: ModelProfile,
    candidate: ModelProfile,
    estimated_input_tokens: int,
) -> Optional[float]:
    """Per-turn savings from switching, or None when pricing is undeclared."""
    if pinned.cost_per_1m is None or candidate.cost_per_1m is None:
        return None
    return max(0.0, (pinned.cost_per_1m - candidate.cost_per_1m)) * (estimated_input_tokens / _TOKENS_PER_M)


def evaluate_cache_breakeven_for_prefix(
    marginal_savings: float,
    warm_prefix_tokens: float,
    pinned_cost_per_1m: float,
    candidate_cost_per_1m: float,
    prefix_cache_weight: float = DEFAULT_PREFIX_CACHE_WEIGHT,
    prefix_cache_discount: float = DEFAULT_PREFIX_CACHE_DISCOUNT,
) -> CacheBreakevenResult:
    """Switch only when marginal_savings + future_cache_value > reprime cost."""
    components = (marginal_savings, warm_prefix_tokens, pinned_cost_per_1m, candidate_cost_per_1m)
    if any(not isinstance(v, (int, float)) or v < 0 for v in components):
        return CacheBreakevenResult(False, marginal_savings, 0.0, 0.0, marginal_savings, "invalid_input")

    future_value = compute_future_cache_value(
        warm_prefix_tokens, pinned_cost_per_1m, prefix_cache_weight, prefix_cache_discount
    )
    reprime_cost = compute_cache_reprime_cost(warm_prefix_tokens, candidate_cost_per_1m)
    total_benefit = marginal_savings + future_value

    if total_benefit <= reprime_cost:
        return CacheBreakevenResult(False, marginal_savings, future_value, reprime_cost, total_benefit, "breakeven_not_met")
    return CacheBreakevenResult(True, marginal_savings, future_value, reprime_cost, total_benefit, "breakeven_pass")


# ─── Pin hold / break evaluation ──────────────────────────────────────────────

PIN_HOLD = "hold"
PIN_BREAK_COMPACTION = "break_compaction"
PIN_BREAK_OVERFLOW = "break_context_overflow"
PIN_BREAK_BREAKEVEN = "break_breakeven"
PIN_BREAK_SCORE_MARGIN = "break_score_margin"
PIN_BREAK_STALE = "break_stale"


@dataclass(frozen=True)
class PinEvaluation:
    hold: bool
    reason: str
    switch_to: Optional[ModelProfile] = None


def evaluate_pin(
    pin: SessionPin,
    pinned_model: Optional[ModelProfile],
    best_alternative: Optional[ModelProfile],
    *,
    compaction_flag: bool = False,
    pinned_fits_context: bool = True,
    dwell_turns: int = 3,
    switch_margin: float = 0.25,
    pinned_score: float = 0.0,
    alternative_score: float = 0.0,
    estimated_input_tokens: int = 0,
    warm_prefix_tokens: int = 0,
) -> PinEvaluation:
    """Decide whether a session pin holds for this turn.

    Break conditions (first match wins):
      1. compaction — the conversation was compacted; prefix cache is gone
      2. context overflow — pinned model can no longer fit the request
      3. voluntary switch past the dwell guard that clears the economics gate:
         cache-breakeven when both models declare ``cost_per_1m``, otherwise a
         composite-score margin (``switch_margin``)
    """
    if compaction_flag:
        return PinEvaluation(False, PIN_BREAK_COMPACTION, best_alternative)
    if pinned_model is None or not pinned_model.healthy:
        return PinEvaluation(False, PIN_BREAK_STALE, best_alternative)
    if not pinned_fits_context:
        return PinEvaluation(False, PIN_BREAK_OVERFLOW, best_alternative)
    if best_alternative is None or best_alternative.id == pin.pinned_model_id:
        return PinEvaluation(True, PIN_HOLD)
    if pin.turns_held < max(0, dwell_turns):
        return PinEvaluation(True, PIN_HOLD)

    savings = compute_marginal_switch_savings(pinned_model, best_alternative, estimated_input_tokens)
    if savings is not None:
        result = evaluate_cache_breakeven_for_prefix(
            savings,
            warm_prefix_tokens,
            pinned_model.cost_per_1m or 0.0,
            best_alternative.cost_per_1m or 0.0,
        )
        if result.should_switch:
            return PinEvaluation(False, PIN_BREAK_BREAKEVEN, best_alternative)
        return PinEvaluation(True, PIN_HOLD)

    if alternative_score - pinned_score > switch_margin:
        return PinEvaluation(False, PIN_BREAK_SCORE_MARGIN, best_alternative)
    return PinEvaluation(True, PIN_HOLD)
