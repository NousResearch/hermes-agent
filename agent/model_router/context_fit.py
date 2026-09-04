"""Context-fit gate and overflow fallback — port of routing/context-fit.ts.

Filters fleet models whose context window cannot accommodate the estimated
input token count (with a configurable safety margin), and resolves a
structured fallback when economical tiers cannot fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import (
    CandidateScore,
    ModelProfile,
    RoutingRequest,
    TIER_FRONTIER,
)

CONTEXT_FIT_EXCEEDED = "context_fit_exceeded"
OUTPUT_HEADROOM_EXCEEDED = "output_headroom_exceeded"
CONTEXT_OVERFLOW_SAME_PROVIDER_FALLBACK = "context_overflow_same_provider_fallback"
CONTEXT_OVERFLOW_FRONTIER_FALLBACK = "context_overflow_frontier_fallback"
CONTEXT_OVERFLOW_NO_FIT = "context_overflow_no_fit"

DEFAULT_SAFETY_MARGIN = 0.9
# Pi's delegation floor is intentionally small: it prevents a candidate whose
# input exactly fills the context window from being selected without reserving
# a large, provider-specific generation budget.
DEFAULT_MIN_OUTPUT_TOKENS = 256


@dataclass(frozen=True)
class ContextFitFilterResult:
    effective_fleet: tuple
    rejected: tuple


@dataclass(frozen=True)
class ContextOverflowResult:
    kind: str  # "selected" | "no_fit"
    model: Optional[ModelProfile]
    reason_code: str


def model_fits_context(
    profile: ModelProfile,
    estimated_input_tokens: int,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    min_output_tokens: int = 0,
) -> bool:
    """Whether input plus the requested output floor fits the safe window.

    ``min_output_tokens=0`` preserves the historical context-only predicate
    for callers outside the router pipeline. Unknown windows remain eligible.
    """
    if not profile.context_window:
        return True  # unknown window → retained
    effective_limit = int(profile.context_window * safety_margin)
    return estimated_input_tokens + max(0, int(min_output_tokens)) <= effective_limit


def select_largest_window_model(candidates) -> Optional[ModelProfile]:
    """Select the healthy model with the largest declared context window."""
    best = None
    best_window = -1
    for model in candidates:
        if not model.healthy:
            continue
        window = model.context_window if model.context_window else 1 << 62
        if window > best_window:
            best_window = window
            best = model
    return best


def select_lowest_cost_model(candidates) -> Optional[ModelProfile]:
    """Select the lowest-cost healthy model (ties broken by model id)."""
    healthy = [m for m in candidates if m.healthy]
    if not healthy:
        return None
    return sorted(healthy, key=lambda m: (m.cost, m.id))[0]


def resolve_context_overflow_fallback(
    fleet,
    estimated_input_tokens: int,
    preferred_provider: Optional[str] = None,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    min_output_tokens: int = 0,
) -> ContextOverflowResult:
    """Escalate when economical/pinned models cannot fit context.

    1. Same-provider largest-fit model
    2. Cheapest frontier model that fits
    3. Structured no-fit (never dispatch undersized)
    """
    fits = lambda m: model_fits_context(
        m, estimated_input_tokens, safety_margin, min_output_tokens
    )

    if preferred_provider:
        same_provider = [m for m in fleet if m.provider == preferred_provider and fits(m)]
        model = select_largest_window_model(same_provider)
        if model:
            return ContextOverflowResult("selected", model, CONTEXT_OVERFLOW_SAME_PROVIDER_FALLBACK)

    frontier = [m for m in fleet if m.tier == TIER_FRONTIER and fits(m)]
    model = select_lowest_cost_model(frontier)
    if model:
        return ContextOverflowResult("selected", model, CONTEXT_OVERFLOW_FRONTIER_FALLBACK)

    return ContextOverflowResult("no_fit", None, CONTEXT_OVERFLOW_NO_FIT)


def filter_fleet_by_context_fit(
    fleet,
    request: RoutingRequest,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    min_output_tokens: int = 0,
) -> ContextFitFilterResult:
    """Remove fleet entries whose window cannot fit the estimated input.

    Honors ``force_model_id`` by leaving the fleet unchanged. Models without
    declared limits are retained (unknown window).
    """
    if request.force_model_id:
        return ContextFitFilterResult(tuple(fleet), ())

    estimated = request.estimated_tokens()
    effective = []
    rejected = []
    output_floor = max(0, int(min_output_tokens))
    for profile in fleet:
        if model_fits_context(profile, estimated, safety_margin, output_floor):
            effective.append(profile)
            continue
        effective_limit = int(profile.context_window * safety_margin)
        input_fits = estimated <= effective_limit
        required = estimated + output_floor
        rejected.append(
            CandidateScore(
                model_id=profile.id,
                score=0.0,
                shortfall=float(max(0, required - effective_limit)),
                rejected_reason=(
                    OUTPUT_HEADROOM_EXCEEDED if input_fits else CONTEXT_FIT_EXCEEDED
                ),
            )
        )
    return ContextFitFilterResult(tuple(effective), tuple(rejected))
