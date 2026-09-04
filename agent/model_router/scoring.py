"""Multi-objective scoring — port of scoring/multi-objective.ts.

Re-ranks candidates using operator-configured frugality weights. At quality
parity (the capability gate has already run), the scorer penalizes cost,
latency, and verbosity to prefer cheaper/faster/leaner models:

    score = capability_score
          - lambda_cost      * norm_cost
          - lambda_latency   * norm_latency
          - lambda_verbosity * norm_verbosity

Normalization is min-max across the viable candidate set so penalties are
scale-invariant. Ties break deterministically by model id.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import CandidateScore, ModelProfile


@dataclass(frozen=True)
class FrugalityWeights:
    lambda_cost: float = 0.5
    lambda_latency: float = 0.1
    lambda_verbosity: float = 0.15


@dataclass(frozen=True)
class ScoredCandidate:
    model_id: str
    capability_score: float
    cost_penalty: float
    latency_penalty: float
    verbosity_penalty: float
    composite_score: float
    rejected_reason: Optional[str] = None


@dataclass(frozen=True)
class MultiObjectiveResult:
    selected: Optional[ScoredCandidate]
    candidates: tuple


_MIDPOINT = 0.5


def _range(values):
    values = list(values)
    if not values:
        return 0.0, 0.0
    return min(values), max(values)


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return _MIDPOINT
    return (value - lo) / (hi - lo)


def score_multi_objective(
    capability_scores,
    fleet,
    weights: FrugalityWeights = FrugalityWeights(),
) -> MultiObjectiveResult:
    """Score and re-rank candidates with cost/latency/verbosity penalties."""
    profile_map = {m.id: m for m in fleet}
    viable = [c for c in capability_scores if c.rejected_reason is None]
    rejected = [c for c in capability_scores if c.rejected_reason is not None]

    scored_rejected = tuple(
        ScoredCandidate(c.model_id, c.score, 0.0, 0.0, 0.0, 0.0, c.rejected_reason)
        for c in rejected
    )
    if not viable:
        return MultiObjectiveResult(None, scored_rejected)

    def metrics(c: CandidateScore):
        profile: Optional[ModelProfile] = profile_map.get(c.model_id)
        if profile is None:
            return 0.0, 0.0, 1.0
        cost = profile.cost_per_1m if profile.cost_per_1m is not None else profile.cost
        return cost, profile.est_latency_ms, profile.verbosity

    raw = [metrics(c) for c in viable]
    cost_lo, cost_hi = _range(m[0] for m in raw)
    lat_lo, lat_hi = _range(m[1] for m in raw)
    verb_lo, verb_hi = _range(m[2] for m in raw)

    scored = []
    for candidate, (cost, latency, verbosity) in zip(viable, raw):
        cost_penalty = weights.lambda_cost * _normalize(cost, cost_lo, cost_hi)
        latency_penalty = weights.lambda_latency * _normalize(latency, lat_lo, lat_hi)
        verbosity_penalty = weights.lambda_verbosity * _normalize(verbosity, verb_lo, verb_hi)
        composite = candidate.score - cost_penalty - latency_penalty - verbosity_penalty
        scored.append(
            ScoredCandidate(
                candidate.model_id,
                candidate.score,
                cost_penalty,
                latency_penalty,
                verbosity_penalty,
                composite,
            )
        )

    scored.sort(key=lambda s: (-s.composite_score, s.model_id))
    return MultiObjectiveResult(scored[0], tuple(scored) + scored_rejected)
