"""Safe default selector — port of pipeline/safe-default.ts.

Selects the first healthy model of the configured safe tier (default
economical), falling back to frontier only when no safe-tier model is
available. Context-fit aware when a request is provided. Never throws.
"""
from __future__ import annotations

from typing import Optional

from .context_fit import DEFAULT_SAFETY_MARGIN, model_fits_context
from .types import ModelProfile, RoutingRequest, TIER_ECONOMICAL, TIER_FRONTIER


def safe_default(
    models,
    request: Optional[RoutingRequest] = None,
    *,
    safe_tier: str = TIER_ECONOMICAL,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    min_output_tokens: int = 0,
) -> Optional[ModelProfile]:
    """Pick the deterministic safe model, or None when nothing is viable."""
    estimated = request.estimated_tokens() if request else 0

    def fits(model: ModelProfile) -> bool:
        if not model.healthy:
            return False
        if not request:
            return True
        return model_fits_context(
            model, estimated, safety_margin, min_output_tokens
        )

    ordered = sorted(models, key=lambda m: m.id)
    for model in ordered:
        if model.tier == safe_tier and fits(model):
            return model
    if safe_tier != TIER_FRONTIER:
        for model in ordered:
            if model.tier == TIER_FRONTIER and fits(model):
                return model
    for model in ordered:
        if fits(model):
            return model
    return None
