"""Hermes model router — staged, cache-aware, turn-boundary routing.

Public API:

- Legacy deterministic router (unchanged behavior): ``Candidate``, ``Features``,
  ``RouteDecision``, ``extract_features``, ``route_turn``.
- Staged pipeline ported from pi-smart-router: ``RouterPipeline``,
  ``RoutingRequest``, ``RoutingDecision``, ``ModelProfile``,
  ``pipeline_from_config``.

Invariants: routing decides once per turn, before agent construction; pure by
default (no network unless ``local_zero`` is explicitly enabled); explicit
candidates only; any failure falls back to the current model; deterministic
tie-breaks.
"""
from __future__ import annotations

# Legacy API — preserved verbatim for existing callers and tests.
from .legacy import (
    Candidate,
    Features,
    RouteDecision,
    extract_features,
    route_turn,
)

# Staged pipeline API.
from .types import (
    CandidateScore,
    Message,
    ModelProfile,
    RoutingDecision,
    RoutingRequest,
    SessionPin,
    TIER_ECONOMICAL,
    TIER_FRONTIER,
    TIER_LOCAL,
    normalize_tier,
)
from .pipeline import (
    MODE_AUTO,
    MODE_OFF,
    MODE_SUGGEST,
    RouterConfig,
    RouterPipeline,
    pipeline_from_config,
    profile_from_config,
    router_config_from_dict,
)

__all__ = [
    # legacy
    "Candidate",
    "Features",
    "RouteDecision",
    "extract_features",
    "route_turn",
    # pipeline
    "CandidateScore",
    "Message",
    "ModelProfile",
    "RoutingDecision",
    "RoutingRequest",
    "SessionPin",
    "RouterConfig",
    "RouterPipeline",
    "pipeline_from_config",
    "profile_from_config",
    "router_config_from_dict",
    "normalize_tier",
    "TIER_LOCAL",
    "TIER_ECONOMICAL",
    "TIER_FRONTIER",
    "MODE_OFF",
    "MODE_SUGGEST",
    "MODE_AUTO",
]
