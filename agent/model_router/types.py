"""Core types for the staged model-routing pipeline.

Ported from pi-smart-router (domain/types) and adapted to Hermes: the router
stays pure by default — no credential discovery, no network unless a stage is
explicitly enabled — and every decision is made once at the turn boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ─── Tiers ────────────────────────────────────────────────────────────────────

TIER_LOCAL = "local"
TIER_ECONOMICAL = "economical"
TIER_FRONTIER = "frontier"

_TIER_ALIASES = {
    "local": TIER_LOCAL,
    "zero-tier": TIER_LOCAL,
    "economical": TIER_ECONOMICAL,
    "economical-cloud": TIER_ECONOMICAL,
    "frontier": TIER_FRONTIER,
    "frontier-cloud": TIER_FRONTIER,
}


def normalize_tier(value: object, *, default: str = TIER_ECONOMICAL) -> str:
    """Map config tier names (Hermes or pi-smart-router style) to Hermes tiers."""
    if isinstance(value, str):
        tier = _TIER_ALIASES.get(value.strip().lower())
        if tier:
            return tier
    return default


# ─── Turn types ───────────────────────────────────────────────────────────────

TURN_TOOL_RESULT = "tool_result"
TURN_PLANNING = "planning"
TURN_SUBAGENT = "subagent"
TURN_MAIN_LOOP = "main_loop"
TURN_UNKNOWN = "unknown"
TURN_TYPES = (TURN_TOOL_RESULT, TURN_PLANNING, TURN_SUBAGENT, TURN_MAIN_LOOP, TURN_UNKNOWN)


@dataclass(frozen=True)
class Message:
    """Minimal message envelope used by routing classifiers."""

    role: str
    content: str = ""
    is_error: Optional[bool] = None
    status: Optional[int] = None


# ─── Fleet ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelProfile:
    """Rich candidate profile. Superset of the legacy ``Candidate``."""

    id: str
    provider: str = ""
    tier: str = TIER_ECONOMICAL
    context_window: int = 0  # 0 = unknown window → treated as fitting
    reasoning: bool = False
    vision: bool = False
    quality: float = 0.5  # relative routing score, not a benchmark claim
    cost: float = 0.5  # relative routing score 0..1, not billing data
    cost_per_1m: Optional[float] = None  # USD per 1M input tokens (breakeven gate)
    est_latency_ms: float = 0.0
    verbosity: float = 1.0
    healthy: bool = True


@dataclass(frozen=True)
class CandidateScore:
    model_id: str
    score: float = 0.0
    shortfall: float = 0.0
    rejected_reason: Optional[str] = None


# ─── Request / decision ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class RoutingRequest:
    prompt_text: str
    session_id: str = ""
    messages: tuple = ()
    turn_type: Optional[str] = None  # classified when None
    has_images: bool = False
    estimated_input_tokens: Optional[int] = None
    compaction_flag: bool = False
    force_model_id: Optional[str] = None

    def estimated_tokens(self) -> int:
        if self.estimated_input_tokens is not None:
            return max(0, int(self.estimated_input_tokens))
        if not self.prompt_text:
            return 0
        return max(1, -(-len(self.prompt_text) // 4))  # ceil(len/4)


@dataclass(frozen=True)
class RoutingDecision:
    selected_model: str
    stage: str
    reason_code: str
    explanation: str = ""
    suggestion: str = ""
    rejected: tuple = ()
    candidates: tuple = ()
    turn_type: str = TURN_UNKNOWN
    routing_latency_ms: float = 0.0
    pinned: bool = False
    features: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StageResult:
    decided: bool
    decision: Optional[RoutingDecision] = None


# ─── Session pin ──────────────────────────────────────────────────────────────

PIN_REASON_AUTO = "auto"
PIN_REASON_LOOP_ESCALATION = "loop_escalation"
PIN_REASON_FLIP_FLOP = "flip_flop"


@dataclass(frozen=True)
class SessionPin:
    session_id: str
    pinned_model_id: str
    pin_reason: str = PIN_REASON_AUTO
    turns_held: int = 0
    consecutive_tool_failures: int = 0
    last_tool_failure_signature: Optional[str] = None
    updated_at: float = 0.0
