"""Capability-aware routing helpers for orchestration/delegation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from agent.model_metadata import MINIMUM_CONTEXT_LENGTH


@dataclass
class Capability:
    name: str
    provider: str = ""
    model: str = ""
    context_length: int = 0
    toolsets: set[str] = field(default_factory=set)
    cost_rank: int = 100
    latency_rank: int = 100
    local: bool = False
    healthy: bool = True
    reason: str = ""

    @classmethod
    def from_agent(cls, agent, *, name: str = "parent") -> "Capability":
        return cls(
            name=name,
            provider=getattr(agent, "provider", "") or "",
            model=getattr(agent, "model", "") or "",
            context_length=int(getattr(agent, "context_length", 0) or 0),
            toolsets=set(getattr(agent, "enabled_toolsets", None) or []),
            healthy=True,
        )


@dataclass
class RouteDecision:
    selected: Capability
    skipped: list[Capability] = field(default_factory=list)
    degraded_to_parent: bool = False


def _eligible(
    cap: Capability,
    *,
    required_context: int,
    required_toolsets: Sequence[str],
) -> tuple[bool, str]:
    if not cap.healthy:
        return False, cap.reason or "unhealthy"
    if cap.context_length and cap.context_length < required_context:
        return False, f"context {cap.context_length} < required {required_context}"
    if required_toolsets and cap.toolsets:
        missing = sorted(set(required_toolsets) - cap.toolsets)
        if missing:
            return False, f"missing toolsets: {', '.join(missing)}"
    return True, ""


def choose_delegate_route(
    parent: Capability,
    candidates: Iterable[Capability],
    *,
    required_context: int = MINIMUM_CONTEXT_LENGTH,
    required_toolsets: Optional[Sequence[str]] = None,
) -> RouteDecision:
    """Choose the cheapest healthy child capability that can run Hermes.

    If no candidate qualifies, degrade to the parent instead of throwing deep in
    child construction.  This is deliberately simple and deterministic; richer
    provider health/cost models can feed the Capability list later.
    """
    skipped: list[Capability] = []
    eligible: list[Capability] = []
    req_tools = list(required_toolsets or [])
    for cap in candidates:
        ok, reason = _eligible(cap, required_context=required_context, required_toolsets=req_tools)
        if ok:
            eligible.append(cap)
        else:
            skipped.append(Capability(**{**cap.__dict__, "reason": reason}))
    if eligible:
        eligible.sort(key=lambda c: (c.cost_rank, c.latency_rank, c.name))
        return RouteDecision(selected=eligible[0], skipped=skipped)
    fallback = Capability(**{**parent.__dict__, "reason": "no eligible delegate candidates; using parent"})
    return RouteDecision(selected=fallback, skipped=skipped, degraded_to_parent=True)


def validate_delegate_capability(
    *,
    provider: str,
    model: str,
    context_length: int,
    required_context: int = MINIMUM_CONTEXT_LENGTH,
) -> Optional[str]:
    if context_length and context_length < required_context:
        return (
            f"Delegation target {provider or 'unknown'}/{model or 'unknown'} has "
            f"context_length={context_length:,}, below Hermes minimum "
            f"{required_context:,}. Pick a larger-context delegation model or "
            "unset delegation.provider/model to inherit the parent."
        )
    return None
