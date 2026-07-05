"""Shared runtime evidence for guardrail and tool-call diagnostics.

The runtime guardrails need a single context-local place to stash turn/session
evidence so later checks can read the same facts without threading bespoke
state through every call site. A ``ContextVar`` keeps the evidence isolated per
turn while still flowing through nested tool calls and copied execution
contexts.
"""

from __future__ import annotations

import contextvars
from contextvars import Token
from typing import Any, Mapping

_RUNTIME_EVIDENCE: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "hermes_runtime_evidence",
    default=None,
)


def current_runtime_evidence() -> dict[str, Any] | None:
    """Return the current evidence mapping for this turn, if any."""
    return _RUNTIME_EVIDENCE.get()


def set_runtime_evidence(evidence: Mapping[str, Any] | None) -> Token:
    """Replace the current evidence mapping for this context."""
    return _RUNTIME_EVIDENCE.set(dict(evidence) if evidence is not None else None)


def update_runtime_evidence(**fields: Any) -> dict[str, Any]:
    """Merge fields into the current evidence mapping and return it."""
    evidence = dict(_RUNTIME_EVIDENCE.get() or {})
    for key, value in fields.items():
        if value is not None:
            evidence[key] = value
    _RUNTIME_EVIDENCE.set(evidence)
    return evidence


def seed_turn_runtime_evidence(*, latest_user_request: Any | None = None, **fields: Any) -> dict[str, Any]:
    """Start a fresh per-turn evidence mapping with the provided fields."""
    evidence: dict[str, Any] = {}
    if latest_user_request is not None:
        evidence["latest_user_request"] = latest_user_request
    for key, value in fields.items():
        if value is not None:
            evidence[key] = value
    _RUNTIME_EVIDENCE.set(evidence)
    return evidence


def clear_runtime_evidence() -> None:
    """Clear the current evidence mapping."""
    _RUNTIME_EVIDENCE.set(None)
