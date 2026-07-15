"""Turn-local fast-mode policy.

Hermes stores the user's fast-mode preference on ``agent.service_tier`` for
backward compatibility with the existing Normal/Fast implementation.  The
``auto`` and ``cold`` values are Hermes policies, not API service tiers:
``auto`` opens the configured fast window on every user turn, while ``cold``
opens it only on the first turn of a logical session.

The policy only returns an ephemeral copy of ``request_overrides``.  It never
mutates the conversation, system prompt, tool schemas, or the agent's persisted
override map, preserving prompt-cache stability across the tool loop.
"""

from __future__ import annotations

import math
import time
from typing import Any


DEFAULT_FAST_AUTO_ON_SECONDS = 60.0


def normalize_fast_auto_on_seconds(value: Any) -> float:
    """Return a positive finite cutoff, falling back to the 60-second default."""
    try:
        cutoff = float(value)
    except (TypeError, ValueError):
        return DEFAULT_FAST_AUTO_ON_SECONDS
    if isinstance(value, bool) or not math.isfinite(cutoff) or cutoff <= 0:
        return DEFAULT_FAST_AUTO_ON_SECONDS
    return cutoff


def _has_prior_session_activity(history: Any) -> bool:
    """Return whether a transcript already contains a conversational turn.

    System-only history is setup for a new session, not evidence of a prior
    user turn.  Any user, assistant, or tool row is conservatively treated as
    existing session activity, including compressed or partially recovered
    transcripts.
    """
    if not isinstance(history, (list, tuple)):
        return False
    return any(
        isinstance(message, dict)
        and message.get("role") in {"user", "assistant", "tool"}
        for message in history
    )


def begin_fast_mode_turn(
    agent: Any,
    conversation_history: Any = None,
    *,
    now: float | None = None,
) -> None:
    """Resolve fast-window eligibility at one user-turn boundary.

    ``cold`` derives its state from durable transcript history so recreating an
    agent process for an existing session does not open another fast window.
    The live message list is a fallback for callers that omit explicit history.
    """
    mode = getattr(agent, "service_tier", None)
    agent._fast_mode_turn_mode = mode
    eligible = mode == "auto"
    if mode == "cold":
        history = conversation_history
        if history is None:
            history = getattr(agent, "_session_messages", None)
        eligible = not _has_prior_session_activity(history)

    agent._fast_mode_turn_eligible = eligible
    agent._fast_mode_turn_started_at = (
        (time.monotonic() if now is None else now) if eligible else None
    )


def invalidate_fast_mode_turn(agent: Any) -> None:
    """Prevent a live policy change from reusing another mode's turn clock."""
    agent._fast_mode_turn_mode = None
    agent._fast_mode_turn_eligible = False
    agent._fast_mode_turn_started_at = None


def effective_request_overrides(
    agent: Any, *, now: float | None = None
) -> dict[str, Any]:
    """Resolve request overrides for the model call starting now.

    Explicit Normal/Fast behavior is unchanged.  Dynamic modes remove any stale
    fast-only key from the copied override map, then re-add the active model's
    provider-specific fast override while the current turn is eligible and
    inside the cutoff.
    """
    overrides = dict(getattr(agent, "request_overrides", {}) or {})
    mode = getattr(agent, "service_tier", None)
    if mode not in {"auto", "cold"}:
        return overrides

    overrides.pop("service_tier", None)
    overrides.pop("speed", None)

    turn_mode = getattr(agent, "_fast_mode_turn_mode", None)
    if turn_mode is not None and turn_mode != mode:
        return overrides

    current = time.monotonic() if now is None else now
    started_at = getattr(agent, "_fast_mode_turn_started_at", None)
    if not isinstance(started_at, (int, float)):
        if mode == "cold" or getattr(agent, "_fast_mode_turn_eligible", None) is False:
            return overrides
        started_at = current
        agent._fast_mode_turn_eligible = True
        agent._fast_mode_turn_started_at = started_at

    cutoff = normalize_fast_auto_on_seconds(
        getattr(agent, "fast_auto_on_seconds", DEFAULT_FAST_AUTO_ON_SECONDS)
    )
    elapsed = max(0.0, current - float(started_at))
    if elapsed <= cutoff:
        from hermes_cli.models import resolve_fast_mode_overrides

        fast_overrides = resolve_fast_mode_overrides(getattr(agent, "model", None))
        if fast_overrides:
            overrides.update(fast_overrides)
    return overrides


def effective_fast_mode_overrides(
    agent: Any, *, now: float | None = None
) -> dict[str, Any]:
    """Return only provider fast-tier keys from the effective request policy."""
    effective = effective_request_overrides(agent, now=now)
    return {
        key: effective[key]
        for key in ("service_tier", "speed")
        if key in effective
    }


def revalidate_fast_mode_request(agent: Any, api_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Re-resolve a dynamic fast policy immediately before provider dispatch."""
    mode = getattr(agent, "service_tier", None)
    if mode not in {"auto", "cold"}:
        return api_kwargs

    kwargs = dict(api_kwargs)
    kwargs.pop("service_tier", None)
    kwargs.pop("speed", None)
    overrides = effective_fast_mode_overrides(agent)

    if getattr(agent, "api_mode", None) == "anthropic_messages":
        from agent.anthropic_adapter import _apply_fast_mode_to_kwargs

        return _apply_fast_mode_to_kwargs(
            kwargs,
            enabled=overrides.get("speed") == "fast",
            model=getattr(agent, "model", "") or "",
            base_url=getattr(agent, "_anthropic_base_url", None),
            is_oauth=bool(getattr(agent, "_is_anthropic_oauth", False)),
            drop_context_1m_beta=bool(
                getattr(agent, "_oauth_1m_beta_disabled", False)
            ),
        )

    kwargs.update(overrides)
    return kwargs
