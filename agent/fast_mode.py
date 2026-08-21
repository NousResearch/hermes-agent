"""Provider-neutral, turn-local fast-mode policy.

``normal`` and ``fast`` keep their existing behavior. ``auto`` opens a
bounded fast window on every user turn; ``cold`` opens the same window only
for the first logical-session turn. The policy changes request metadata only:
conversation messages, prompts, tools, and persisted overrides stay stable.
"""

from __future__ import annotations

import math
import time
from typing import Any


DEFAULT_FAST_AUTO_ON_SECONDS = 60.0
DYNAMIC_FAST_MODES = frozenset({"auto", "cold"})


def normalize_fast_auto_on_seconds(value: Any) -> float:
    """Return a positive finite cutoff, or the 60-second default."""
    try:
        cutoff = float(value)
    except (TypeError, ValueError):
        return DEFAULT_FAST_AUTO_ON_SECONDS
    if isinstance(value, bool) or not math.isfinite(cutoff) or cutoff <= 0:
        return DEFAULT_FAST_AUTO_ON_SECONDS
    return cutoff


def has_prior_session_activity(history: Any) -> bool:
    """Whether persisted history proves a logical turn already occurred."""
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
    """Resolve dynamic-mode eligibility at a user-turn boundary."""
    mode = getattr(agent, "service_tier", None)
    agent._fast_mode_turn_mode = mode
    eligible = mode == "auto"
    if mode == "cold":
        history = conversation_history
        if history is None:
            history = getattr(agent, "_session_messages", None)
        eligible = not has_prior_session_activity(history)

    agent._fast_mode_turn_eligible = eligible
    agent._fast_mode_turn_started_at = (
        (time.monotonic() if now is None else now) if eligible else None
    )


def invalidate_fast_mode_turn(agent: Any) -> None:
    """Prevent a live policy change from reusing a prior mode's clock."""
    agent._fast_mode_turn_mode = None
    agent._fast_mode_turn_eligible = False
    agent._fast_mode_turn_started_at = None


def effective_request_overrides(
    agent: Any, *, now: float | None = None
) -> dict[str, Any]:
    """Return a request-local copy with the current fast policy applied."""
    overrides = dict(getattr(agent, "request_overrides", {}) or {})
    mode = getattr(agent, "service_tier", None)
    if mode not in DYNAMIC_FAST_MODES:
        return overrides

    overrides.pop("service_tier", None)
    overrides.pop("speed", None)

    if getattr(agent, "_fast_mode_turn_mode", None) != mode:
        return overrides
    if not getattr(agent, "_fast_mode_turn_eligible", False):
        return overrides

    started_at = getattr(agent, "_fast_mode_turn_started_at", None)
    if not isinstance(started_at, (int, float)):
        return overrides

    current = time.monotonic() if now is None else now
    cutoff = normalize_fast_auto_on_seconds(
        getattr(agent, "fast_auto_on_seconds", DEFAULT_FAST_AUTO_ON_SECONDS)
    )
    if max(0.0, current - float(started_at)) <= cutoff:
        from hermes_cli.models import resolve_fast_mode_overrides

        fast_overrides = resolve_fast_mode_overrides(
            getattr(agent, "model", None),
            provider=getattr(agent, "provider", None),
            api_mode=getattr(agent, "api_mode", None),
            base_url=(
                getattr(agent, "_anthropic_base_url", None)
                or getattr(agent, "base_url", None)
            ),
        )
        if fast_overrides:
            overrides.update(fast_overrides)
    return overrides


def effective_fast_mode_overrides(
    agent: Any, *, now: float | None = None
) -> dict[str, Any]:
    """Return only provider fast-tier keys from the effective policy."""
    effective = effective_request_overrides(agent, now=now)
    return {
        key: effective[key] for key in ("service_tier", "speed") if key in effective
    }


def revalidate_fast_mode_request(
    agent: Any, api_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Re-resolve a dynamic policy immediately before provider dispatch."""
    if getattr(agent, "service_tier", None) not in DYNAMIC_FAST_MODES:
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
            drop_context_1m_beta=bool(getattr(agent, "_oauth_1m_beta_disabled", False)),
        )

    kwargs.update(overrides)
    return kwargs
