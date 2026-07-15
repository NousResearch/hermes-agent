"""Turn-local fast-mode policy.

Hermes stores the user's fast-mode preference on ``agent.service_tier`` for
backward compatibility with the existing Normal/Fast implementation.  The
``auto`` value is a Hermes policy, not an API service tier: requests that begin
within the configured window receive the provider-specific fast override and
later requests in the same turn do not.

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


def begin_fast_mode_turn(agent: Any, *, now: float | None = None) -> None:
    """Reset the auto-fast clock for one user turn."""
    if getattr(agent, "service_tier", None) == "auto":
        agent._fast_mode_turn_started_at = time.monotonic() if now is None else now
    else:
        agent._fast_mode_turn_started_at = None


def effective_request_overrides(
    agent: Any, *, now: float | None = None
) -> dict[str, Any]:
    """Resolve request overrides for the model call starting now.

    Explicit Normal/Fast behavior is unchanged.  Auto mode removes any stale
    fast-only key from the copied override map, then re-adds the active model's
    provider-specific fast override while the turn is inside the cutoff.
    """
    overrides = dict(getattr(agent, "request_overrides", {}) or {})
    if getattr(agent, "service_tier", None) != "auto":
        return overrides

    overrides.pop("service_tier", None)
    overrides.pop("speed", None)

    current = time.monotonic() if now is None else now
    started_at = getattr(agent, "_fast_mode_turn_started_at", None)
    if not isinstance(started_at, (int, float)):
        started_at = current
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
