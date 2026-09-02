"""Bounded fast-mode windows (``/fast auto`` and ``/fast cold``).

``agent.service_tier`` is ``None`` (normal), ``"priority"`` (static fast),
``"auto"`` or ``"cold"``. The static value is pinned into
``agent.request_overrides`` at agent build time; the two bounded modes
instead open a wall-clock window at each user-turn boundary and layer the
provider's fast override onto the request kwargs only while it is open:

- ``auto`` — every user turn opens a window of ``agent.fast_auto_seconds``.
- ``cold`` — only the first turn of a session (no prior history) opens it.

Only per-request params (``service_tier`` / ``speed``) vary between requests;
the system prompt, tools, and messages are untouched, so the prompt cache is
preserved across the window boundary.
"""

from __future__ import annotations

import time
from typing import Any

BOUNDED_MODES = frozenset({"auto", "cold"})
DEFAULT_WINDOW_SECONDS = 60


def begin_turn(agent: Any, conversation_history: Any) -> None:
    """Open (or refuse) the fast window at a user-turn boundary."""
    mode = getattr(agent, "service_tier", None)
    agent._fast_until = 0.0
    if mode not in BOUNDED_MODES:
        return
    if mode == "cold" and any(
        isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool")
        for m in (conversation_history or ())
    ):
        return
    try:
        window = float(getattr(agent, "fast_auto_seconds", DEFAULT_WINDOW_SECONDS))
    except (TypeError, ValueError):
        window = DEFAULT_WINDOW_SECONDS
    agent._fast_until = time.monotonic() + max(window, 0.0)


def effective_request_overrides(agent: Any) -> dict[str, Any]:
    """``agent.request_overrides`` plus the fast override while the window is open."""
    overrides = dict(getattr(agent, "request_overrides", None) or {})
    if getattr(agent, "service_tier", None) not in BOUNDED_MODES:
        return overrides
    if time.monotonic() >= getattr(agent, "_fast_until", 0.0):
        return overrides
    from hermes_cli.models import resolve_fast_mode_overrides

    base_url = getattr(agent, "base_url", None)
    if getattr(agent, "api_mode", None) == "anthropic_messages":
        base_url = getattr(agent, "_anthropic_base_url", None) or base_url
    fast = resolve_fast_mode_overrides(
        getattr(agent, "model", None),
        provider=getattr(agent, "provider", None),
        base_url=base_url,
    )
    if fast:
        overrides.update(fast)
    return overrides
