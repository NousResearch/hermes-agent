"""Mid-turn route-change gate for side-effecting tools.

During a turn, ``try_activate_fallback()`` swaps provider/model/base_url **in place**, and the
tool loop keeps running — a pending ``git push`` / DB write / sent message executes on a route
the caller never selected. The one-shot fallback notice is a display path only. Discarding the
turn's *answer* after the fact cannot un-send a message. (#117495)

The gate is a **provenance latch**: it captures the acting route at turn start and, at tool
dispatch, detects "route changed since the turn began". It does NOT interpret fallbacks —
``switch_model`` (manual /model), credential-pool rotation, and fallback activation all
consistently latch as "route changed", which is the honest semantic: the acting route changed
mid-turn, and side-effecting tools halt at the boundary. The fallback notice already tells the
user what changed; the gate decides whether tools may still run across it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Tools that cannot change the outside world: pure reads over the filesystem/web, pure
# computation, planning state. Anything NOT in this set is treated as potentially side-effecting.
# Conservative widen-and-shift policy: an unlisted new read-only tool is gated on fallback turns
# (fail-safe, not fail-open) until it's added here.
_READ_ONLY_TOOLS = frozenset({
    "read_file", "search_files", "vision_analyze", "session_search", "memory_recall",
    "memory_search", "todo_list", "web_search", "web_extract", "browser_snapshot",
    "browser_observe", "browser_page_info", "browser_screenshot", "tool_search",
    "tool_describe",
})

# Keys that identify the acting route on AIAgent. base_url is included: two providers can share
# a model slug over different gateways, and a key→key rotation (same slug, different credential)
# is exactly the class of mid-turn substitution this gate is about. (issue #117495)
_ROUTE_KEYS = ("model", "provider", "base_url")


def capture_route_snapshot(agent) -> dict:
    """The acting route: model slug, provider name, base_url (may be None)."""
    return {k: getattr(agent, k, None) for k in _ROUTE_KEYS}


def latch_turn_route(agent) -> None:
    """Snapshot the acting route at turn start (called once per turn, before the loop)."""
    agent._turn_route_snapshot = capture_route_snapshot(agent)


def _gate_enabled(agent) -> bool:
    """fallback.halt_on_route_change (default False) — strictly opt-in."""
    try:
        from hermes_cli.config import load_config
        return bool((load_config() or {}).get("fallback", {}).get("halt_on_route_change", False))
    except Exception:
        return False


def check_tool_dispatch(agent, function_name: str):
    """Return a block message string when dispatch must halt, else None."""
    if function_name in _READ_ONLY_TOOLS:
        return None
    if not _gate_enabled(agent):
        return None
    snapshot = getattr(agent, "_turn_route_snapshot", None)
    if not snapshot:
        return None
    current = capture_route_snapshot(agent)
    if current != snapshot:
        old = _format_route(snapshot)
        new = _format_route(current)
        logger.warning("Route changed mid-turn (%s -> %s); blocking side-effecting tool %r",
                       old, new, function_name)
        return (
            f"[ROUTE_CHANGED_BLOCKED] Tool '{function_name}' can change external state, but the "
            f"model route changed mid-turn ({old} -> {new}) and side-effecting tools are halted "
            f"under fallback.halt_on_route_change. Ask the user how to proceed, or continue "
            f"with read-only tools only."
        )
    return None


def _format_route(route: dict) -> str:
    model = route.get("model") or "?"
    provider = route.get("provider") or "?"
    base = route.get("base_url") or "?"
    return f"{model}@{provider} ({base})"