"""
Achilli Tendon -- Subagent orchestration health monitor.

Hooks into subagent_stop to track child lifecycle events. Exposes
tendon_health and monitor_subagents tools enriched with session telemetry.

Known limitation: pre_tool_call does NOT fire for delegate_task. Tendon
observes children post-hoc via subagent_stop only.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session ledger
# ---------------------------------------------------------------------------

_LEDGER: List[Dict[str, Any]] = []
_MAX_LEDGER = 1000
_STUCK_THRESHOLD = 300  # seconds


def _get_stuck_threshold() -> int:
    import os
    try:
        return int(os.environ.get("ACHILLI_TENDON_STUCK_THRESHOLD", "300"))
    except (ValueError, TypeError):
        return 300


def _get_max_ledger() -> int:
    import os
    try:
        return int(os.environ.get("ACHILLI_TENDON_MAX_LEDGER", "1000"))
    except (ValueError, TypeError):
        return 1000


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _on_subagent_stop(
    child_role: Optional[str] = None,
    child_summary: Optional[str] = None,
    child_status: Optional[str] = None,
    duration_ms: int = 0,
    parent_session_id: Optional[str] = None,
    **_: Any,
) -> None:
    """Record every subagent_stop event in the session ledger."""
    entry = {
        "timestamp": time.time(),
        "child_role": child_role,
        "child_summary": (child_summary or "")[:200],
        "child_status": child_status,
        "duration_ms": duration_ms,
        "parent_session_id": parent_session_id,
    }
    _LEDGER.append(entry)
    max_len = _get_max_ledger()
    if len(_LEDGER) > max_len:
        del _LEDGER[: len(_LEDGER) - max_len]
    logger.debug(
        "tendon: recorded subagent_stop role=%s status=%s duration=%dms",
        child_role, child_status, duration_ms,
    )


def _on_session_end(**_: Any) -> None:
    """Check for stuck children at end of each turn."""
    if not _LEDGER:
        return
    # Check if the most recent child completed abnormally or is still running
    recent = _LEDGER[-1]
    duration_s = recent.get("duration_ms", 0) / 1000.0
    threshold = _get_stuck_threshold()
    if duration_s > threshold:
        logger.warning(
            "tendon: child '%s' took %.1fs (threshold %ds) -- possible stuck subagent",
            recent.get("child_role", "?"),
            duration_s,
            threshold,
        )


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def _tendon_health_tool(*_, **__) -> dict:
    """Return aggregated health dashboard for the current session."""
    total = len(_LEDGER)
    completed = sum(1 for e in _LEDGER if e["child_status"] == "completed")
    failed = sum(1 for e in _LEDGER if e["child_status"] == "failed")
    interrupted = sum(1 for e in _LEDGER if e["child_status"] == "interrupted")
    durations = [e["duration_ms"] for e in _LEDGER if e["duration_ms"] > 0]
    avg_ms = sum(durations) / len(durations) if durations else 0
    max_ms = max(durations) if durations else 0

    longest_role = None
    if _LEDGER:
        longest = max(_LEDGER, key=lambda e: e.get("duration_ms", 0))
        longest_role = longest.get("child_role")

    return {
        "status": "healthy" if failed == 0 else "degraded",
        "session_children_total": total,
        "completed": completed,
        "failed": failed,
        "interrupted": interrupted,
        "avg_duration_ms": round(avg_ms),
        "max_duration_ms": max_ms,
        "longest_running_role": longest_role,
        "stuck_threshold_s": _get_stuck_threshold(),
    }


def _monitor_subagents_tool(*_, **__) -> dict:
    """Return live status of tracked subagents in this session."""
    total = len(_LEDGER)
    by_status: Dict[str, int] = {}
    by_role: Dict[str, int] = {}
    for e in _LEDGER:
        status = e.get("child_status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        role = e.get("child_role") or "unnamed"
        by_role[role] = by_role.get(role, 0) + 1

    last_event = _LEDGER[-1] if _LEDGER else None
    return {
        "tracked_events": total,
        "by_status": by_status,
        "by_role": by_role,
        "last_event": last_event,
        "ledger_capacity": _get_max_ledger(),
    }


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    ctx.register_hook("subagent_stop", _on_subagent_stop)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_tool(
        name="tendon_health",
        description="Return subagent orchestration health dashboard with session telemetry (achilli-tendon)",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=_tendon_health_tool,
    )
    ctx.register_tool(
        name="monitor_subagents",
        description="Return live status of tracked subagents with per-role and per-status breakdowns (achilli-tendon)",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=_monitor_subagents_tool,
    )
