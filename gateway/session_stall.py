"""Gateway session stall notification policy.

Consumes ``AIAgent.get_activity_summary()`` as the **single progress source** and owns
only the notify-once policy for "pending inbound + stale progress"; it never derives a
parallel progress clock from turn-start or inbound timestamps.  Pending inbound is a
stall *policy gate*, not a delivery obligation and not process liveness.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Optional


def should_emit_session_stall_notification(
    *, timeout_seconds: float, idle_seconds: Optional[float], has_pending_inbound: bool,
    already_notified: bool,
) -> bool:
    """Return True when a stall warning should be sent for this session."""
    return (
        timeout_seconds > 0 and has_pending_inbound and not already_notified
        and idle_seconds is not None and idle_seconds >= timeout_seconds
    )


def should_clear_session_stall_notification(
    *, timeout_seconds: float, idle_seconds: Optional[float], has_pending_inbound: bool,
) -> bool:
    """Return True when a prior stall notice may be cleared (episode ended)."""
    if not has_pending_inbound or timeout_seconds <= 0:
        return True
    # Unknown progress holds the latch: observation gaps are not recovery.
    return idle_seconds is not None and idle_seconds < timeout_seconds


def format_session_stall_notification(idle_seconds: float) -> str:
    """User-facing stall warning (ASCII minutes).

    See #72016.
    """
    mins = max(1, int(idle_seconds // 60))
    return f"⚠️ Agent session appears stalled (last activity {mins} min ago). Try /new to reset."


def _finite_float(value: Any) -> Optional[float]:
    """``value`` as a finite float, or None (bools are rejected as non-numeric)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def resolve_session_idle_seconds_from_activity(
    activity: Optional[Mapping[str, Any]], *, now: Optional[float] = None,
) -> Optional[float]:
    """Idle seconds from a shared activity snapshot: a finite ``seconds_since_activity``, else
    derived from ``last_activity_at`` / ``last_activity_ts``.  None when there is no usable
    progress timestamp — callers must not fall back to turn-start or inbound clocks.

    See #72039.
    """
    if not activity:
        return None
    idle = _finite_float(activity.get("seconds_since_activity"))
    if idle is not None:
        return max(0.0, idle)
    ts = activity.get("last_activity_at")
    when = _finite_float(activity.get("last_activity_ts") if ts is None else ts)
    if when is None:
        return None
    return max(0.0, float(time.time() if now is None else now) - when)
