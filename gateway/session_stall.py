"""Gateway session stall notification policy (#72016 item 2).

Consumes the shared activity observation contract from
``agent.session_activity`` / ``AIAgent.get_activity_summary()``
(#72039) as the **single progress source**. This module owns only the
notify-once policy for "pending inbound + stale progress"; it does not
invent a parallel progress clock from turn-start or inbound event
timestamps.

Boundaries (keep separate):
- ``gateway/shutdown_watchdog.py`` — process / event-loop liveness
- ``gateway/delivery_ledger.py`` — outbound delivery obligations
- Pending inbound here is a stall *policy gate* (queued follow-up exists),
  not an outbound obligation and not a progress timestamp.

Notification / timeout / kill / retry policy stay in their own components;
the shared contract remains observation-only (timestamp + bounded
description + provenance).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional


def should_emit_session_stall_notification(
    *,
    timeout_seconds: float,
    idle_seconds: Optional[float],
    has_pending_inbound: bool,
    already_notified: bool,
) -> bool:
    """Return True when a stall warning should be sent for this session."""
    if timeout_seconds <= 0:
        return False
    if not has_pending_inbound:
        return False
    if already_notified:
        return False
    if idle_seconds is None:
        return False
    return idle_seconds >= timeout_seconds


def should_clear_session_stall_notification(
    *,
    timeout_seconds: float,
    idle_seconds: Optional[float],
    has_pending_inbound: bool,
) -> bool:
    """Return True when a prior stall notice may be cleared (episode ended)."""
    if not has_pending_inbound:
        return True
    if timeout_seconds <= 0:
        return True
    # Unknown progress: hold the latch. Do not treat observation gaps as recovery.
    if idle_seconds is None:
        return False
    return idle_seconds < timeout_seconds


def format_session_stall_notification(idle_seconds: float) -> str:
    """User-facing stall warning (ASCII minutes; matches issue #72016 copy)."""
    mins = max(1, int(idle_seconds // 60))
    return (
        f"⚠️ Agent session appears stalled (last activity {mins} min ago). "
        f"Try /new to reset."
    )


def resolve_session_idle_seconds_from_activity(
    activity: Optional[Mapping[str, Any]],
    *,
    now: Optional[float] = None,
) -> Optional[float]:
    """Idle seconds from a shared activity snapshot only (#72039 contract).

    Prefers ``seconds_since_activity`` when present and finite; otherwise
    derives from ``last_activity_at`` / ``last_activity_ts``. Returns
    ``None`` when there is no usable progress timestamp — callers must
    not fall back to turn-start or pending-inbound clocks.
    """
    if not activity:
        return None

    elapsed = activity.get("seconds_since_activity")
    if elapsed is not None:
        try:
            idle = float(elapsed)
        except (TypeError, ValueError):
            idle = None
        else:
            if math.isfinite(idle):
                if idle < 0:
                    return 0.0
                return idle
            # Non-finite: fall through to last_activity_at / last_activity_ts

    ts = activity.get("last_activity_at")
    if ts is None:
        ts = activity.get("last_activity_ts")
    if ts is None:
        return None
    try:
        when = float(ts)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(when):
        return None

    if now is None:
        import time as _time

        clock = float(_time.time())
    else:
        clock = float(now)
    idle = clock - when
    if idle < 0:
        return 0.0
    return idle
