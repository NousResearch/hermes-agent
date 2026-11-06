"""Scheduled, pre-announced gateway restarts (issue #135051).

A ``shutdown +N`` / ``wall`` equivalent for Hermes: instead of learning about a
restart at the instant it interrupts a turn, the gateway can be told ahead of
time that it will restart at an announced moment. The scheduler lives in the
running gateway process (``GatewayShutdownMixin.schedule_restart``); the CLI
reaches it through the control socket (``schedule-restart`` / ``cancel-restart``
verbs) and the same notice plumbing the shutdown message uses, just earlier.

This module holds the pure data/format pieces only — no gateway imports — so it
stays importable from both the gateway and the CLI without cycles.
"""
from __future__ import annotations

import dataclasses
import datetime
import re
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    import asyncio

# Re-broadcast leads before the restart, mirroring ``wall`` re-announcements
# as the deadline approaches.
REBROADCAST_LEAD_SECONDS = (60.0, 10.0)

# Floor for how far out a restart may be scheduled: shorter is indistinguishable
# from an immediate restart and races the announce/drain dance.
MIN_SCHEDULED_LEAD_SECONDS = 5.0

# Ceiling so a typo'd offset cannot park the gateway for months.
MAX_SCHEDULED_LEAD_SECONDS = 7 * 24 * 3600

_HHMM_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


@dataclasses.dataclass
class ScheduledRestart:
    """One armed, pre-announced restart.

    ``wall_time`` is an absolute epoch timestamp (the displayed time); ``task``
    is the armed loop task that announces and then triggers the existing
    ``request_restart()`` drain path.
    """

    wall_time: float
    reason: str = ""
    task: Optional[asyncio.Task] = None
    # Fire-and-forget announcement tasks (schedule-time / cancel notices) so a
    # cancellation can interrupt an in-flight delivery and never let a
    # "will restart" arrive after the "cancelled" notice.
    announce_tasks: list = dataclasses.field(default_factory=list)

    def remaining_seconds(self) -> float:
        return max(self.wall_time - time.time(), 0.0)


def format_countdown(seconds: float) -> str:
    """Human-readable countdown (``65`` → ``1 minute``, ``30`` → ``30 seconds``)."""
    s = round(seconds)
    if s < 60:
        return f"{s} second" + ("s" if s != 1 else "")
    if s < 3600:
        m = s // 60
        return f"{m} minute" + ("s" if m != 1 else "")
    h = s // 3600
    return f"{h} hour" + ("s" if h != 1 else "")


def format_wallclock(ts: float) -> str:
    """Wall-clock display: ``HH:MM`` today, ``YYYY-MM-DD HH:MM`` otherwise."""
    local = datetime.datetime.fromtimestamp(ts)
    if local.date() == datetime.datetime.now().date():
        return local.strftime("%H:%M")
    return local.strftime("%Y-%m-%d %H:%M")


def parse_wallclock_string(raw: str) -> Optional[float]:
    """Parse a wall-clock target into an epoch timestamp, or None.

    Accepts ``HH:MM`` (the next occurrence in local time), ISO-8601 date-times,
    and bare epoch seconds.
    """
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return datetime.datetime.fromisoformat(text.replace(" ", "T")).timestamp()
    except ValueError:
        pass
    match = _HHMM_RE.fullmatch(text)
    if match is not None:
        hour, minute = int(match.group(1)), int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            now = datetime.datetime.now()
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += datetime.timedelta(days=1)
            return target.timestamp()
    try:
        return float(text)
    except ValueError:
        return None
