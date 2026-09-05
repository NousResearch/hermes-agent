"""Parsing and evaluation for per-task Kanban dispatch time gates.

The dispatcher evaluates gates from UTC epoch seconds on each tick. Daily
windows are evaluated in their IANA timezone's current wall clock, so repeated
and skipped local times around DST need no timer rescheduling.
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_WINDOW_RE = re.compile(
    r"^(?P<start>[01]\d|2[0-3]):(?P<start_min>[0-5]\d)-"
    r"(?P<end>[01]\d|2[0-3]):(?P<end_min>[0-5]\d)\s+"
    r"(?P<timezone>\S+)$"
)


def normalize_dispatch_after(value: Optional[str | int | float]) -> Optional[int]:
    """Return a timezone-aware ISO instant as UTC epoch seconds.

    Numeric epoch values are accepted for internal callers and persisted rows.
    Fractional instants round up so a task can never dispatch early.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("dispatch_after must be a timezone-aware ISO timestamp")
    if isinstance(value, (int, float)):
        epoch = math.ceil(value)
    else:
        raw = str(value).strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw)
        except ValueError as exc:
            raise ValueError(f"invalid dispatch_after timestamp: {value!r}") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("dispatch_after must be a timezone-aware ISO timestamp")
        epoch = math.ceil(parsed.timestamp())
    if epoch < 0:
        raise ValueError("dispatch_after must not be before the Unix epoch")
    return epoch


def normalize_dispatch_window(value: Optional[str]) -> Optional[str]:
    """Validate and canonicalize ``HH:MM-HH:MM IANA/Zone``."""
    if value is None or not str(value).strip():
        return None
    raw = str(value).strip()
    match = _WINDOW_RE.fullmatch(raw)
    if match is None:
        raise ValueError(
            "dispatch_window must use 'HH:MM-HH:MM IANA/Timezone' "
            "(for example '23:00-05:30 America/Chicago')"
        )
    start = f"{match['start']}:{match['start_min']}"
    end = f"{match['end']}:{match['end_min']}"
    if start == end:
        raise ValueError("dispatch_window start and end must differ")
    timezone_name = match["timezone"]
    try:
        ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown IANA timezone in dispatch_window: {timezone_name!r}") from exc
    return f"{start}-{end} {timezone_name}"


def dispatch_gate_open(
    dispatch_after: Optional[int], dispatch_window: Optional[str], *, now: Optional[int] = None,
) -> bool:
    """Whether both configured gates permit dispatch at ``now``."""
    current = int(time.time() if now is None else now)
    if dispatch_after is not None and current < int(dispatch_after):
        return False
    if not dispatch_window:
        return True

    normalized = normalize_dispatch_window(dispatch_window)
    assert normalized is not None
    times, timezone_name = normalized.split(" ", 1)
    start_text, end_text = times.split("-", 1)
    start_hour, start_minute = (int(part) for part in start_text.split(":"))
    end_hour, end_minute = (int(part) for part in end_text.split(":"))
    local = datetime.fromtimestamp(current, ZoneInfo(timezone_name))
    minute = local.hour * 60 + local.minute
    start = start_hour * 60 + start_minute
    end = end_hour * 60 + end_minute
    if start < end:
        return start <= minute < end
    return minute >= start or minute < end
