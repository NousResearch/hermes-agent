"""Elapsed-time logic for chronoception — wall-clock only, no external deps.

Tracks the wall-clock time of each session's previous turn and, on the next
turn, renders a compact clock line (current time + delta) and/or a one-shot
notice when the agent resumed after a long idle gap. Wall clock (not monotonic)
so a machine that slept counts the sleep. A backward clock (skew/NTP) yields a
non-positive delta and is ignored.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

_FENCE_OPEN = "<turn-clock>"
_FENCE_CLOSE = "</turn-clock>"
_NOTE = (
    "[Runtime timing for your own reasoning, not the user's words. "
    "Judge staleness with it; do not narrate it.]"
)

_LOCK = threading.Lock()
_LAST_WALL: "OrderedDict[str, float]" = OrderedDict()
_MAX_SESSIONS = 2048


def _humanize(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)}s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = minutes / 60.0
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24.0:.1f} days"


def _last_completion(session_id: str) -> Optional[float]:
    """Return the previous completed-turn wall stamp without modifying it."""
    with _LOCK:
        return _LAST_WALL.get(session_id)


def record_completion(session_id: str, wall_now: Optional[float] = None) -> None:
    """Record when a turn finished so agent runtime is not counted as idle."""
    stamp = time.time() if wall_now is None else wall_now
    with _LOCK:
        _LAST_WALL[session_id] = stamp
        _LAST_WALL.move_to_end(session_id)
        while len(_LAST_WALL) > _MAX_SESSIONS:
            _LAST_WALL.popitem(last=False)


def _wrap(body: str, max_chars: int) -> str:
    text = f"{_FENCE_OPEN}\n{_NOTE}\n\n{body}\n{_FENCE_CLOSE}"
    if len(text) <= max_chars:
        return text
    overhead = len(_FENCE_OPEN) + len(_FENCE_CLOSE) + len(_NOTE) + 4
    body = body[: max(20, max_chars - overhead - 1)] + "…"
    return f"{_FENCE_OPEN}\n{_NOTE}\n\n{body}\n{_FENCE_CLOSE}"


def build(session_id: str, settings: Dict[str, Any]) -> Optional[str]:
    """Return the fenced timing block for this turn, or ``None`` to stay silent."""
    wall_now = time.time()
    prev = _last_completion(session_id)
    delta = (wall_now - prev) if prev is not None else None

    parts = []
    if settings["clock"]:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(wall_now))
        if delta is not None and delta > 0:
            parts.append(f"clock {ts}, +{_humanize(delta)} since your last turn.")
        else:
            parts.append(f"clock {ts} (first turn this session).")

    if delta is not None and delta > 0 and delta >= float(settings["gap_report_seconds"]):
        if settings["clock"]:
            parts.append(
                "You were idle that long; time-sensitive state (files, jobs, "
                "health, the clock) may have moved on."
            )
        else:
            parts.append(
                f"~{_humanize(delta)} idle since your last turn; "
                "time-sensitive state may have moved on."
            )

    if not parts:
        return None
    return _wrap(" ".join(parts), int(settings["max_chars"]))


def reset_for_tests() -> None:
    with _LOCK:
        _LAST_WALL.clear()
