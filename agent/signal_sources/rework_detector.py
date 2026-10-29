"""Detect rework: how many retries of the same task_id happened recently.

A "retry" is when the same ``task_id`` appears in a turn log within the
rework window (default 10 minutes). Multiple retries in that window
indicate the user wasn't satisfied with the first attempt — a strong
negative signal on the skill that produced the original output.

This is a stateful detector: it tracks every ``task_id`` it sees and
returns the count for a given task_id within the window. Older
``task_id`` entries are evicted after the window expires.

Design:
- Pure-ish: takes a list of ``(task_id, timestamp)`` tuples and returns
  a count. The producer wires the actual storage and time source.
- Configurable window so callers can tune per environment.

Usage::

    from agent.signal_sources.rework_detector import count_recent
    # events: list of (task_id, timestamp) within the window
    count = count_recent(events, target_task_id="t-123", window_sec=600)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ReworkEvent:
    """A single (task_id, timestamp) event from a turn log."""

    task_id: str
    timestamp: float


def filter_window(
    events: Iterable[ReworkEvent],
    *,
    now: float,
    window_sec: float,
) -> List[ReworkEvent]:
    """Keep only events with timestamp >= now - window_sec.

    Stable order: input order is preserved.
    """
    cutoff = now - window_sec
    return [e for e in events if e.timestamp >= cutoff]


def count_recent(
    events: Iterable[ReworkEvent],
    target_task_id: str,
    *,
    now: Optional[float] = None,
    window_sec: float = 600.0,
) -> int:
    """Count how many ``target_task_id`` events fall in the recent window.

    Args:
        events: Iterable of ``ReworkEvent`` from a turn log.
        target_task_id: The task_id to count.
        now: Reference time (seconds since epoch). ``None`` uses
            ``time.time()``. Tests should pass an explicit value.
        window_sec: Window size in seconds. Default 10 minutes.

    Returns:
        Integer count (>=0). Zero if no recent events match.

    Note:
        The returned count includes the *current* attempt itself if it
        is in the input list. Callers that want the count of *retries*
        (excluding the original) should subtract 1, or call this
        function with the events list minus the current event.
    """
    if now is None:
        now = time.time()
    if window_sec <= 0:
        return 0
    cutoff = now - window_sec
    return sum(
        1 for e in events if e.task_id == target_task_id and e.timestamp >= cutoff
    )


def count_rework_retries(
    events: Iterable[ReworkEvent],
    target_task_id: str,
    *,
    now: Optional[float] = None,
    window_sec: float = 600.0,
) -> int:
    """Count *retries* (excludes the current attempt itself).

    Convenience wrapper around :func:`count_recent` that subtracts the
    current event from the input list before counting. Pass the full
    events list including the current attempt.
    """
    if now is None:
        now = time.time()
    cutoff = now - window_sec
    same_id_in_window = [
        e for e in events if e.task_id == target_task_id and e.timestamp >= cutoff
    ]
    return max(len(same_id_in_window) - 1, 0)


__all__ = [
    "ReworkEvent",
    "filter_window",
    "count_recent",
    "count_rework_retries",
]
