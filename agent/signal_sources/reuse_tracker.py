"""Track skill reuse → outcome mapping.

When the same skill is invoked multiple times across turns, the
outcome of the *next* invocation after the current one tells us whether
the skill was reusable. This is a strong positive signal: a skill that
gets reused successfully is contributing long-term value, not just
one-shot answers.

Design:
- Stateful: maintains a per-skill history of ``(timestamp, success)``
  pairs. The producer wires the actual storage.
- Pure helpers: :func:`mark_invocation`, :func:`lookup_reuse_outcome`
  operate on a list so the producer can persist it however it wants.

Usage::

    from agent.signal_sources.reuse_tracker import (
        mark_invocation,
        lookup_reuse_outcome,
    )

    history = []  # producer persists this
    history = mark_invocation(history, timestamp=time.time(), success=True)
    reused = lookup_reuse_outcome(history, after_timestamp=t0)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class ReuseEntry:
    """One row in a skill's reuse history."""

    timestamp: float
    success: bool


def mark_invocation(
    history: List[ReuseEntry],
    *,
    timestamp: Optional[float] = None,
    success: bool = True,
) -> List[ReuseEntry]:
    """Append a new reuse record.

    Returns a new list; does not mutate the input.
    """
    if timestamp is None:
        timestamp = time.time()
    return [*history, ReuseEntry(timestamp=timestamp, success=success)]


def lookup_reuse_outcome(
    history: Iterable[ReuseEntry],
    *,
    after_timestamp: float,
    immediate_only: bool = True,
) -> Optional[bool]:
    """Return the outcome of the next reuse after ``after_timestamp``.

    Args:
        history: Per-skill reuse history.
        after_timestamp: Reference timestamp; the function returns the
            outcome of the *next* entry whose timestamp is greater than
            this value.
        immediate_only: If True, return only the *next* entry (the one
            immediately following). If False, walk the entire tail and
            return the *majority* outcome. Default True — the closest
            reuse is the strongest signal.

    Returns:
        ``True`` if the next reuse succeeded, ``False`` if it failed,
        ``None`` if no reuse happened after ``after_timestamp``.
    """
    tail = sorted(
        (e for e in history if e.timestamp > after_timestamp),
        key=lambda e: e.timestamp,
    )
    if not tail:
        return None
    if immediate_only:
        return tail[0].success
    successes = sum(1 for e in tail if e.success)
    failures = len(tail) - successes
    if successes > failures:
        return True
    if failures > successes:
        return False
    return None  # tie — ambiguous, do not signal either way


def merge_history(
    *histories: Iterable[ReuseEntry],
) -> List[ReuseEntry]:
    """Concatenate and sort by timestamp.

    Useful when merging per-session histories into a global view.
    """
    out: List[ReuseEntry] = []
    for h in histories:
        out.extend(h)
    out.sort(key=lambda e: e.timestamp)
    return out


__all__ = [
    "ReuseEntry",
    "mark_invocation",
    "lookup_reuse_outcome",
    "merge_history",
]
