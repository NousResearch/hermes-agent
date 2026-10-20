"""Ticker downtime windows: the gap between a ticker's last heartbeat and its next startup.

A one-shot due while no ticker was running was never *missed* — nothing was alive to fire it
(host powered off overnight, gateway stopped, laptop asleep past the window). The due scan's
one-shot grace gate already makes that distinction for a slot skipped by an unwritable store
(``store_health.outage_covers``): skipped, not missed, so it fires once on recovery. Downtime is
the same shape with a different cause, and the evidence is the ticker heartbeat marker — stamped
every loop, so a due instant later than the last stamp proves nothing was looping then.

The window must be captured at startup, BEFORE the first heartbeat of the new ticker overwrites
the marker, which is why this is a record set by the scheduler's boot path rather than something
the due scan can derive on its own.

Firing a wall-clock one-shot hours late is a judgement call, not a default: the window only
releases the grace gate when the operator opts in with ``cron.catch_up_missed_oneshots``.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class DowntimeWindow:
    """Epoch seconds of a store's ticker gap: last heartbeat before the gap, first stamp after."""

    since: float
    until: float


_windows: dict[str, DowntimeWindow] = {}
_lock = threading.Lock()


def record_startup_gap(cron_dir: Path, last_heartbeat: Optional[float], now: float,
                       min_gap: float) -> Optional[DowntimeWindow]:
    """Record (and return) the downtime window a starting ticker just ended, or None.

    ``last_heartbeat`` is the marker's stamp read before this ticker writes its own; None (no
    marker — a store that has never been ticked) is not evidence of downtime. A gap at or under
    ``min_gap`` is ordinary tick slip, not downtime.
    """
    if last_heartbeat is None or now - last_heartbeat <= min_gap:
        return None
    window = DowntimeWindow(since=last_heartbeat, until=now)
    with _lock:
        _windows[_key(cron_dir)] = window
    return window


def covers(cron_dir: Path, due_at: float, grace: float) -> bool:
    """Whether a run due at ``due_at`` fell inside this store's recorded ticker downtime.

    ``grace`` widens the window at the start: a slot due just before the last heartbeat is still
    one the dying ticker demonstrably never got to.
    """
    with _lock:
        window = _windows.get(_key(cron_dir))
    return window is not None and window.since - grace <= due_at <= window.until


def forget_stores(stores: Iterable[str]) -> None:
    """Drop the windows of ``stores`` (``store_health.store_key`` values) this process no longer
    ticks, so a departed profile cannot keep releasing its grace gate."""
    stores = set(stores)
    with _lock:
        for store in [s for s in _windows if s in stores]:
            del _windows[store]


def _key(cron_dir: Path) -> str:
    """Record key of a store: its resolved path. Identical to ``store_health``'s key (and to what
    ``store_health.store_key`` yields for the same home), so ``forget_stores`` can be driven by the
    same departed-store set."""
    return os.path.realpath(cron_dir)
