"""Fresh-store preparation and construction locks for multi-actor backend tests.

Call ``prepare_fresh_store`` once per fresh path in the parent to create and close the store before actors open it concurrently, so the test races operations rather than schema construction, including SQLite FTS initialization. The helper itself does not enforce a single call. Where parent initialization is impossible (a scope store created lazily on first touch), ``construction_lock(store_path)`` serializes every context entry for that path, not just its first construction. There is no global construction lock, so distinct paths can construct concurrently.

Accounting: each successful prepare, completed construction-lock context and successful ``serialized_child_open`` call is tallied per path with elapsed monotonic time, including lock waits. ``init_stats()`` returns a snapshot, and ``reset_init_stats()`` clears both the tally and the path-lock registry (call it from a fixture, per test or per suite).

Multiprocess note: the per-path locks and tallies here are per-process. ``serialized_child_open`` accepts a parent-supplied ``multiprocessing`` lock around construction only; operations on the returned handle run outside that lock. Its tally stays in the calling process rather than being aggregated across children.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

__all__ = [
    "construction_lock",
    "init_stats",
    "prepare_fresh_store",
    "reset_init_stats",
    "serialized_child_open",
]

_REGISTRY_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.Lock] = {}
_STATS: dict[str, dict[str, float]] = {}


def _tally(path: str, kind: str, elapsed: float) -> None:
    with _REGISTRY_GUARD:
        entry = _STATS.setdefault(path, {"prepares": 0, "constructions": 0, "wall_clock_s": 0.0})
        entry[kind] += 1
        entry["wall_clock_s"] += elapsed


def prepare_fresh_store(db_path: Path, open_store: Callable[[Path], Any]) -> None:
    """Parent-side init: construct the store at ``db_path`` once and close it.

    ``open_store`` opens the backend's store at the path (for SQLite: ``lambda p: open_persistence(profile_scope(home=p.parent), eager=True)``); the returned handle must expose ``close()``. After this, actors open the path concurrently with no lock.
    """
    started = time.monotonic()
    handle = open_store(db_path)
    handle.close()
    _tally(str(db_path), "prepares", time.monotonic() - started)


@contextmanager
def construction_lock(db_path: Path) -> Iterator[None]:
    """Serialize the FIRST construction of one store path (lazy scope stores only).

    Per-path, never global: distinct paths get distinct locks, so unrelated stores construct concurrently. Hold it around construction only; raced operations run outside it.
    """
    key = str(db_path)
    with _REGISTRY_GUARD:
        lock = _PATH_LOCKS.setdefault(key, threading.Lock())
    started = time.monotonic()
    with lock:
        yield
    _tally(key, "constructions", time.monotonic() - started)


def serialized_child_open(
    db_path: Path,
    open_store: Callable[[Path], Any],
    shared_lock: Optional[Any] = None,
) -> Any:
    """Child-actor open: construct a handle on an already-initialized store.

    With ``shared_lock`` (a ``multiprocessing`` lock passed by the parent) construction is serialized across processes, for stores whose parent init is impossible; without it the open runs bare, which is correct once ``prepare_fresh_store`` ran. Operations on the returned handle stay fully concurrent either way.
    """
    started = time.monotonic()
    if shared_lock is not None:
        with shared_lock:
            handle = open_store(db_path)
    else:
        handle = open_store(db_path)
    _tally(str(db_path), "constructions", time.monotonic() - started)
    return handle


def init_stats() -> dict[str, dict[str, float]]:
    """Per-path prepare/construction counts and wall clock, for the receipt."""
    with _REGISTRY_GUARD:
        return {path: dict(entry) for path, entry in _STATS.items()}


def reset_init_stats() -> None:
    """Start a fresh tally (fixture-scoped; the accounting is additive otherwise)."""
    with _REGISTRY_GUARD:
        _STATS.clear()
        _PATH_LOCKS.clear()
