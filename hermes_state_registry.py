from __future__ import annotations

import contextlib
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

from hermes_state_common import stat_db_file_identity as _stat_db_file_identity

if TYPE_CHECKING:
    from hermes_state import SessionDB

logger = logging.getLogger(__name__)


class _TeardownBarrier:
    __slots__ = ("event", "pending")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.pending = 0


class _Generation:
    __slots__ = ("path", "db", "refcount", "identity", "retired")

    def __init__(self, path: Path, db: "SessionDB", identity: Optional[Tuple[int, int]]) -> None:
        self.path = path
        self.db = db
        self.refcount = 1
        self.identity = identity
        self.retired = False


_lock = threading.Lock()
_generations: Dict[Path, _Generation] = {}
_retired: Dict[int, _Generation] = {}
_opening: Dict[Path, threading.Event] = {}
_tearing_down: Dict[Path, _TeardownBarrier] = {}
_path_lifecycle_locks: Dict[Path, threading.Lock] = {}


def _open_session_db(path: Path) -> "SessionDB":
    from hermes_state import SessionDB
    return SessionDB(db_path=path)


def _teardown(db: "SessionDB") -> None:
    with contextlib.suppress(Exception):
        db._shared_registry_owned = False
    try:
        db.close()
    except Exception:
        logger.debug("Error closing shared SessionDB", exc_info=True)


def _path_lifecycle_lock_locked(path: Path) -> threading.Lock:
    lock = _path_lifecycle_locks.get(path)
    if lock is None:
        lock = threading.Lock()
        _path_lifecycle_locks[path] = lock
    return lock


def _admit_teardown_locked(path: Path) -> _TeardownBarrier:
    barrier = _tearing_down.get(path)
    if barrier is None:
        barrier = _tearing_down[path] = _TeardownBarrier()
    barrier.pending += 1
    return barrier


def _finish_teardown(path: Path, barrier: _TeardownBarrier) -> None:
    with _lock:
        barrier.pending -= 1
        if barrier.pending > 0:
            return
        if _tearing_down.get(path) is barrier:
            del _tearing_down[path]
        barrier.event.set()


def _teardown_generation(
    path: Path,
    db: "SessionDB",
    *,
    barrier: Optional[_TeardownBarrier] = None,
) -> None:
    with _lock:
        lifecycle_lock = _path_lifecycle_lock_locked(path)
    try:
        with lifecycle_lock:
            _teardown(db)
    finally:
        if barrier is not None:
            _finish_teardown(path, barrier)


def _db_path_of(db: "SessionDB") -> Optional[Path]:
    path = getattr(db, "db_path", None)
    if path is None:
        return None
    try:
        return Path(path)
    except (TypeError, ValueError):
        return None


def _finish_opening(path: Path, opening: threading.Event) -> None:
    if _opening.get(path) is opening:
        del _opening[path]
    opening.set()


def acquire(db_path: Optional[Path] = None) -> "SessionDB":
    from hermes_state import _default_db_path

    if db_path is not None:
        raw_path = Path(db_path)
    else:
        raw_path = Path(_default_db_path())

    try:
        path = raw_path.resolve()
    except OSError:
        path = raw_path

    while True:
        wait_for: Optional[threading.Event] = None
        with _lock:
            generation = _generations.get(path)
            if generation is not None:
                current = _stat_db_file_identity(path)
                if current is not None and generation.identity is not None and current != generation.identity:
                    generation.retired = True
                    del _generations[path]
                    _retired[id(generation.db)] = generation
                else:
                    generation.refcount += 1
                    return generation.db
            
            teardown = _tearing_down.get(path)
            if teardown is not None:
                wait_for = teardown.event
            else:
                opening = _opening.get(path)
                if opening is None:
                    opening = _opening[path] = threading.Event()
                    lifecycle_lock = _path_lifecycle_lock_locked(path)
                else:
                    wait_for = opening

        if wait_for is not None:
            wait_for.wait()
            continue

        try:
            with lifecycle_lock:
                db = _open_session_db(path)
                db._shared_registry_owned = True
                identity = _stat_db_file_identity(path)
        except BaseException:
            with _lock:
                _finish_opening(path, opening)
            raise

        with _lock:
            teardown = _tearing_down.get(path)
            if teardown is None:
                existing = _generations.get(path)
                if existing is not None:
                    existing.refcount += 1
                    winner = existing.db
                else:
                    _generations[path] = _Generation(path, db, identity)
                    winner = db
            _finish_opening(path, opening)

        if teardown is not None:
            _teardown_generation(path, db)
            teardown.event.wait()
            continue

        if winner is not db:
            _teardown_generation(path, db)

        return winner


def release(db: "SessionDB") -> bool:
    if db is None:
        return False
    
    key = id(db)
    teardown_barrier: Optional[_TeardownBarrier] = None

    with _lock:
        generation = _retired.get(key)
        if generation is None:
            path = _db_path_of(db)
            if path is None:
                return False
            generation = _generations.get(path)
            if generation is None or generation.db is not db:
                return False

        generation.refcount -= 1
        needs_teardown = generation.refcount <= 0

        if needs_teardown:
            if generation.retired:
                _retired.pop(key, None)
            elif _generations.get(generation.path) is generation:
                del _generations[generation.path]
            
            teardown_barrier = _admit_teardown_locked(generation.path)

    if needs_teardown:
        _teardown_generation(generation.path, db, barrier=teardown_barrier)

    return True


def close_all() -> int:
    teardown_barriers: Dict[Path, _TeardownBarrier] = {}
    
    with _lock:
        active_teardowns = list(_tearing_down.values())
        generations = list(_generations.values()) + list(_retired.values())
        
        all_paths = {g.path for g in generations}
        for path in all_paths:
            teardown_barriers[path] = _admit_teardown_locked(path)
            
        _generations.clear()
        _retired.clear()
        
        for generation in generations:
            generation.retired = True

    by_path: Dict[Path, List[_Generation]] = {}
    for generation in generations:
        by_path.setdefault(generation.path, []).append(generation)

    for path, path_generations in by_path.items():
        with _lock:
            lifecycle_lock = _path_lifecycle_lock_locked(path)
        try:
            with lifecycle_lock:
                for generation in path_generations:
                    _teardown(generation.db)
        finally:
            _finish_teardown(path, teardown_barriers[path])

    for barrier in active_teardowns:
        barrier.event.wait()

    return len(generations)


def live_shared_session_dbs() -> List["SessionDB"]:
    with _lock:
        return [g.db for g in _generations.values() if not g.retired]


@contextlib.contextmanager
def borrow_live_shared_session_dbs() -> Iterator[List["SessionDB"]]:
    with _lock:
        borrowed_generations = [
            g for g in _generations.values() if not g.retired
        ]
        borrowed = [g.db for g in borrowed_generations]
        for g in borrowed_generations:
            g.refcount += 1
    try:
        yield borrowed
    finally:
        for db in reversed(borrowed):
            release(db)


def stats() -> Dict[str, int]:
    with _lock:
        return {
            "live_generations": len(_generations),
            "retired_generations": len(_retired),
            "total_refcounts": sum(g.refcount for g in _generations.values()),
        }


def release_or_close(db: "SessionDB") -> None:
    if not release(db):
        try:
            db.close()
        except Exception:
            logger.debug("release_or_close fallback close failed", exc_info=True)


# ---- BEGIN PLUGIN-COMPAT ----
def close_shared_session_dbs() -> int:
    return close_all()

def get_shared_session_db(db_path: Optional[Path] = None) -> "SessionDB":
    return acquire(db_path)

def release_shared_session_db(db: "SessionDB") -> bool:
    return release(db)
# ---- END PLUGIN-COMPAT ----
