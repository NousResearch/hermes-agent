"""Process-wide shared SessionDB registry.

Each bare ``SessionDB()`` mints its own writer connection, lock, close-time WAL checkpoint
and token writer thread, and one connection's close-time checkpoint can race another's
growth (lost/reordered-page-write corruption). This module owns that boundary: one shared
``SessionDB`` per resolved path per process, refcounted, with generation-aware retirement
when the file is replaced (snapshot restore, recovery swap).

Lifecycle rules:
- ``acquire(path)`` returns the current generation for *path* and bumps its refcount.
- ``close()`` on a shared instance RELEASES one refcount instead of tearing the
  connection down: the registry owns the physical lifecycle and only closes on the
  final release, so legacy call sites return their reference instead of leaking it.
- ``release(db)`` decrements the generation *db was acquired from* (object-keyed, so an
  inode replacement cannot strand a still-owned generation); the final release of a
  retired generation tears it down.
- On inode change the old generation is RETIRED (never lent again) but stays alive until
  its holders release. If the replacement open fails the registry keeps NO path entry.
- All teardown happens OUTSIDE the registry lock: a final release's WAL checkpoint must
  never stall acquisition for every state.db.
- A final close/checkpoint is serialized with the next open for the same path; no new
  generation is published while the previous generation is still tearing down.
- A path can have SEVERAL closes admitted at once (the current generation's final release
  plus a retired generation's drain). The path barrier COUNTS them and is lifted only by
  the last one to settle, so neither ``acquire`` nor ``close_all`` can escape while any
  handle for that path is still inside checkpoint/WAL-unlink.
- Maintenance callers borrow handles with a temporary registry reference instead of
  iterating an unpinned snapshot.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

from hermes_state_common import stat_db_file_identity as _stat_db_file_identity

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typed only
    from hermes_state import SessionDB

logger = logging.getLogger(__name__)


class _TeardownBarrier:
    """Accounting for every admitted-but-unfinished physical close of one path.

    A path can own more than one close at a time: the current generation's final
    release and a retired generation's drain are admitted independently under
    ``_lock`` and only meet at the lifecycle mutex. One event per path is honest
    only if the LAST admitted teardown settles it. Signalling on the first lets
    ``close_all()`` return and ``acquire()`` publish a replacement while an older
    handle is still inside ``PRAGMA wal_checkpoint``/sidecar unlink -- the exact
    overlap (#102827) this registry exists to forbid.
    """

    __slots__ = ("event", "pending")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.pending = 0


class _Generation:
    """One shared SessionDB generation: instance, refcount, file identity."""

    __slots__ = ("path", "db", "refcount", "identity", "retired")

    def __init__(self, path: Path, db: "SessionDB", identity: Optional[Tuple[int, int]]) -> None:
        self.path = path
        self.db = db
        self.refcount = 1
        self.identity = identity
        self.retired = False


_lock = threading.Lock()
# path → live generation; retired generations move to _retired (keyed by id(db)) until
# their last holder releases.
_generations: Dict[Path, _Generation] = {}
_retired: Dict[int, _Generation] = {}
# Paths whose next generation is being constructed. Construction runs outside _lock
# (schema reconciliation can take seconds), but peers for the SAME file must wait or
# every cold caller opens its own writer before a winner is chosen.
_opening: Dict[Path, threading.Event] = {}
# A final close/checkpoint must finish before a replacement writer is opened
# for the same path. The barrier is admitted while holding _lock and lifted
# only after the LAST admitted physical teardown, so acquire cannot slip
# through the generation-removal/open gap and close_all cannot report a
# finished sweep over a close that is still running.
_tearing_down: Dict[Path, _TeardownBarrier] = {}
# Open and close are both performed outside _lock. This per-path mutex closes
# the race between checking _tearing_down and entering sqlite3.connect(),
# including retired-generation drains after an inode replacement.
_path_lifecycle_locks: Dict[Path, threading.Lock] = {}


def _open_session_db(path: Path) -> "SessionDB":
    """Construct the SessionDB for *path* (call-time import avoids cycles; tests patch this)."""
    from hermes_state import SessionDB

    return SessionDB(db_path=path)


def _teardown(db: "SessionDB") -> None:
    """Close a shared instance, clearing its registry-owned flag first."""
    with contextlib.suppress(Exception):
        db._shared_registry_owned = False
    try:
        db.close()
    except Exception:
        logger.debug("Error closing shared SessionDB", exc_info=True)


def _path_lifecycle_lock_locked(path: Path) -> threading.Lock:
    """Return the lifecycle mutex for *path* (caller holds ``_lock``)."""
    lock = _path_lifecycle_locks.get(path)
    if lock is None:
        lock = threading.Lock()
        _path_lifecycle_locks[path] = lock
    return lock


def _admit_teardown_locked(path: Path) -> _TeardownBarrier:
    """Register one pending physical close for *path* (caller holds ``_lock``).

    Admission shares the lock section that removes the generation, so a peer
    release, ``acquire`` or ``close_all`` taking the lock next always sees this
    teardown accounted for.
    """
    barrier = _tearing_down.get(path)
    if barrier is None:
        barrier = _tearing_down[path] = _TeardownBarrier()
    barrier.pending += 1
    return barrier


def _finish_teardown(path: Path, barrier: _TeardownBarrier) -> None:
    """Settle one admitted teardown; only the last one lifts the path barrier."""
    with _lock:
        barrier.pending -= 1
        if barrier.pending > 0:
            return
        if _tearing_down.get(path) is barrier:
            _tearing_down.pop(path, None)
        barrier.event.set()


def _teardown_generation(
    path: Path,
    db: "SessionDB",
    *,
    barrier: Optional[_TeardownBarrier] = None,
) -> None:
    """Close *db* under its path lifecycle mutex, then settle its barrier slot."""
    with _lock:
        lifecycle_lock = _path_lifecycle_lock_locked(path)
    try:
        with lifecycle_lock:
            _teardown(db)
    finally:
        if barrier is not None:
            _finish_teardown(path, barrier)


def _db_path_of(db: "SessionDB") -> Optional[Path]:
    """``Path(db.db_path)`` or None when absent/unconvertible."""
    path = getattr(db, "db_path", None)
    try:
        return None if path is None else Path(path)
    except (TypeError, ValueError):
        return None


def _finish_opening(path: Path, opening: threading.Event) -> None:
    """Drop the per-path construction marker and wake waiters (caller holds _lock)."""
    if _opening.get(path) is opening:
        _opening.pop(path, None)
    opening.set()


def acquire(db_path: Optional[Path] = None) -> "SessionDB":
    """Return the shared SessionDB for *db_path*, incrementing its refcount. If the file was
    replaced (different inode) since the generation opened, that generation is RETIRED
    but stays alive for its holders, and a fresh one is opened in its place. Raises
    whatever ``SessionDB.__init__`` raises; on a replacement-open failure the registry
    holds NO entry for the path."""
    from hermes_state import _default_db_path

    raw_path = Path(db_path) if db_path is not None else Path(_default_db_path())
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
                    # File replaced: retire this generation so it is never lent again, then elect one
                    # caller to open the replacement. It stays alive for its holders, tracked in
                    # ``_retired`` by ``id(db)`` so their releases find it after the path remaps.
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
            # Another caller is constructing or closing this path; wait without holding the
            # global lock. A failed opener signals too, so a waiter can retry.
            wait_for.wait()
            continue

        # Open OUTSIDE the registry lock; the per-path marker prevents redundant writers without
        # serialising other files, the lifecycle mutex keeps the open off a same-path close.
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
                if existing is not None:  # Defensive: installed by explicit registry manipulation mid-open.
                    existing.refcount += 1
                    winner = existing.db
                else:
                    _generations[path] = _Generation(path, db, identity)
                    winner = db
            _finish_opening(path, opening)
        if teardown is not None:
            # A shutdown or retired-generation final release was admitted while this opener was
            # constructing: never publish into that window — discard the speculative handle and
            # go round again once the barrier lifts.
            _teardown_generation(path, db)
            teardown.event.wait()
            continue
        if winner is not db:
            _teardown_generation(path, db)
        return winner


def release(db: "SessionDB") -> bool:
    """Decrement the refcount of a shared SessionDB. ``True`` if *db* was shared; ``False``
    if it is not registry-managed (caller owns close()). The final release tears the
    generation down OUTSIDE the registry lock. Lookup is object-keyed, so holders of an
    old generation release into its retired record, not into whatever the path names."""
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
                # Not shared (bare SessionDB()); the caller owns close().
                return False
        generation.refcount -= 1
        needs_teardown = generation.refcount <= 0
        if needs_teardown:
            if generation.retired:
                _retired.pop(key, None)
            elif _generations.get(generation.path) is generation:
                _generations.pop(generation.path, None)
            # A retired generation's drain is admitted too: it checkpoints and unlinks the same
            # sidecars as the current one, so a replacement writer must not open on top of it.
            teardown_barrier = _admit_teardown_locked(generation.path)
    # Teardown OUTSIDE the lock: stopping the token writer, WAL checkpoint and read-pool
    # drain must not block acquisition for every other state.db.
    if needs_teardown:
        _teardown_generation(generation.path, db, barrier=teardown_barrier)
    return True


def close_all() -> int:
    """Close every shared SessionDB regardless of refcount; returns the count. For gateway
    shutdown, after all agents and cron jobs finished. Idempotent."""
    teardown_barriers: Dict[Path, _TeardownBarrier] = {}
    with _lock:
        active_teardowns = list(_tearing_down.values())
        generations = list(_generations.values()) + list(_retired.values())
        for path in {generation.path for generation in generations}:
            teardown_barriers[path] = _admit_teardown_locked(path)
        _generations.clear()
        _retired.clear()
        for generation in generations:
            generation.retired = True
    # Teardown outside the lock, one path at a time. Holding the lifecycle
    # mutex across all generations for a path prevents an old retired handle
    # and the current handle from checkpointing the same sidecars concurrently.
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
    # A final release that removed its generation before this sweep took _lock still owns
    # its physical close; wait for it rather than return over a running teardown.
    for barrier in active_teardowns:
        barrier.event.wait()
    return len(generations)


def live_shared_session_dbs() -> List["SessionDB"]:
    """Snapshot of every live (non-retired) shared SessionDB (refcounts untouched), for
    in-process maintenance. A concurrent final release may close an instance, in which
    case the callee sees ``_conn is None``."""
    with _lock:
        return [g.db for g in _generations.values() if not g.retired]


@contextlib.contextmanager
def borrow_live_shared_session_dbs() -> Iterator[List["SessionDB"]]:
    """Borrow live handles with registry references pinned for the whole block.

    Maintenance must not operate on the unowned snapshot returned by
    :func:`live_shared_session_dbs`: the last real owner could otherwise
    release and physically close the connection between the snapshot and the
    maintenance call. Each borrowed generation gets one temporary reference;
    the ``finally`` block releases it even when maintenance raises.
    """
    with _lock:
        borrowed_generations = [
            generation for generation in _generations.values() if not generation.retired
        ]
        borrowed = [generation.db for generation in borrowed_generations]
        for generation in borrowed_generations:
            generation.refcount += 1
    try:
        yield borrowed
    finally:
        for db in reversed(borrowed):
            release(db)


def stats() -> Dict[str, int]:
    """Registry census for tests and diagnostics (no locks held long)."""
    with _lock:
        return {
            "live_generations": len(_generations), "retired_generations": len(_retired),
            "total_refcounts": sum(g.refcount for g in _generations.values()),
        }


def release_or_close(db: "SessionDB") -> None:
    """Release a shared instance, or close it when it is not registry-managed. Drop-in for a
    plain ``db.close()``: read-only opens, CLI one-shots and test fakes fall back."""
    if not release(db):
        try:
            db.close()
        except Exception:
            logger.debug("release_or_close fallback close failed", exc_info=True)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def close_shared_session_dbs() -> int:
    return close_all()

def get_shared_session_db(db_path: Optional[Path] = None) -> "SessionDB":
    return acquire(db_path)

def release_shared_session_db(db: "SessionDB") -> bool:
    return release(db)
# ---- END PLUGIN-COMPAT ----
