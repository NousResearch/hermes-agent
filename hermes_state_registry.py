"""Process-wide shared SessionDB registry (#90837).

A gateway process opens state.db from many call sites — the runner's
``AsyncSessionDB``, the ``SessionStore`` per-path cache, per-agent lazy
recall (``run_agent._get_session_db_for_recall``), per-job cron opens,
and per-message opens in mirror / channel_directory / slash_commands /
shutdown_flush / session_search / react_to_message.  Each bare
``SessionDB()`` mints its own writer connection, ``self._lock``,
close-time WAL checkpoint, and async token-writer thread.  With N
independent writer connections on one WAL file, mutual exclusion relies
only on SQLite's WAL write lock plus each instance's busy_timeout retry
ladder — and one connection's close-time checkpoint can race another's
growth, producing the lost/reordered-page-write signature reported
across 11+ incidents (#90837).

This module owns that boundary: one shared ``SessionDB`` per resolved
path per process, refcounted, with generation-aware retirement when the
underlying file is replaced (snapshot restore, recovery swap).

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
  the last one to settle, so neither ``acquire`` nor ``close_all`` / ``close_all_under``
  can escape while any handle for that path is still inside checkpoint/WAL-unlink.
- Maintenance callers borrow handles with a temporary registry reference instead of
  iterating an unpinned snapshot.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typed only
    from hermes_state import SessionDB

logger = logging.getLogger(__name__)


def _stat_db_file_identity(path: Path) -> Optional[Tuple[int, int]]:
    """Return ``(st_dev, st_ino)`` for *path*, or None when unavailable.

    Mirrors the hermes_state helper of the same name; kept local so this
    module has no import-time dependency on hermes_state (which imports
    this module — the cycle is resolved by deferring SessionDB lookup
    to call time).
    """
    import os

    try:
        st = os.stat(path)
    except OSError:
        return None
    # Windows volumes (and some network FS) report st_ino=0; a (0, 0)
    # identity would false-positive every check. Skip the inode half of
    # the guard there.
    if not st.st_dev or not st.st_ino:
        return None
    return (st.st_dev, st.st_ino)


class _Generation:
    """One shared SessionDB generation: instance, refcount, file identity."""

    __slots__ = ("db", "refcount", "identity", "retired")

    def __init__(self, db: "SessionDB", identity: Optional[Tuple[int, int]]) -> None:
        self.db = db
        self.refcount = 1
        self.identity = identity
        self.retired = False


_lock = threading.Lock()
# path → live generation (never retired).  A retired generation leaves
# this table immediately on retirement and lives on in _retired until
# its last holder releases.
_generations: Dict[Path, _Generation] = {}
# Object-keyed retired generations still draining holders.
_retired: Dict[int, _Generation] = {}  # id(db) → generation
# Paths whose next generation is currently being constructed.  Construction
# stays outside _lock because schema reconciliation can take seconds, but peers
# for the SAME file must wait: otherwise every cold caller opens a writable
# SQLite connection before the registry chooses one winner.
_opening: Dict[Path, threading.Event] = {}


def _open_session_db(path: Path) -> "SessionDB":
    """Construct the SessionDB for *path* (call-time import avoids cycles)."""
    from hermes_state import SessionDB

    return SessionDB(db_path=path)


def _teardown(db: "SessionDB") -> None:
    """Close a shared instance, clearing its registry-owned flag first."""
    try:
        db._shared_registry_owned = False
    _close_quietly(db, "Error closing shared SessionDB")


def _close_quietly(db: "SessionDB", debug_message: str) -> None:
    """close() that never propagates. A lost WAL generation whose capture failed is data at risk,
    not teardown noise: the handle stays open and the operator has to act, so that one surfaces."""
    try:
        db.close()
    except Exception as exc:
        from hermes_state_dbfile import RetiredGenerationCaptureError
        if isinstance(exc, RetiredGenerationCaptureError):
            logger.error("SessionDB for %s did not settle at close: %s", _db_path_of(db), exc)
        else:
            logger.debug(debug_message, exc_info=True)


def acquire(db_path: Optional[Path] = None) -> "SessionDB":
    """Return the shared SessionDB for *db_path*, incrementing its refcount.

    The same resolved path always returns the same ``SessionDB`` instance
    within one process, so all long-lived in-process callers share one
    writer connection, one ``self._lock``, and one token-writer thread.

    If the underlying file was replaced (different inode) since the
    shared generation was opened — e.g. by ``hermes sessions recover`` or
    a snapshot restore — the current generation is RETIRED (never lent
    again) but stays alive for its existing holders, and a fresh
    generation is opened in its place.

    Raises whatever ``SessionDB.__init__`` raises (malformed, locked,
    etc.).  On a replacement-open failure the registry holds NO entry for
    the path, so the next acquire retries fresh rather than handing out
    a closed stale object.
    """
    from hermes_state import _default_db_path

    raw_path = Path(db_path) if db_path is not None else Path(_default_db_path())
    try:
        path = raw_path.resolve()
    except OSError:
        path = raw_path

    while True:
        with _lock:
            generation = _generations.get(path)
            if generation is not None:
                current = _stat_db_file_identity(path)
                if (
                    current is not None
                    and generation.identity is not None
                    and current != generation.identity
                ):
                    # File replaced: retire the live generation (its
                    # holders keep it until they release) and elect one
                    # caller to construct the replacement below.
                    _retire_generation_locked(path, generation)
                else:
                    generation.refcount += 1
                    return generation.db

            opening = _opening.get(path)
            if opening is None:
                opening = threading.Event()
                _opening[path] = opening
                break

        # Another caller is constructing this path.  Do not hold the global
        # registry lock while waiting: unrelated databases continue opening.
        # A failed opener signals too, so one waiter can retry as the successor.
        opening.wait()

    # Open a fresh generation OUTSIDE the lock.  The per-path opening marker
    # prevents redundant writer connections without serialising other files.
    try:
        db = _open_session_db(path)
        db._shared_registry_owned = True
        identity = _stat_db_file_identity(path)
    except BaseException:
        with _lock:
            if _opening.get(path) is opening:
                _opening.pop(path, None)
            opening.set()
        raise

    with _lock:
        existing = _generations.get(path)
        if existing is not None:
            # Defensive: a generation may have been installed by explicit
            # registry manipulation while this open was in flight.
            existing.refcount += 1
            winner = existing.db
        else:
            _generations[path] = _Generation(db, identity)
            winner = db
        if _opening.get(path) is opening:
            _opening.pop(path, None)
        opening.set()
    if winner is not db:
        _teardown(db)
    return winner


def _retire_generation_locked(path: Path, generation: _Generation) -> None:
    """Retire *generation* so it is never lent again (caller holds _lock).

    The instance stays alive — its holders still own references — and is
    tracked in ``_retired`` keyed by ``id(db)`` so their releases find
    the right generation even after the path maps to a new one.
    """
    generation.retired = True
    if _generations.get(path) is generation:
        del _generations[path]
    _retired[id(generation.db)] = generation


def release(db: "SessionDB") -> bool:
    """Decrement the refcount of a shared SessionDB.

    Returns ``True`` if *db* was a shared instance and its refcount was
    decremented; ``False`` if *db* is not registry-managed (caller owns
    its own close()).  The final release of a generation tears it down —
    OUTSIDE the registry lock, so a close-time WAL checkpoint never
    stalls acquisition for every state.db in the process.

    Object-keyed lookup means an inode replacement cannot strand a
    still-owned generation: holders of the old generation release into
    the retired record, not into whatever the path currently names.
    """
    if db is None:
        return False
    key = id(db)
    with _lock:
        generation = _retired.get(key)
        if generation is None:
            path = getattr(db, "db_path", None)
            if path is None:
                return False
            try:
                path = Path(path)
            except (TypeError, ValueError):
                return False
            generation = _generations.get(path)
            if generation is None or generation.db is not db:
                # Not a shared instance (caller used SessionDB()
                # directly) — nothing to do; the caller owns close().
                return False
        generation.refcount -= 1
        needs_teardown = generation.refcount <= 0
        if needs_teardown:
            if generation.retired:
                _retired.pop(key, None)
            else:
                path = getattr(db, "db_path", None)
                if path is not None:
                    try:
                        _generations.pop(Path(path), None)
                    except (TypeError, ValueError):
                        pass
    # Teardown OUTSIDE the lock: it stops the token writer, checkpoints
    # the WAL, and drains the read pool — none of which may hold up
    # acquisition for every other state.db in the process.
    if needs_teardown:
        _teardown(db)
    return True


def _path_is_under(path: Path, root: Path) -> bool:
    """True when *path* is *root* or a file inside it (normcase, resolved)."""
    try:
        path_key = os.path.normcase(str(path.resolve()))
        root_key = os.path.normcase(str(root.resolve()))
    except OSError:
        path_key = os.path.normcase(str(path))
        root_key = os.path.normcase(str(root))
    return path_key == root_key or path_key.startswith(root_key + os.sep)


def _teardown_swept_generations(
    generations: List[_Generation],
    teardown_barriers: Dict[Path, _TeardownBarrier],
    active_teardowns: List[_TeardownBarrier],
) -> int:
    """Close *generations* outside the registry lock; wait for already-admitted teardowns."""
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


def close_all() -> int:
    """Close every shared SessionDB in this process, regardless of refcount.

    Called at gateway shutdown (after all agents and cron jobs have
    finished) to release every WAL write lock and drain every
    token-writer thread cleanly.  Returns the number of instances
    closed.  Idempotent.
    """
    closed = 0
    with _lock:
        generations = list(_generations.values()) + list(_retired.values())
        _generations.clear()
        _retired.clear()
        for generation in generations:
            generation.retired = True
    return _teardown_swept_generations(generations, teardown_barriers, active_teardowns)


def close_all_under(directory: str | Path) -> int:
    """Force-close every shared SessionDB whose file lives under *directory*; returns the count.

    Profile delete rmtree (and a same-name recreate) fails while this process still holds
    ``state.db``. Same contract as ``MemoryStore.release_all_under``: a live holder is
    expected to fail afterward; a process that holds none is a no-op returning 0.

    A final ``release()`` can drop the generation and admit teardown before the physical
    close finishes. Wait for those directory-matching barriers even when no generation
    remains, otherwise rmtree still sees the open handle.
    """
    try:
        root = Path(directory).expanduser().resolve()
    except OSError:
        root = Path(directory).expanduser()
    teardown_barriers: Dict[Path, _TeardownBarrier] = {}
    with _lock:
        generations = [
            generation
            for generation in list(_generations.values()) + list(_retired.values())
            if _path_is_under(generation.path, root)
        ]
        # Collect by directory, not by remaining generations: a last release already
        # popped the generation and left only ``_tearing_down``.
        active_teardowns = [
            barrier for path, barrier in _tearing_down.items()
            if _path_is_under(path, root)
        ]
        selected_paths = {generation.path for generation in generations}
        for path in selected_paths:
            teardown_barriers[path] = _admit_teardown_locked(path)
        for generation in generations:
            generation.retired = True
            if _generations.get(generation.path) is generation:
                _generations.pop(generation.path, None)
            _retired.pop(id(generation.db), None)
    return _teardown_swept_generations(generations, teardown_barriers, active_teardowns)


def live_shared_session_dbs() -> List["SessionDB"]:
    """Snapshot of every live (non-retired) shared SessionDB in this process.

    For periodic in-process maintenance (the gateway housekeeping tick's
    deferred-FTS retry). Refcounts are NOT touched: the caller only invokes
    a method on an instance that some holder already keeps alive; a
    concurrent final release closes it and the callee sees ``_conn is None``.
    """
    with _lock:
        return [g.db for g in _generations.values() if not g.retired]


def stats() -> Dict[str, int]:
    """Registry census for tests and diagnostics (no locks held long)."""
    with _lock:
        live = len(_generations)
        retired = len(_retired)
        refs = sum(g.refcount for g in _generations.values())
        return {
            "live_generations": live,
            "retired_generations": retired,
            "total_refcounts": refs,
        }


# ── Backwards-compatible aliases (hermes_state re-exports) ──
# Kept so call sites and tests can import either from hermes_state
# (the historical path) or from this module directly.

def get_shared_session_db(db_path: Optional[Path] = None) -> "SessionDB":
    return acquire(db_path)


def release_shared_session_db(db: "SessionDB") -> bool:
    return release(db)


def close_shared_session_dbs() -> int:
    return close_all()


def release_or_close(db: "SessionDB") -> None:
    """Release a shared instance, or close it when it is not registry-managed.

    The one-line cleanup for call sites that previously did a plain
    ``db.close()``: shared instances return their refcount to the
    registry (the registry owns the lifecycle), anything else — read-only
    opens, CLI one-shots, test fakes — falls back to a direct close.
    """
    if not release(db):
        _close_quietly(db, "release_or_close fallback close failed")


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
