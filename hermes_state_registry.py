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

- ``acquire(path)`` returns the current generation for *path*,
  incrementing its refcount.  Same path ⇒ same instance ⇒ same writer
  connection.
- ``close()`` on a shared instance is a NO-OP.  The registry — not any
  individual caller — owns the connection lifecycle, so one caller's
  ``close()`` can never tear down a writer other callers still hold.
- ``release(db)`` decrements the generation *db was acquired from*
  (object-keyed, not pathname-keyed, so an inode replacement cannot
  strand a still-owned generation).  The final release of a retired
  generation tears it down.
- On inode change, the old generation is RETIRED — never lent again —
  but stays alive until its existing holders release.  If a replacement
  open fails, the registry is left WITHOUT a path entry (never a closed
  stale object), so the next acquire retries fresh.
- All teardown happens OUTSIDE the registry lock: a final release's
  WAL checkpoint must never stall acquisition for every state.db.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

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


def _open_session_db(path: Path) -> "SessionDB":
    """Construct the SessionDB for *path* (call-time import avoids cycles)."""
    from hermes_state import SessionDB

    return SessionDB(db_path=path)


def _teardown(db: "SessionDB") -> None:
    """Close a shared instance, clearing its registry-owned flag first."""
    try:
        db._shared_registry_owned = False
    except Exception:
        pass
    try:
        db.close()
    except Exception:
        logger.debug("Error closing shared SessionDB", exc_info=True)


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

    path = Path(db_path) if db_path is not None else Path(_default_db_path())

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
                    # holders keep it until they release) and fall
                    # through to opening a fresh one below.
                    _retire_generation_locked(path, generation)
                    generation = None
                else:
                    generation.refcount += 1
                    return generation.db

        # Open a fresh generation OUTSIDE the lock: construction can
        # take seconds (write-lock patience) and must not block every
        # other state.db acquisition in the process.
        db = _open_session_db(path)
        db._shared_registry_owned = True
        identity = _stat_db_file_identity(path)
        with _lock:
            existing = _generations.get(path)
            if existing is not None and existing is not generation:
                # Someone else opened a generation while we were
                # constructing.  Ours loses — close it (outside the
                # lock) and use theirs.
                existing.refcount += 1
                winner = existing.db
            else:
                _generations[path] = _Generation(db, identity)
                winner = db
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
    # Teardown outside the lock, one generation at a time.
    for generation in generations:
        _teardown(generation.db)
        closed += 1
    return closed


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
        try:
            db.close()
        except Exception:
            logger.debug("release_or_close fallback close failed", exc_info=True)
