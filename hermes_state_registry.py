"""Process-wide shared SessionDB registry.

Each bare ``SessionDB()`` mints its own writer connection, lock, close-time WAL checkpoint
and token writer thread, and one connection's close-time checkpoint can race another's
growth (lost/reordered-page-write corruption). This module owns that boundary: one shared
``SessionDB`` per resolved path per process, refcounted, with generation-aware retirement
when the file is replaced (snapshot restore, recovery swap).

Lifecycle rules:
- ``acquire(path)`` returns the current generation for *path* and bumps its refcount.
- ``close()`` on a shared instance is a NO-OP: the registry owns the connection lifecycle.
- ``release(db)`` decrements the generation *db was acquired from* (object-keyed, so an
  inode replacement cannot strand a still-owned generation); the final release of a
  retired generation tears it down.
- On inode change the old generation is RETIRED (never lent again) but stays alive until
  its holders release. If the replacement open fails the registry keeps NO path entry.
- All teardown happens OUTSIDE the registry lock: a final release's WAL checkpoint must
  never stall acquisition for every state.db.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from hermes_state_common import stat_db_file_identity as _stat_db_file_identity

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typed only
    from hermes_state import SessionDB

logger = logging.getLogger(__name__)


class _Generation:
    """One shared SessionDB generation: instance, refcount, file identity."""

    __slots__ = ("db", "refcount", "identity", "retired")

    def __init__(self, db: "SessionDB", identity: Optional[Tuple[int, int]]) -> None:
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
            opening = _opening.get(path)
            if opening is None:
                opening = _opening[path] = threading.Event()
                break
        # Another caller is constructing this path; wait without holding the global
        # lock. A failed opener signals too, so a waiter can retry.
        opening.wait()

    # Open OUTSIDE the lock; the per-path marker prevents redundant writers without
    # serialising other files.
    try:
        db = _open_session_db(path)
        db._shared_registry_owned = True
        identity = _stat_db_file_identity(path)
    except BaseException:
        with _lock:
            _finish_opening(path, opening)
        raise

    with _lock:
        existing = _generations.get(path)
        if existing is not None:  # Defensive: installed by explicit registry manipulation mid-open.
            existing.refcount += 1
            winner = existing.db
        else:
            _generations[path] = _Generation(db, identity)
            winner = db
        _finish_opening(path, opening)
    if winner is not db:
        _teardown(db)
    return winner


def release(db: "SessionDB") -> bool:
    """Decrement the refcount of a shared SessionDB. ``True`` if *db* was shared; ``False``
    if it is not registry-managed (caller owns close()). The final release tears the
    generation down OUTSIDE the registry lock. Lookup is object-keyed, so holders of an
    old generation release into its retired record, not into whatever the path names."""
    if db is None:
        return False
    key = id(db)
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
        if needs_teardown and generation.retired:
            _retired.pop(key, None)
        elif needs_teardown and (path := _db_path_of(db)) is not None:
            _generations.pop(path, None)
    # Teardown OUTSIDE the lock: stopping the token writer, WAL checkpoint and read-pool
    # drain must not block acquisition for every other state.db.
    if needs_teardown:
        _teardown(db)
    return True


def close_all() -> int:
    """Close every shared SessionDB regardless of refcount; returns the count. For gateway
    shutdown, after all agents and cron jobs finished. Idempotent."""
    with _lock:
        generations = list(_generations.values()) + list(_retired.values())
        _generations.clear()
        _retired.clear()
        for generation in generations:
            generation.retired = True
    for generation in generations:
        _teardown(generation.db)
    return len(generations)


def live_shared_session_dbs() -> List["SessionDB"]:
    """Snapshot of every live (non-retired) shared SessionDB (refcounts untouched), for
    in-process maintenance. A concurrent final release may close an instance, in which
    case the callee sees ``_conn is None``."""
    with _lock:
        return [g.db for g in _generations.values() if not g.retired]


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
