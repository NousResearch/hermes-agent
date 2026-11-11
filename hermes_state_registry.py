"""Process-wide shared SessionDB registry.

Long-lived in-process callers share one SessionDB per resolved path, and that SessionDB serves through the one
store holder ``persistence.sqlite._registry`` lends for the file to every caller in the process. That registry owns
the physical lifecycle: one writer per path, refcounts, generation-aware retirement when the file is replaced
(snapshot restore, recovery swap), the teardown barrier, and the forced closes. This module lends one SessionDB per
holder generation.

Lifecycle rules:
- ``acquire(path)`` returns the SessionDB for the path's current generation and takes one reference on it.
- ``close()`` on a shared instance RELEASES one reference instead of tearing the connection down: the registry
  closes only on the final release, so legacy call sites return their reference instead of leaking it.
- ``release(db)`` returns one reference to the generation *db* serves through; the final release of a retired
  generation tears it down.
- Maintenance callers borrow handles with a temporary registry reference instead of iterating an unpinned snapshot.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from persistence.sqlite import _registry as _holders

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typed only
    from hermes_state import SessionDB

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# id(holder) -> the SessionDB lent for that holder generation. An entry keeps its holder alive, so the id is not
# reused while it is here; entries whose generation was torn down are dropped at the next lend.
_lent: dict[int, SessionDB] = {}


def _lend(holder) -> SessionDB:
    """The one SessionDB serving through the registry-owned *holder* (call-time import avoids cycles)."""
    from hermes_state import SessionDB

    with _lock:
        for key in [key for key, db in _lent.items() if not db._holder._shared_registry_owned]:
            del _lent[key]
        db = _lent.get(id(holder))
        if db is None:
            db = _lent[id(holder)] = SessionDB(holder=holder)
        return db


def _close_quietly(db: SessionDB, debug_message: str) -> None:
    """close() that never propagates. A lost WAL generation whose capture failed is data at risk,
    not teardown noise: the handle stays open and the operator has to act, so that one surfaces."""
    try:
        db.close()
    except Exception as exc:
        from hermes_state_dbfile import RetiredGenerationCaptureError
        if isinstance(exc, RetiredGenerationCaptureError):
            logger.error("SessionDB for %s did not settle at close: %s", getattr(db, "db_path", None), exc)
        else:
            logger.debug(debug_message, exc_info=True)


def acquire(db_path: Optional[Path] = None) -> SessionDB:
    """Return the shared SessionDB for *db_path*, taking one reference on its generation. If the file was
    replaced (different inode) since the generation opened, that generation is RETIRED but stays alive for its
    holders, and a fresh one is opened in its place. Raises whatever opening the store holder raises; on a
    replacement-open failure the registry holds NO entry for the path."""
    from hermes_state import _default_db_path

    return _lend(_holders.acquire(Path(db_path) if db_path is not None else Path(_default_db_path())))


def release(db: SessionDB) -> bool:
    """Return one reference of a shared SessionDB. ``True`` if *db* was shared; ``False`` if it is not
    registry-managed (caller owns close()). Lookup is object-keyed: only the SessionDB the registry lent counts,
    so a proxy over it falls back to its own close(). The final release tears the generation down OUTSIDE the
    registry lock; holders of an old generation release into its retired record, not into whatever the path
    names."""
    holder = getattr(db, "_holder", None)
    with _lock:
        lent = holder is not None and _lent.get(id(holder)) is db
    return lent and _holders.release(holder)


def close_all() -> int:
    """Close every shared generation regardless of refcount; returns the count. For gateway
    shutdown, after all agents and cron jobs finished. Idempotent."""
    return _holders.close_all()


def close_all_under(directory: str | Path) -> int:
    """Force-close every shared generation whose file lives under *directory*; returns the count.

    Profile delete rmtree (and a same-name recreate) fails while this process still holds
    ``state.db``. Same contract as ``MemoryStore.release_all_under``: a live holder is
    expected to fail afterward; a process that holds none is a no-op returning 0.
    """
    return _holders.close_all_under(directory)


def other_generations_for_path(db_path: Path, *, exclude: Optional[SessionDB] = None) -> list[str]:
    """Describe every other LIVE generation THIS process holds for *db_path*, other than the one *exclude*
    serves through. RETIRED generations are not holders here: see
    ``persistence.sqlite._registry.other_generations_for_path``."""
    return _holders.other_generations_for_path(db_path, exclude=getattr(exclude, "_holder", None))


def live_shared_session_dbs() -> list[SessionDB]:
    """Snapshot of every live (non-retired) shared SessionDB (refcounts untouched), for
    in-process maintenance. A concurrent final release may close an instance, in which
    case the callee sees ``_conn is None``."""
    return [_lend(holder) for holder in _holders.live_shared_session_dbs()]


@contextlib.contextmanager
def borrow_live_shared_session_dbs() -> Iterator[list[SessionDB]]:
    """Borrow live handles with registry references pinned for the whole block.

    Maintenance must not operate on the unowned snapshot returned by
    :func:`live_shared_session_dbs`: the last real owner could otherwise
    release and physically close the connection between the snapshot and the
    maintenance call. Each borrowed generation gets one temporary reference,
    released even when maintenance raises.
    """
    with _holders.borrow_live_shared_session_dbs() as holders:
        yield [_lend(holder) for holder in holders]


def stats() -> dict[str, int]:
    """Registry census for tests and diagnostics (no locks held long)."""
    return _holders.stats()


def release_or_close(db: SessionDB) -> None:
    """Release a shared instance, or close it when it is not registry-managed. Drop-in for a
    plain ``db.close()``: read-only opens, CLI one-shots and test fakes fall back."""
    if not release(db):
        _close_quietly(db, "release_or_close fallback close failed")
