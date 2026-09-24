"""Profile-level session-storage health: one process-wide latch per state.db path.

A structurally corrupt state.db used to show up as an empty or partial Desktop sidebar,
green readiness and a 500 from ``/api/sessions``: every surface guessed on its own and
none said "the store is damaged" (#72046). This module is the single place that fact is
recorded. Writers (``SessionDB._halt_db_corrupt``), readers (``SessionDB`` read helpers)
and the readiness probe publish into it; ``gateway.readiness`` (``state_db`` /
``session_store`` checks, hence ``/api/status`` ``components.storage``) and the session
list endpoints read from it, so Desktop and readiness cannot disagree.

Only structural corruption latches: bare ``SQLITE_CORRUPT`` / ``SQLITE_NOTADB`` with no
FTS provenance. FTS-scoped damage has its own fail-open path with canonical rows intact,
and a malformed-schema row is healed by the web open path, so neither is reported here.

The latch never clears on its own. A corrupt image does not heal, and a store that
flickers between "ok" and "corrupt" is the silent failure this replaces. It resets when
the process restarts, which is the recovery boundary ``StateDbCorruptError`` already
documents (stop Hermes, recover or restore, start again). No marker is written to disk:
the file it would describe is the one that is damaged.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional

from hermes_state_errors import (
    classify_persistence_error,
    is_fts_scoped_corruption_error,
    is_malformed_schema_error,
)

logger = logging.getLogger(__name__)

STORAGE_OK = "ok"
STORAGE_CORRUPT = "corrupt"

_lock = threading.Lock()
_corrupt: Dict[str, str] = {}  # resolved db path -> first error text (log only, never served)


def _key(db_path) -> str:
    return str(Path(db_path).expanduser().resolve(strict=False))


def is_structural_corruption_error(exc: BaseException) -> bool:
    """Canonical B-tree/schema/freelist damage: a corrupt/NOTADB error SQLite does not scope to
    the FTS index, and not the malformed-schema case the web open path repairs."""
    return (
        isinstance(exc, sqlite3.DatabaseError)
        and not is_fts_scoped_corruption_error(exc)
        and not is_malformed_schema_error(exc)
        and classify_persistence_error(exc) == "corrupt"
    )


def mark_storage_corrupt(db_path, reason: object) -> None:
    """Latch *db_path* as corrupt for the life of this process (idempotent)."""
    key = _key(db_path)
    with _lock:
        if key in _corrupt:
            return
        _corrupt[key] = str(reason)
    logger.error(
        "state.db at %s is structurally corrupt (%s); session storage is reported as corrupt "
        "until Hermes restarts on a recovered or restored file. Stop Hermes, then run "
        "`hermes sessions recover --source %s --inspect-only` or restore a snapshot.",
        db_path, reason, db_path,
    )


def note_storage_error(db_path, exc: BaseException) -> bool:
    """Latch *db_path* when *exc* is structural corruption; True when it was."""
    if not is_structural_corruption_error(exc):
        return False
    mark_storage_corrupt(db_path, exc)
    return True


def storage_state(db_path) -> str:
    """``"corrupt"`` once this process has seen structural corruption on *db_path*, else ``"ok"``."""
    with _lock:
        return STORAGE_CORRUPT if _key(db_path) in _corrupt else STORAGE_OK


def storage_corrupt_reason(db_path) -> Optional[str]:
    """The first error text latched for *db_path* (logs/diagnostics only), or None."""
    with _lock:
        return _corrupt.get(_key(db_path))


def reset_storage_state(db_path=None) -> None:
    """Forget the latch for *db_path* (all paths when None). For tests and a verified in-process
    recovery; nothing in the runtime clears it on its own."""
    with _lock:
        if db_path is None:
            _corrupt.clear()
        else:
            _corrupt.pop(_key(db_path), None)


__all__ = [
    "STORAGE_CORRUPT",
    "STORAGE_OK",
    "is_structural_corruption_error",
    "mark_storage_corrupt",
    "note_storage_error",
    "reset_storage_state",
    "storage_corrupt_reason",
    "storage_state",
]
