"""Error-classification predicates for storage failures.

A persistence contract error answers by its type: its ``cause`` (or its class, for contention and a full disk) decides. Anything else, a live ``sqlite3`` exception or a string that survived RPC wrapping, is classified by its SQLite result code and then its text.
"""

import errno
import re
import sqlite3

from persistence.core.errors import PERSISTENCE_ERROR_CAUSES, DiskFull, PersistenceError, StoreContention

__all__ = [
    "PERSISTENCE_ERROR_CAUSES",
    "classify_persistence_error",
    "is_disk_full_error",
    "is_fts_scoped_corruption_error",
    "is_lock_contention",
    "is_malformed_db_error",
    "is_sqlite_lock_error",
]

# Malformed schema: ``sqlite_master`` itself is inconsistent (typically a DUPLICATE
# ``CREATE VIRTUAL TABLE messages_fts`` row). SQLite parses the whole schema while
# preparing the FIRST statement, so EVERY statement raises (even ``PRAGMA
# journal_mode`` during __init__); only ``PRAGMA writable_schema=ON`` +
# sqlite_master surgery still work. Canonical rows are intact; recovery rebuilds
# only the FTS layer.
_MALFORMED_SCHEMA_MARKERS = ("malformed database schema",)
_MALFORMED_DB_MARKERS = (*_MALFORMED_SCHEMA_MARKERS, "database disk image is malformed")


def is_malformed_db_error(exc: BaseException) -> bool:
    """Malformed-schema OR generic corrupt-image error. Diagnostics / offline
    recovery only — runtime repair must use :func:`is_malformed_schema_error`."""
    return isinstance(exc, sqlite3.DatabaseError) and any(
        marker in str(exc).lower() for marker in _MALFORMED_DB_MARKERS
    )


# Lock contention by result code. SQLite keeps SQLITE_BUSY when FTS5's xConnect loses the race
# on its %_config read but replaces the text with "vtable constructor failed: messages_fts",
# so a phrase match read a busy store as a hard failure.
_SQLITE_LOCK_CODES = (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)


def _sqlite_primary_code(exc_or_str) -> int | None:
    """Primary result code (extended codes keep it in the low byte); None when unknown."""
    code = getattr(exc_or_str, "sqlite_errorcode", None)
    return code & 0xFF if isinstance(code, int) else None


def is_sqlite_lock_error(exc_or_str) -> bool:
    """SQLITE_BUSY / SQLITE_LOCKED: wait and retry, never treat as damage. A known result code
    decides; only without one (our own re-raised messages, RPC-wrapped strings) does the text."""
    code = _sqlite_primary_code(exc_or_str)
    if code is not None:
        return code in _SQLITE_LOCK_CODES
    text = str(exc_or_str).lower()
    return "locked" in text or "busy" in text


def _typed_cause(exc_or_str) -> str | None:
    """The ``cause`` a contract error declares, or None when it declares none (``unknown``) or is not one."""
    if isinstance(exc_or_str, PersistenceError) and exc_or_str.cause != "unknown":
        return exc_or_str.cause
    return None


def is_lock_contention(exc_or_str) -> bool:
    """Busy right now: wait and retry. A contract error answers by type (``StoreContention``); anything else by
    :func:`is_sqlite_lock_error`."""
    if isinstance(exc_or_str, StoreContention):
        return True
    if _typed_cause(exc_or_str) is not None:
        return False
    return is_sqlite_lock_error(exc_or_str)


# "Filesystem cannot accept another write" substrings (OSError, sqlite3, wrapped RPC strings).
_DISK_FULL_MARKERS = (
    "no space left on device", "not enough space", "database or disk is full",  # SQLITE_FULL
    "disk full", "full disk", "enospc",
)


def is_disk_full_error(exc: BaseException | str | None) -> bool:
    """Disk-full / ENOSPC: the contract's DiskFull, OSError(ENOSPC), SQLITE_FULL, or matching strings."""
    if exc is None:
        return False
    if isinstance(exc, DiskFull):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return True
    lowered = (exc if isinstance(exc, str) else str(exc)).lower()
    return any(marker in lowered for marker in _DISK_FULL_MARKERS)


# "Database FILE structurally damaged" substrings. "database disk image is
# malformed" contains "disk", so this check MUST run before the disk bucket in
# classify_persistence_error or B-tree corruption reads as "free some disk space".
# Kept as plain substrings so sqlite3.DatabaseError, wrapped RPC strings, and logged message text all match
# the same helper. See #77386.
_DB_CORRUPTION_MARKERS = (
    "malformed", "file is not a database", "not a database", "database corruption",
)

# The module constant exists on Python 3.11+; the numeric value is stable across SQLite releases.
SQLITE_CORRUPT_VTAB = getattr(sqlite3, "SQLITE_CORRUPT_VTAB", 267)

# Every FTS object hangs off this prefix: the virtual tables and their _data/_idx/_content/
# _docsize/_config shadow b-trees. FTS5 names the table in its own corruption reports.
_FTS_OBJECT_RE = re.compile(r"\bmessages_fts\w*")


def is_fts_scoped_corruption_error(exc_or_str) -> bool:
    """Corruption SQLite itself attributes to the FTS index layer.

    A known result code outranks prose: ``SQLITE_CORRUPT_VTAB`` is FTS-scoped even with
    the generic malformed-image text older SQLite builds emit, while bare ``SQLITE_CORRUPT``
    / ``SQLITE_NOTADB`` carry no object scope and any other known code contradicts
    FTS-looking prose, so both fail closed. Only without a code (Python < 3.11, RPC-wrapped
    strings) does the text decide, and then only an ``fts5:`` corruption report or a
    corruption marker that names a ``messages_fts*`` object counts.
    """
    if exc_or_str is None:
        return False
    code = getattr(exc_or_str, "sqlite_errorcode", None)
    if code is not None:
        return code == SQLITE_CORRUPT_VTAB
    text = (exc_or_str if isinstance(exc_or_str, str) else str(exc_or_str)).lower()
    if not _FTS_OBJECT_RE.search(text):
        return False
    if text.startswith("fts5:") and "corrupt" in text:
        return True
    return any(marker in text for marker in _DB_CORRUPTION_MARKERS)


_PERSISTENCE_CAUSE_BY_PHRASE = (
    (("turn lease",), "turn_lease"),
    (("closed by compression",), "compression_closed"),
    (("being compressed", "compression lease"), "compression"),
    # RPC-wrapped errors lose their exception type; retain the same sidecar/main-file split.
    (("deleted state.db-wal", "deleted state.db-shm"), "deleted_wal"),
    (("was replaced underneath",), "replaced"),
    (_DB_CORRUPTION_MARKERS, "corrupt"),
    (("locked", "busy"), "locked"),
    # A flush rejected by the session-row FK: the row was removed under a live agent (#123583).
    (("foreign key constraint failed",), "session_row_missing"),
)


def classify_persistence_error(exc_or_str) -> str:
    """Coarse cause bucket (PERSISTENCE_ERROR_CAUSES) so the user's guidance
    matches: "locked" = busy, retry; "disk" = full/read-only/permissions;
    "compression" = a live lease refused the write; "compression_closed" = adopt
    the rotated session id; "turn_lease" = fencing, not storage; "corrupt" =
    file damage (repair path, not disk space); "fts_index" = SQLite scoped the
    corruption to the FTS index (the transcript store is not damaged); "replaced" =
    main-file replacement; "deleted_wal" = a retired sidecar generation requiring
    capture inspection; "session_row_missing" = the session row was deleted under a
    live agent (FK rejection; the flush recreates it).

    A contract error that declares its cause answers with it; any other error or string is
    classified by result code, then by phrase. Corruption comes BEFORE the lock/disk buckets
    ("disk image is malformed" contains "disk")."""
    if exc_or_str is None:
        return "unknown"
    cause = _typed_cause(exc_or_str)
    if cause is not None:
        return cause
    # Provenance before prose: an FTS-scoped result code (or, without one, an fts5 report
    # naming messages_fts*) is index damage, never whole-file corruption (#97794).
    if is_fts_scoped_corruption_error(exc_or_str):
        return "fts_index"
    if _sqlite_primary_code(exc_or_str) in _SQLITE_LOCK_CODES:
        return "locked"
    if getattr(exc_or_str, "sqlite_errorcode", None) == sqlite3.SQLITE_CONSTRAINT_FOREIGNKEY:
        return "session_row_missing"
    text = str(exc_or_str).lower()
    for markers, cause in _PERSISTENCE_CAUSE_BY_PHRASE:
        if any(marker in text for marker in markers):
            return cause
    if is_disk_full_error(exc_or_str) or any(m in text for m in ("disk", "readonly", "read-only")):
        return "disk"
    return "unknown"
