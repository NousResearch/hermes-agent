"""Exception types and error-classification predicates for the state store.
Shared by hermes_state and its mixins; string predicates match wrapped RPC
strings as well as live sqlite3 exceptions."""

import errno
import re
import sqlite3
from pathlib import Path

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


# SQLITE_IOERR as a substring (wrapped strings still classify).
_DISK_IO_ERROR_MARKER = "disk i/o error"

# "Store BUSY, not gone" — HTTP callers map these to 503 instead of 500. Corruption
# is deliberately absent: a malformed store must surface, not be retried into a timeout.
_TRANSIENT_SQLITE_MARKERS = (
    _DISK_IO_ERROR_MARKER, "database is locked", "database table is locked", "busy",
)


def _is_no_more_rows(exc: sqlite3.Error) -> bool:
    """Transient engine error on contended WAL appends (retries like locked/busy);
    message-scoped because some builds raise it as InterfaceError."""
    return "no more rows available" in str(exc).lower()


def is_transient_sqlite_error(exc: BaseException) -> bool:
    """"Busy right now", not "damaged": one predicate so retry and the HTTP
    503-vs-500 split cannot drift apart."""
    return isinstance(exc, sqlite3.OperationalError) and any(
        marker in str(exc).lower() for marker in _TRANSIENT_SQLITE_MARKERS
    )


def is_malformed_schema_error(exc: BaseException) -> bool:
    """Only SQLite's explicit malformed-schema text: a generic "disk image is
    malformed" may be any B-tree page, so runtime repair must fail closed on it."""
    return isinstance(exc, sqlite3.DatabaseError) and any(
        marker in str(exc).lower() for marker in _MALFORMED_SCHEMA_MARKERS
    )


# "Filesystem cannot accept another write" substrings (OSError, sqlite3, wrapped RPC strings).
_DISK_FULL_MARKERS = (
    "no space left on device", "not enough space", "database or disk is full",  # SQLITE_FULL
    "disk full", "full disk", "enospc",
)


def is_disk_full_error(exc: BaseException | str | None) -> bool:
    """Disk-full / ENOSPC: OSError(ENOSPC), SQLITE_FULL, or matching strings."""
    if exc is None:
        return False
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return True
    lowered = (exc if isinstance(exc, str) else str(exc)).lower()
    return any(marker in lowered for marker in _DISK_FULL_MARKERS)


# Every classify_persistence_error bucket; consumers enumerate this tuple.
PERSISTENCE_ERROR_CAUSES = (
    "locked", "compression", "compression_closed", "turn_lease", "corrupt", "corrupt_unconfirmed",
    "fts_index", "replaced", "disk", "unknown",
)


# "Database FILE structurally damaged" substrings. "database disk image is
# malformed" contains "disk", so this check MUST run before the disk bucket in
# classify_persistence_error or B-tree corruption reads as "free some disk space".
# Kept as plain substrings so sqlite3.DatabaseError, wrapped RPC strings, and logged message text all match
# the same helper. See #77386.
_DB_CORRUPTION_MARKERS = (
    "malformed", "file is not a database", "not a database", "database corruption",
)

# SQLite result codes: the module constants exist on Python 3.11+, the numeric values are
# stable across every SQLite release.
SQLITE_CORRUPT = getattr(sqlite3, "SQLITE_CORRUPT", 11)
SQLITE_NOTADB = getattr(sqlite3, "SQLITE_NOTADB", 26)
SQLITE_CORRUPT_VTAB = getattr(sqlite3, "SQLITE_CORRUPT_VTAB", 267)

# Every FTS object hangs off this prefix: the virtual tables and their _data/_idx/_content/
# _docsize/_config shadow b-trees. FTS5 names the table in its own corruption reports.
_FTS_OBJECT_RE = re.compile(r"\bmessages_fts\w*")


def is_fts_scoped_corruption_error(exc_or_str) -> bool:
    """Corruption SQLite itself attributes to the FTS index layer: the ONE provenance rule
    shared by the write-repair gate (``SessionDB._is_fts_write_corruption_error``), the
    gateway transcript retry and :func:`classify_persistence_error` (#96038, #97794).

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


def _is_unscoped_corruption_code(code) -> bool:
    """No code at all, or a corruption-class primary code that names no object."""
    return code is None or (int(code) & 0xFF) in (SQLITE_CORRUPT, SQLITE_NOTADB)


def verify_canonical_tables_healthy(db_path) -> bool:
    """Read-only ``PRAGMA quick_check`` on a fresh connection; True only when every reported
    problem names a ``messages_fts*`` object (or there is none).

    Answers "is this corruption report confined to the derived index layer?" for errors whose
    result code carries no scope (#97794: a healthy 22k-message store was declared structurally
    corrupt on the strength of an FTS-layer error). quick_check walks every b-tree page, the
    damage class behind a malformed-image error, and skips only the index-content cross-check
    that never raises on a write; on a multi-GB store that is seconds instead of minutes. Any
    probe failure (cannot open, locked, the check itself raising) returns False so callers keep
    the conservative verdict.
    """
    try:
        conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True, timeout=1.0)
    except sqlite3.Error:
        return False
    try:
        rows = conn.execute("PRAGMA quick_check").fetchall()
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    problems = [str(row[0]) for row in rows if row and str(row[0]).lower() != "ok"]
    return all(_FTS_OBJECT_RE.search(line.lower()) for line in problems)


class CompressionSessionClosedError(RuntimeError):
    """A durable write targeted a parent already closed by compression."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(
            f"Session {session_id!r} is closed by compression; "
            "adopt its live continuation before appending messages"
        )


class CompressionSessionBusyError(RuntimeError):
    """A non-owner tried to write while compression owns the session."""


class SessionCompressionInProgressError(CompressionSessionBusyError):
    """A concurrent writer collided with a *live* compression lock — transient
    (the compressor publishes in seconds; ``_execute_write`` waits), unlike the
    parent class's other case (a compressor whose own lease is gone: permanent,
    fail fast). Subclassing keeps every existing handler working."""


class SessionTurnLeaseLostError(RuntimeError):
    """A transcript write presented a turn-lease holder that no longer owns it.
    Fail-fast fencing (no ``_execute_write`` retry): a later writer may already
    be persisting a newer turn, and landing this one would interleave a stale reply."""


class StateDbReplacedError(RuntimeError):
    """The state.db path no longer names the file this SessionDB opened
    (out-of-band cp/mv/restore). In-place FTS repair and fail-open trigger
    dropping cannot fix a generation mismatch; they amplify it."""


class DeletedWalGenerationError(StateDbReplacedError):
    """A live process holds a deleted state.db-wal / -shm generation. Opening or
    writing through this handle would mint a second WAL inode (split-brain ->
    intermittent SQLITE_CORRUPT / IOERR). Stop the writers; never unlink the WAL
    yourself. Subclasses StateDbReplacedError so every consumer that diverts
    transcripts on a replaced store handles this identically."""


# SQLite header application_id (offset 68). Distinct from inode: ``cp`` onto the
# same path keeps st_ino and truncates+rewrites.
_STATE_DB_APPLICATION_ID_OFFSET = 68
_STATE_DB_GENERATION_KEY = "db_file_generation"
_STATE_DB_REPLACED_MSG = (
    "FATAL: state.db was replaced underneath the gateway; refusing further "
    "writes to this file. Divert transcripts to sessions/<id>.jsonl (and the "
    "gateway pending_messages spool) and restore or reopen after operator intervention."
)
_DELETED_WAL_GENERATION_MSG = (
    "FATAL: a live process holds a deleted state.db-wal or state.db-shm "
    "inode while the path names a different (or missing) generation. "
    "Refusing to open or write so a second WAL cannot be minted. "
    "Stop the gateway, dashboard, and cron writers that hold the deleted "
    "sidecar, then reopen. Do not delete the WAL yourself. "
    "database.journal_mode: delete is operator containment, not a new default."
)


class StateDbCorruptError(sqlite3.DatabaseError):
    """A live SessionDB observed structural (non-FTS, non-replaced) corruption and
    is quarantined: sticky for the handle's life — writes fail fast, no reopen,
    no close-time checkpoint (a handle that kept writing after the first error
    checkpointed 15 pages under wrong page numbers and turned a readable file
    into "file is not a database"; SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE on 3.12+
    also stops SQLite's own). Subclasses sqlite3.DatabaseError so every degrade
    path keeps working. Recovery boundary: restart on a repaired/restored file.

    Stopping the writes is what prevents that; skipping the explicit checkpoint is the second line of
    defence. SQLite still runs its own last-connection checkpoint inside ``close()`` (and deletes the
    ``-wal`` sidecar) unless ``SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE`` is set — Python exposes it via
    ``Connection.setconfig()`` on 3.12+, so quarantine disables the close-time checkpoint there and the WAL
    survives on disk for forensics; on 3.11 the internal checkpoint is unavoidable (post-quarantine it can
    only carry pre-corruption committed frames, since no further writes are accepted). See #90837.
    """


_STATE_DB_CORRUPT_MSG = (
    "FATAL: state.db reported structural corruption (database disk image is "
    "malformed outside the FTS shadow tables) on a live handle; refusing further "
    "writes, automatic reopen, and the close-time WAL checkpoint on this file. "
    "Stop the gateway, then run `hermes sessions recover --source <state.db> "
    "--inspect-only` or restore a snapshot. Unwritten transcripts are diverted to "
    "sessions/<id>.jsonl (and the gateway pending_messages spool)."
)


_PERSISTENCE_CAUSE_BY_TYPE = (
    (SessionTurnLeaseLostError, "turn_lease"),
    (CompressionSessionClosedError, "compression_closed"),
    (CompressionSessionBusyError, "compression"),
    (StateDbReplacedError, "replaced"),
    (StateDbCorruptError, "corrupt"),
)
_PERSISTENCE_CAUSE_BY_PHRASE = (
    (("turn lease",), "turn_lease"),
    (("closed by compression",), "compression_closed"),
    (("being compressed", "compression lease"), "compression"),
    (("was replaced underneath", "deleted state.db-wal", "deleted state.db-shm"), "replaced"),
    (_DB_CORRUPTION_MARKERS, "corrupt"),
    (("locked", "busy"), "locked"),
)


def _refine_corrupt_cause(exc_or_str) -> str:
    """``corrupt`` unless the error carries the store's ``db_path`` (stamped by the SessionDB
    quarantine), its result code is unscoped, and a read-only quick_check finds the canonical
    tables intact: then ``corrupt_unconfirmed``. The quarantine itself is untouched (a live
    handle that observed corruption must stop writing either way); only the guidance changes,
    so a healthy file is never sent down the recover / restore path (#97794). Memoised on the
    error object so the probe runs once per failure, not once per consumer."""
    cached = getattr(exc_or_str, "_persistence_cause", None)
    if cached is not None:
        return cached
    cause = "corrupt"
    db_path = getattr(exc_or_str, "db_path", None)
    if (
        db_path
        and _is_unscoped_corruption_code(getattr(exc_or_str, "sqlite_errorcode", None))
        and verify_canonical_tables_healthy(db_path)
    ):
        cause = "corrupt_unconfirmed"
    try:
        exc_or_str._persistence_cause = cause
    except (AttributeError, TypeError):  # plain strings
        pass
    return cause


def classify_persistence_error(exc_or_str) -> str:
    """Coarse cause bucket (PERSISTENCE_ERROR_CAUSES) so the user's guidance
    matches: "locked" = busy, retry; "disk" = full/read-only/permissions;
    "compression" = a live lease refused the write; "compression_closed" = adopt
    the rotated session id; "turn_lease" = fencing, not storage; "corrupt" =
    file damage (repair path, not disk space); "corrupt_unconfirmed" = corruption
    reported but the canonical tables verify intact (restart, do not recover);
    "fts_index" = SQLite scoped the corruption to the FTS index (the transcript
    store is not damaged); "replaced" = stop writing."""
    if exc_or_str is None:
        return "unknown"
    # Lease refusals contain neither "locked" nor "busy": match by type first,
    # then by phrase for strings that survived RPC wrapping. Order matters:
    # StateDbReplacedError covers DeletedWalGenerationError; corruption comes
    # BEFORE the lock/disk buckets ("disk image is malformed" contains "disk").
    for exc_type, cause in _PERSISTENCE_CAUSE_BY_TYPE:
        if isinstance(exc_or_str, exc_type):
            return _refine_corrupt_cause(exc_or_str) if cause == "corrupt" else cause
    # Provenance before prose: an FTS-scoped result code (or, without one, an fts5 report
    # naming messages_fts*) is index damage, never whole-file corruption (#97794).
    if is_fts_scoped_corruption_error(exc_or_str):
        return "fts_index"
    text = str(exc_or_str).lower()
    for markers, cause in _PERSISTENCE_CAUSE_BY_PHRASE:
        if any(marker in text for marker in markers):
            return _refine_corrupt_cause(exc_or_str) if cause == "corrupt" else cause
    if is_disk_full_error(exc_or_str) or any(m in text for m in ("disk", "readonly", "read-only")):
        return "disk"
    return "unknown"
