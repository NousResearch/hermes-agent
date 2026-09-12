"""Exception types and error-classification predicates for the state store.
Shared by hermes_state and its mixins; string predicates match wrapped RPC
strings as well as live sqlite3 exceptions."""

import errno
import sqlite3

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
    "locked", "compression", "compression_closed", "turn_lease", "corrupt", "replaced", "disk",
    "unknown",
)


# "Database FILE structurally damaged" substrings. "database disk image is
# malformed" contains "disk", so this check MUST run before the disk bucket in
# classify_persistence_error or B-tree corruption reads as "free some disk space".
# Kept as plain substrings so sqlite3.DatabaseError, wrapped RPC strings, and logged message text all match
# the same helper. See #77386.
_DB_CORRUPTION_MARKERS = (
    "malformed", "file is not a database", "not a database", "database corruption",
)


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


def classify_persistence_error(exc_or_str) -> str:
    """Coarse cause bucket (PERSISTENCE_ERROR_CAUSES) so the user's guidance
    matches: "locked" = busy, retry; "disk" = full/read-only/permissions;
    "compression" = a live lease refused the write; "compression_closed" = adopt
    the rotated session id; "turn_lease" = fencing, not storage; "corrupt" =
    file damage (repair path, not disk space); "replaced" = stop writing."""
    if exc_or_str is None:
        return "unknown"
    # Lease refusals contain neither "locked" nor "busy": match by type first,
    # then by phrase for strings that survived RPC wrapping. Order matters:
    # StateDbReplacedError covers DeletedWalGenerationError; corruption comes
    # BEFORE the lock/disk buckets ("disk image is malformed" contains "disk").
    for exc_type, cause in _PERSISTENCE_CAUSE_BY_TYPE:
        if isinstance(exc_or_str, exc_type):
            return cause
    text = str(exc_or_str).lower()
    for markers, cause in _PERSISTENCE_CAUSE_BY_PHRASE:
        if any(marker in text for marker in markers):
            return cause
    if is_disk_full_error(exc_or_str) or any(m in text for m in ("disk", "readonly", "read-only")):
        return "disk"
    return "unknown"
