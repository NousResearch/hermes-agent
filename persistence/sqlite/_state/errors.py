"""Exception types and error-classification predicates for the state store.
Shared by the store holder and the store functions; string predicates match wrapped RPC
strings as well as live sqlite3 exceptions."""

import sqlite3

from storage_errors import _MALFORMED_SCHEMA_MARKERS, is_sqlite_lock_error
from storage_errors import classify_persistence_error as _classify_storage_error

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
    return isinstance(exc, sqlite3.OperationalError) and (
        is_sqlite_lock_error(exc) or any(marker in str(exc).lower() for marker in _TRANSIENT_SQLITE_MARKERS)
    )


def is_malformed_schema_error(exc: BaseException) -> bool:
    """Only SQLite's explicit malformed-schema text: a generic "disk image is
    malformed" may be any B-tree page, so runtime repair must fail closed on it."""
    return isinstance(exc, sqlite3.DatabaseError) and any(
        marker in str(exc).lower() for marker in _MALFORMED_SCHEMA_MARKERS
    )


# Every classify_persistence_error bucket; consumers enumerate this tuple.
PERSISTENCE_ERROR_CAUSES = (
    "locked", "compression", "compression_closed", "turn_lease", "corrupt", "fts_index",
    "replaced", "deleted_wal", "disk", "session_row_missing", "unknown",
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


class PruneRowUnresolvedError(RuntimeError):
    """An in-place prune rewrite could not name the exact live row of a changed message. Raised before
    any write, so the caller commits through ``archive_and_compact`` instead."""


class PruneRowStaleError(PruneRowUnresolvedError):
    """The named live row no longer holds the body the pruner is pruning: another writer already
    rewrote this generation. Unlike plain ambiguity this must NOT fall back to the full writer —
    that writer archives rows it never compared and would republish the stale held text over the
    newer one. Subclassing keeps callers that only know the parent failing closed-enough (they stop
    the in-place path); the prune caller tells the two apart and skips the pass entirely."""


class HeldRowRewrittenError(RuntimeError):
    """A full-writer commit found one of the held rows it was about to archive already rewritten in
    place by another writer. Raised INSIDE the commit's write transaction, after the caller's
    precheck but before any row is archived: a write that lands in that window is exactly the lost
    race the precheck cannot see (#124102). Callers treat it like ``StaleHeldHistory`` — nothing
    written, the pass is a no-op and the newer generation keeps the session."""


class SessionTurnLeaseLostError(RuntimeError):
    """A transcript write presented a turn-lease holder that no longer owns it.
    Fail-fast fencing (no ``_execute_write`` retry): a later writer may already
    be persisting a newer turn, and landing this one would interleave a stale reply."""


class SessionActiveWriteGuardError(RuntimeError):
    """Raised when an active turn lease or compression lock rejects session deletion."""


class StateDbReplacedError(RuntimeError):
    """The state.db path no longer names the file this store holder opened
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
STORAGE_RECOVERY_DOCS_URL = "https://hermes-agent.nousresearch.com/docs/user-guide/session-storage-recovery"

# Two layers (#110054): the first sentence is for the person reading a chat bubble or a banner (what
# happened, nothing is lost, the one thing to do); the rest is the operator detail. The phrase
# "deleted state.db-wal or state.db-shm" is the classifier's RPC-wrapped fingerprint — keep it.
_DELETED_WAL_GENERATION_MSG = (
    "FATAL: session storage stopped writing because another Hermes process still holds a deleted "
    "state.db-wal or state.db-shm inode (an old copy of the write-ahead log). Nothing is lost: quit "
    "every Hermes process on this profile (Desktop app, gateway, dashboard, cron), run `hermes doctor` "
    "(it names the processes still holding the log), then start Hermes again. Do not delete the WAL "
    "yourself and do not run `hermes doctor --fix` while they are running. "
    f"Guide: {STORAGE_RECOVERY_DOCS_URL} "
    "Detail: the path names a different (or missing) generation than the one this process holds "
    "open; opening or writing through it would mint a second WAL (split-brain). "
    "database.journal_mode: delete is operator containment, not a new default."
)


class StateDbCorruptError(sqlite3.DatabaseError):
    """A live store holder observed structural (non-FTS, non-replaced) corruption and
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
    # The WAL-generation error subclasses StateDbReplacedError so existing write diversion keeps
    # working; classify it first because its recovery artifact and operator action are different.
    (DeletedWalGenerationError, "deleted_wal"),
    (StateDbReplacedError, "replaced"),
    (StateDbCorruptError, "corrupt"),
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
    live agent (FK rejection; the flush recreates it)."""
    if exc_or_str is None:
        return "unknown"
    # Lease refusals contain neither "locked" nor "busy": match this backend's types first,
    # then the shared classifier (result code, then phrase for strings that survived RPC
    # wrapping). Order matters: StateDbReplacedError covers DeletedWalGenerationError.
    for exc_type, cause in _PERSISTENCE_CAUSE_BY_TYPE:
        if isinstance(exc_or_str, exc_type):
            return cause
    return _classify_storage_error(exc_or_str)
