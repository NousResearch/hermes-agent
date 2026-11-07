"""Backend-neutral error taxonomy for the persistence boundary.

These exceptions are for exceptional conditions only. Expected negative results (a CAS loser receiving False, a claim returning None, a missing-row read returning None or an empty page) stay typed returns in the capability protocols; raising here instead of returning is a contract violation.

No backend-specific exception (sqlite3.Error, driver timeouts) ever crosses the boundary; backends translate into this taxonomy, chaining the original as ``__cause__``. ``TransientStoreError`` means retrying the same operation may succeed; ``PermanentStoreError`` means it will not. Every class carries ``cause``, one of :data:`PERSISTENCE_ERROR_CAUSES`, the bucket the store's error classification answers for it; a class whose ``cause`` is ``"unknown"`` is classified by its message text, as before. A class that replaces one of the store's built-in exception types keeps that built-in (``ValueError``, ``RuntimeError``) as a second base, so an ``except`` clause written for the built-in keeps catching it.

``PersistenceError`` is the catchable root of the taxonomy, not an error condition of its own: application boundaries catch it to contain the whole family, and backends never raise it directly.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from .identity import SessionId

__all__ = [
    "MUTATION_GUARD_REFUSALS",
    "PERSISTENCE_ERROR_CAUSES",
    "STORE_FAILURES",
    "AuthorityRefusal",
    "CapabilityUnsupported",
    "CompressionBusy",
    "CompressionClosed",
    "CompressionInProgress",
    "CompressionPublishRefused",
    "Conflict",
    "CooldownRestoreRefused",
    "CorruptStoreError",
    "DeclarativeDrift",
    "DeletedWalGeneration",
    "DiskFull",
    "ExportTooLarge",
    "FenceStale",
    "HeldRowRewritten",
    "MalformedSchema",
    "NotFound",
    "PermanentStoreError",
    "PersistenceError",
    "ProofRejected",
    "PruneRowStale",
    "PruneRowUnresolved",
    "RetiredSessionError",
    "RewindRefusalReason",
    "RewindRefused",
    "SearchIndexCorrupt",
    "SessionLedgered",
    "SessionRowMissing",
    "SessionWriteGuarded",
    "StaleFenceKind",
    "StoreContention",
    "StoreQuarantined",
    "StoreReplaced",
    "StoreUnavailable",
    "TransientStoreError",
    "TurnLeaseLost",
    "UnknownExecution",
    "busy_refusal",
]

# Every bucket of the store's error classification; consumers enumerate this tuple.
PERSISTENCE_ERROR_CAUSES = (
    "locked", "compression", "compression_closed", "turn_lease", "corrupt", "fts_index",
    "replaced", "deleted_wal", "disk", "session_row_missing", "unknown",
)


class PersistenceError(Exception):
    """Catchable root for every error the persistence boundary is allowed to raise; never raised directly."""

    cause = "unknown"


class NotFound(PersistenceError):
    """An operation that requires an existing row addressed one that does not exist in the caller's scope.

    Only for operations whose contract says absence is exceptional; lookups whose contract returns None/empty keep the typed return.
    """

    def __init__(self, entity: str, entity_id: str) -> None:
        self.entity = entity
        self.entity_id = entity_id
        super().__init__(f"{entity} {entity_id!r} not found in scope")


class SessionRowMissing(NotFound):
    """A transcript write found its session row gone (removed under a live writer); the caller recreates the row and replays the write."""

    cause = "session_row_missing"

    def __init__(self, session_id: SessionId) -> None:
        super().__init__("session", str(session_id))
        self.args = (f"FOREIGN KEY constraint failed: session {str(session_id)!r} has no row",)


class Conflict(PersistenceError):
    """A write collided with concurrent state in a way the operation's contract does not express as a typed loser result."""

    def __init__(self, entity: str, entity_id: str) -> None:
        self.entity = entity
        self.entity_id = entity_id
        super().__init__(f"conflicting concurrent state on {entity} {entity_id!r}")


class _SessionConflict(Conflict):
    """A ``Conflict`` on the ``session`` entity whose message is the store's own text for the refusal."""

    default_message = "conflicting concurrent state on session {session_id!r}"

    def __init__(self, session_id: SessionId, message: Optional[str] = None) -> None:
        self.session_id = session_id
        super().__init__("session", str(session_id))
        self.args = (message if message is not None else self.default_message.format(session_id=str(session_id)),)


class CompressionClosed(Conflict, RuntimeError):
    """A transcript write addressed a session already ended by compression; the operation performed no work.

    Carries ``tip``: the live continuation of the closed session as the backend resolved it in the same boundary crossing (the newest compression continuation; equal to the closed session id when none exists), so the agent's flush can adopt the tip and retry once without a second read. Whether the tip itself is still live is the caller's check.
    """

    cause = "compression_closed"

    def __init__(self, session_id: SessionId, tip: SessionId) -> None:
        self.session_id = session_id
        self.tip = tip
        super().__init__("session", str(session_id))
        self.args = (
            f"Session {str(session_id)!r} is closed by compression; adopt its live continuation before appending messages",
        )


class CompressionBusy(_SessionConflict, RuntimeError):
    """Compression owns the session and the caller does not hold its lease: a compressor whose own lease is gone fails fast, nothing written."""

    cause = "compression"
    default_message = "Compression lease lost before publication: {session_id}"


class CompressionInProgress(CompressionBusy):
    """A writer collided with a live compression lease held by another party; the compressor publishes in seconds, so the write may succeed later. Nothing written."""

    default_message = "Session {session_id!r} is being compressed by another writer"


class TurnLeaseLost(_SessionConflict, RuntimeError):
    """A transcript write presented a turn lease it no longer holds, or a live turn lease of another party refused a transcript rewrite; nothing written."""

    cause = "turn_lease"
    default_message = "Session turn lease lost; refusing transcript write for {session_id!r}"


class SessionWriteGuarded(_SessionConflict, RuntimeError):
    """A guarded delete refused because the session or a delegate child in its cascade holds a live turn lease or compression lease; nothing removed.

    The guard does not say which lease or which cascade member refused, so neither does this error: ``entity_id`` is the requested root.
    """

    default_message = "session '{session_id}' (or a delegate child) has an active turn lease or compression lock"


RewindRefusalReason = Literal["history_changed", "target_unidentified", "target_changed", "handoff_lost"]

_REWIND_REFUSAL_TEXT: dict[RewindRefusalReason, str] = {
    "history_changed": "session history changed before the rewind could be persisted",
    "target_unidentified": "rewind target has no durable row identity",
    "target_changed": "rewind target changed before it could be persisted",
    "handoff_lost": "rewind did not retain its compaction handoff",
}


class RewindRefused(Conflict, RuntimeError):
    """A user-turn rewind refused on a transcript precondition: ``reason`` is ``history_changed``, ``target_unidentified`` or ``target_changed`` (refused before any write) or ``handoff_lost`` (checked after the rewind committed: the composite carrier's handoff scaffold did not come back). The message is the store's text (``message``, else the reason's usual text), which callers show as-is."""

    def __init__(self, session_id: SessionId, reason: RewindRefusalReason, message: Optional[str] = None) -> None:
        self.session_id = session_id
        self.reason: RewindRefusalReason = reason
        super().__init__("transcript", str(session_id))
        self.args = (_REWIND_REFUSAL_TEXT[reason] if message is None else message,)


class HeldRowRewritten(_SessionConflict, RuntimeError):
    """A compaction commit found a held row already rewritten in place by another writer; nothing written, and the newer generation keeps the session."""

    default_message = "a held row of session {session_id} was rewritten in place before the commit"


class PruneRowUnresolved(_SessionConflict, RuntimeError):
    """An in-place prune rewrite could not name the exact live row of a changed message; nothing written, so the caller commits through the full compaction instead."""

    default_message = "a pruned message of session {session_id} is not rewritten in place"


class PruneRowStale(PruneRowUnresolved):
    """The named live row no longer holds the body being pruned: another writer rewrote this generation. Unlike plain ambiguity the caller must not fall back to the full compaction; it skips the pass."""

    default_message = "a row of session {session_id} no longer holds the pruned message"


class CooldownRestoreRefused(_SessionConflict, RuntimeError):
    """A compression-cooldown restore refused because the stored cooldown is not the one the snapshot was taken from; nothing written."""


class CompressionPublishRefused(_SessionConflict, RuntimeError):
    """A compression-child publish refused on a precondition of the parent or child (not the lease); nothing written. The message is the store's text."""


class SessionLedgered(_SessionConflict, RuntimeError):
    """A session with gateway admission or worker history cannot be moved between stores (offline profile repair)."""

    default_message = "{session_id} has gateway admission/worker history; not moved"





class TransientStoreError(PersistenceError):
    """The backend failed in a way that may succeed on retry (disk I/O, an exhausted pool, contention). ``reason`` is ``"busy"`` when the store was overloaded: a write that waited its whole patience for its turn."""

    cause = "disk"

    def __init__(self, *args: object, reason: Optional[str] = None) -> None:
        super().__init__(*args)
        self.reason = reason


class StoreContention(TransientStoreError):
    """Another writer holds the store or a lock on it; retrying after a wait may succeed.

    The one transient condition a waiter polls through (the turn-lease wait); every other ``TransientStoreError`` propagates to the caller at once.
    """

    cause = "locked"

    def __init__(self, *args: object, reason: Optional[str] = "busy") -> None:
        super().__init__(*args, reason=reason)


class DiskFull(TransientStoreError):
    """The filesystem cannot accept another write (no space left on the device)."""


class PermanentStoreError(PersistenceError):
    """The backend failed in a way that will not succeed on retry (constraint breakage, corrupt state, unsupported schema)."""


class CorruptStoreError(PermanentStoreError):
    """The store reported corruption; the operation performed no work."""

    cause = "corrupt"


class StoreQuarantined(CorruptStoreError):
    """Structural damage: this handle is quarantined (writes fail fast for its life) and the store's health is latched. The agent's flush diverts its batch to the transcript rescue on this condition and on ``StoreReplaced``, never on schema or index damage."""


class MalformedSchema(CorruptStoreError):
    """The store's schema catalog is malformed but repairable; canonical rows are intact and health is not latched."""


class SearchIndexCorrupt(CorruptStoreError):
    """Corruption confined to the search index: canonical rows are intact, health is not latched, and a rebuild of the index recovers."""

    cause = "fts_index"


class DeclarativeDrift(PermanentStoreError, RuntimeError):
    """The store lags the declared schema (a missing table or column) and browsing could not catch it up; raised by physical admin operations only."""

    def __init__(self, detail: str, *, owned: bool = False) -> None:
        super().__init__(detail)
        self.owned = owned


class StoreUnavailable(PersistenceError):
    """The backing store cannot be reached or opened at all; no operation outcome can be reported."""


class StoreReplaced(PersistenceError, RuntimeError):
    """The physical store was replaced underneath a live handle; the handle is fenced and must be re-acquired. The agent's flush diverts its batch to the transcript rescue on this condition."""

    cause = "replaced"


class DeletedWalGeneration(StoreReplaced):
    """A live process holds a deleted generation of the store's write-ahead log; writing through this handle would split the log, so writes stop until every holder exits."""

    cause = "deleted_wal"


class ExportTooLarge(PersistenceError, ValueError):
    """A bounded export refused to run because the session has more exportable messages than the caller's limit."""

    def __init__(self, session_id: SessionId, message_count: int, limit: int) -> None:
        self.session_id = session_id
        self.message_count = message_count
        self.limit = limit
        super().__init__(
            f"Session '{session_id}' has more than {limit:,} exportable messages; the JSON/JSONL "
            "backup is built in memory and capped per session. Use the dashboard Sessions page's streaming "
            "Export action, or set sessions.max_export_messages: 0 in config.yaml to disable the guard."
        )


class CapabilityUnsupported(PersistenceError):
    """The operation requires a capability this backend does not declare."""

    def __init__(self, capability: str) -> None:
        self.capability = capability
        super().__init__(f"backend does not support capability {capability!r}")


class AuthorityRefusal(PersistenceError, ValueError):
    """A refusal by the state authority. ``reason`` is the bounded wire code (callers compare it as a string); ``field`` and ``detail`` name the check that fired and why, so a client and the gateway log can tell the reasons' many checks apart instead of reporting the bare code.

    The codes store code raises: ``admission_conflict``, ``admission_started``, ``invalid_params``, ``not_found``, ``permission_denied``, ``queue_full``, ``revision_conflict``, ``runtime_coordination_required``, ``session_busy``, ``session_idle``, ``stale_epoch``, ``stale_generation``, ``storage_unavailable``, ``unknown_execution``, and the worker codes ``producer_not_started``, ``worker_not_live``, ``unsupported_producer``. Application code raises its own codes with the same class, and veto callables refuse by raising it.
    """

    def __init__(self, reason: str, *, field: str | None = None, detail: str | None = None) -> None:
        self.reason = reason
        self.field = _one_line(field, 80)
        self.detail = _one_line(detail, 160)
        super().__init__(reason)


def _one_line(value: object, limit: int) -> str | None:
    """A refusal's ``field``/``detail`` as the log line and the client frame carry it: whitespace collapses to single spaces, other non-printables drop, and the length is bounded below the client's frame limit."""
    if value is None:
        return None
    text = "".join(ch for ch in " ".join(str(value).split()) if ch.isprintable())
    return text[:limit] or None


class StaleFenceKind(Enum):
    """Which fence refused the operation; its value is the refusal's ``reason``."""

    STALE_EPOCH = "stale_epoch"
    STALE_GENERATION = "stale_generation"
    REVISION_CONFLICT = "revision_conflict"


class FenceStale(AuthorityRefusal):
    """An authority operation carried a fence value the store has moved past; no work was performed.

    ``kind`` discriminates the refused fence and fixes ``reason``; ``entity``/``entity_id`` name what the fence guards (the epoch singleton, a session, an admission, a worker execution).
    """

    def __init__(self, kind: StaleFenceKind, entity: str, entity_id: str) -> None:
        self.kind = kind
        self.entity = entity
        self.entity_id = entity_id
        super().__init__(kind.value)


class UnknownExecution(AuthorityRefusal):
    """Started work whose fate is unknown blocks this operation (fail-closed); it clears only through an explicit resolution or a proof-verified worker adoption."""

    def __init__(self, entity: str, entity_id: str) -> None:
        self.entity = entity
        self.entity_id = entity_id
        super().__init__("unknown_execution")


class ProofRejected(AuthorityRefusal):
    """A possession proof (adoption secret digest, terminal receipt proof) did not match the stored evidence; the operation performed no work."""

    def __init__(self, entity: str, entity_id: str) -> None:
        self.entity = entity
        self.entity_id = entity_id
        super().__init__("permission_denied")


class RetiredSessionError(AuthorityRefusal):
    """The addressed session id is permanently retired and cannot be created again; ``reason`` is the code the refusing operation answers (``session_deleted`` for an exclusive create, ``not_found`` for a plain create)."""

    def __init__(self, session_id: SessionId, reason: str = "session_deleted") -> None:
        self.session_id = session_id
        super().__init__(reason)


# The store failures a settlement retry loop retries; a refusal is never among them.
STORE_FAILURES = (TransientStoreError, PermanentStoreError, StoreUnavailable, StoreReplaced)

# What a transcript-guard check refuses with when a live turn lease or compression lease protects a target.
MUTATION_GUARD_REFUSALS = (TurnLeaseLost, CompressionInProgress)


def busy_refusal(exc: BaseException) -> AuthorityRefusal:
    """The client-facing refusal for one of :data:`MUTATION_GUARD_REFUSALS`; a held compression lease is named in ``field``, because stopping the turn does not end it."""
    return AuthorityRefusal("session_busy", field="compression" if isinstance(exc, CompressionInProgress) else None)
