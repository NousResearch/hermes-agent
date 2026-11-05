"""The SQLite message repository: the transcript verbs over the conversation store functions."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, ContextManager, Iterator, Mapping, Optional, Sequence

from hermes_state_errors import classify_persistence_error
from persistence import (
    CooldownRestoreRefused, PersistenceError, RewindRefused, RowVisibility, SessionRowMissing,
)
from persistence.conversation.messages import (
    AnchoredView, AnchoredWindow, BatchAppendProgress, BatchAppendResult, CompactionResult, CompressionCooldown, ConversationMessage,
    ConversationPeer, ConversationRequest, ConversationRevision, CooldownSnapshot, DelegationDelivery,
    HeldCoverage, HeldMessage, MarkerPurgeReport, MessageRecord, MessageStorageState, NewMessage, PagedMessage, PromptPage,
    PrUrlHit, PrunedRowChange, PruneRewriteResult, Reaction, ReadCheck, ReplaceMode, ReplaceResult, ResumeConversations,
    RewindPlan, RowDisposition, RowOutcome, RowStamp, RowUids, SegmentPage, SessionTimeline, TimelineEntry, TranscriptRead, TranscriptRow,
    TranscriptView,
    UnseenReaction, UserTurnRewind, Watermark,
)
from persistence.core.identity import JsonValue, MessageId, SessionId
from session_rules.turns import (
    _DB_PERSISTED_MARKER, MERGED_DURABLE_ROWS, OWN_ROW, RETIRED_DURABLE_ROWS, RETIRED_ROW, UNNAMED_DURABLE_ROWS, message_uid_or_none,
)

from .._backend import SQLiteStore, register_repository
from .._holder import StoreHolder
from ._store.api_content import set_latest_user_api_content, set_message_api_content
from ._store.compression import (
    clear_compression_failure_cooldown, get_compression_failure_cooldown_row, record_compression_failure_cooldown,
    restore_compression_failure_cooldown_row,
)
from ._store.detached_results import append_completion_note, append_delegation_delivery
from ._store.identity import _restore_row_identity
from ._store.messages import (
    _append_messages_in_transaction, _ensure_display_order, _execute_transcript_write,
    _row_to_message_dict, _stamp_tool_call_uids, append_message, append_messages_batch, archive_and_compact, assert_resume_safe,
    clear_messages, conversation_revision, deactivate_message, deactivate_messages_by_display_kind, display_message_count,
    find_pr_url_messages, get_active_message_ids, get_active_message_watermark, get_message_role, get_messages,
    get_messages_around, get_messages_as_conversation, get_resume_conversations, has_gateway_input_owner,
    has_platform_message_id, latest_conversation_boundary, latest_conversation_role, latest_message_row_id,
    message_count, purge_stale_tool_call_markers, replace_messages, resolve_active_row_id,
    set_latest_matching_message_display_kind, set_message_reaction, set_user_message_content, take_unseen_reactions,
)
from ._store.messages_sql import _parse_tool_calls
from ._store.prune_rewrite import rewrite_pruned_rows, rewritten_held_row_ids
from ._store.rewind import plan_user_turn_rewind, retain_refused_turn_for_display, rewind_user_turn
from ._store.rules import DB_ROW_SNAPSHOT, RESTORED_FROM_STORE
from ._store.search import get_anchored_view
from ._store.todo import get_latest_todo_result
from ._store.transcript_repair import _active_logical_message_row, transcript_row_snapshot
from ._errors import refusals
from ._store import stored_reads, timeline

__all__ = ["SQLiteMessageRepository"]

# The legacy store's rewind refusal texts, by the contract reason each one is.
_REWIND_REFUSALS = {
    "session history changed before the rewind could be persisted": "history_changed",
    "active transcript changed before the rewind could be persisted": "history_changed",
    "rewind target has no durable row identity": "target_unidentified",
    "rewind target changed before it could be persisted": "target_changed",
    "rewind did not retain its compaction handoff": "handoff_lost",
}

# ``NewMessage`` fields that travel under their own name when set.
_MESSAGE_OPTIONALS = (
    "tool_name", "tool_call_id", "display_kind", "api_content", "platform_message_id", "effect_disposition",
    "finish_reason", "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items",
    "codex_message_items", "token_count", "message_uid",
)
_WIRE_FIELDS = (
    "role", "content", "timestamp", "tool_call_id", "tool_calls", "tool_name", "effect_disposition", "token_count",
    "finish_reason", "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items",
    "codex_message_items", "platform_message_id", "api_content", "display_kind", "display_metadata", "message_uid",
    "absorbed_message_uids", "tool_call_uids", "tool_call_uid",
)
_SIDECARS = ("reasoning_details", "codex_reasoning_items", "codex_message_items")


@contextmanager
def _refusals(holder: StoreHolder, session_id: Optional[str] = None) -> Iterator[None]:
    """``refusals`` plus the transcript writes' own: a rejected session-row foreign key is ``SessionRowMissing``
    and a rewind's ``RuntimeError`` is ``RewindRefused`` with its reason."""
    try:
        with refusals(holder, session_id):
            yield
    except PersistenceError as exc:
        if session_id is not None and classify_persistence_error(exc.__cause__) == "session_row_missing":
            raise SessionRowMissing(SessionId(session_id)) from exc.__cause__
        raise
    except RuntimeError as exc:
        reason = _REWIND_REFUSALS.get(str(exc))
        if reason is None or session_id is None:
            raise
        raise RewindRefused(SessionId(session_id), reason, str(exc)) from exc


def _visibility(active: Any, compacted: Any) -> RowVisibility:
    if active:
        return RowVisibility.ACTIVE
    return RowVisibility.COMPACTION_ARCHIVED if compacted else RowVisibility.REWOUND


def _message_dict(message: NewMessage) -> dict[str, Any]:
    """One ``NewMessage`` as the message dict the legacy store's writers take."""
    msg: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.timestamp is not None:
        msg["timestamp"] = message.timestamp
    if message.tool_calls:
        msg["tool_calls"] = list(message.tool_calls)
    for name in _MESSAGE_OPTIONALS:
        value = getattr(message, name)
        if value is not None:
            msg[name] = value
    if message.display_metadata is not None:
        msg["display_metadata"] = dict(message.display_metadata)
    if message.compressed_summary:
        msg["_compressed_summary"] = True
    if message.observed:
        msg["observed"] = True
    if message.repair_target is not None:
        msg["_row_id"] = int(message.repair_target)
    if message.db_row_snapshot is not None:
        msg[DB_ROW_SNAPSHOT] = message.db_row_snapshot
    if message.absorbed_message_uids:
        msg["_absorbed_message_uids"] = list(message.absorbed_message_uids)
    if message.tool_call_uids is not None:
        msg["_tool_call_uids"] = dict(message.tool_call_uids)
    if message.tool_call_uid is not None:
        msg["_tool_call_uid"] = message.tool_call_uid
    if message.data_use:
        msg["data_use"] = list(message.data_use)
    if message.merged_durable_rows:
        msg[MERGED_DURABLE_ROWS] = message.merged_durable_rows
    if message.unnamed_durable_rows:
        msg[UNNAMED_DURABLE_ROWS] = message.unnamed_durable_rows
    if message.absorbed_row_ids:
        msg["_absorbed_row_ids"] = [int(i) for i in message.absorbed_row_ids]
    if message.restored_from_store:
        msg[RESTORED_FROM_STORE] = True
    return msg


def _held_dict(held: HeldMessage) -> dict[str, Any]:
    """One ``HeldMessage`` as the held message dict the legacy store compares."""
    msg: dict[str, Any] = {"role": held.role, "content": held.content}
    if held.tool_call_id is not None:
        msg["tool_call_id"] = held.tool_call_id
    if held.tool_calls:
        msg["tool_calls"] = list(held.tool_calls)
    if held.timestamp is not None:
        msg["timestamp"] = held.timestamp
    if held.row_id is not None:
        msg["_row_id"] = int(held.row_id)
    if held.persisted:
        msg["_db_persisted"] = True
    if held.db_row_snapshot is not None:
        msg[DB_ROW_SNAPSHOT] = held.db_row_snapshot
    if held.message_uid is not None:
        msg["message_uid"] = held.message_uid
    if held.absorbed_row_ids:
        msg["_absorbed_row_ids"] = [int(i) for i in held.absorbed_row_ids]
    if held.merged_durable_rows:
        msg[MERGED_DURABLE_ROWS] = held.merged_durable_rows
    if held.retired_durable_rows:
        msg[RETIRED_DURABLE_ROWS] = [_held_dict(row) for row in held.retired_durable_rows]
    if held.tool_call_ids:
        msg["tool_call_ids"] = list(held.tool_call_ids)
    if held.own_row:
        msg[OWN_ROW] = True
    if held.retired_row:
        msg[RETIRED_ROW] = True
    return msg


def _pruned_original(change: PrunedRowChange) -> dict[str, Any]:
    """The message dict the legacy rewrite names, compares and diffs: ``before`` as written, with the held original's
    fields over it, so the row id and ownership version come from the original alone."""
    before = {key: value for key, value in _message_dict(change.before).items() if key not in ("_row_id", DB_ROW_SNAPSHOT)}
    return {**before, **_held_dict(change.original)}


def _transcript_row(row: Mapping[str, Any]) -> TranscriptRow:
    """A stored row dict (``_row_to_message_dict``) as its record."""
    fields = {name: row.get(name) for name in _WIRE_FIELDS}
    return TranscriptRow(
        id=MessageId(int(row["id"])), session_id=SessionId(row["session_id"]),
        visibility=_visibility(row.get("active"), row.get("compacted")), observed=bool(row.get("observed")),
        compressed_summary=row.get("_compressed_summary") is True, **fields)


def _sidecar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _message_record(row: Mapping[str, Any]) -> MessageRecord:
    """A stored row dict as its typed record (reactions are not part of the record's metadata)."""
    metadata = row.get("display_metadata")
    if isinstance(metadata, dict):
        metadata = {key: value for key, value in metadata.items() if key != "reactions"} or None
    return MessageRecord(
        id=MessageId(int(row["id"])), role=row["role"], content=row.get("content"), timestamp=float(row["timestamp"]),
        tool_calls=tuple(row.get("tool_calls") or ()), tool_name=row.get("tool_name"),
        tool_call_id=row.get("tool_call_id"), display_kind=row.get("display_kind"),
        compressed_summary=bool(row.get("_compressed_summary")),
        visibility=_visibility(row.get("active"), row.get("compacted")), api_content=row.get("api_content"),
        display_metadata=metadata, platform_message_id=row.get("platform_message_id"),
        observed=bool(row.get("observed")), effect_disposition=row.get("effect_disposition"),
        finish_reason=row.get("finish_reason"), reasoning=row.get("reasoning"),
        reasoning_content=row.get("reasoning_content"),
        **{name: _sidecar(row.get(name)) for name in _SIDECARS}, message_uid=row.get("message_uid"))


def _conversation(messages: Sequence[Mapping[str, Any]]) -> tuple[ConversationMessage, ...]:
    return tuple(ConversationMessage.from_openai_message(m) for m in messages)


def _stamp(msg: Mapping[str, Any], stored: Mapping[int, Mapping[str, Any]]) -> RowStamp:
    """The row state a write left on one message dict; a kept row the write did not restamp takes its stored
    timestamp."""
    row_id = int(msg["_row_id"])
    timestamp = msg.get("timestamp")
    if timestamp is None:
        timestamp = stored[row_id]["timestamp"]
    return RowStamp(id=MessageId(row_id), timestamp=float(timestamp), db_row_snapshot=msg.get(DB_ROW_SNAPSHOT),
                    message_uid=msg.get("message_uid") or stored.get(row_id, {}).get("message_uid") or "",
                    tool_call_uids=msg.get("_tool_call_uids"), tool_call_uid=msg.get("_tool_call_uid"))


def _append_chunk(holder: StoreHolder, session_id: SessionId, messages: list[dict[str, Any]], **guards: Any) -> int:
    """One batch write transaction; returns the fresh-row count. A restored dict the repair left unwritten (stamped
    persisted, no ``_row_id``) takes, in the same transaction, the row state of the newest active row of its role
    carrying its uid, as a row repaired in place does."""
    if not messages:
        return 0

    def _do(conn) -> int:
        inserted = _append_messages_in_transaction(holder, conn, session_id, messages, **guards)
        for msg in messages:
            if msg.get(RESTORED_FROM_STORE) and msg.get(_DB_PERSISTED_MARKER) and "_row_id" not in msg:
                row = _active_logical_message_row(conn, session_id, msg["role"], message_uid_or_none(msg))
                msg.update({"_row_id": int(row["id"]), "timestamp": row["timestamp"], DB_ROW_SNAPSHOT: transcript_row_snapshot(row)})
                _restore_row_identity(row, msg)
        return inserted
    return _execute_transcript_write(holder, _do, messages, patience_s=holder._TRANSCRIPT_WRITE_PATIENCE_S)


def _row_outcomes(rows: Sequence[Mapping[str, Any]], count: int) -> tuple[RowOutcome, ...]:
    """The batch's outcomes from the row state the write stamped on each dict and its fresh-row ``count``. Fresh ids
    are minted inside the write, so they are the ``count`` largest; any other row resolved to an existing target:
    ADOPTED when it took a canonical row other than lent metadata, REPAIRED otherwise."""
    fresh = set(sorted({int(row["_row_id"]) for row in rows}, reverse=True)[:count])
    outcomes = []
    for row in rows:
        row_id, canonical = int(row["_row_id"]), row.get("_canonical_row")
        if row_id in fresh:
            disposition = RowDisposition.APPENDED
        elif canonical is not None and not canonical.get("_metadata_only"):
            disposition = RowDisposition.ADOPTED
        else:
            disposition = RowDisposition.REPAIRED
        outcomes.append(RowOutcome(id=MessageId(row_id), disposition=disposition, timestamp=float(row["timestamp"]),
                                   db_row_snapshot=row.get(DB_ROW_SNAPSHOT), canonical_row=canonical,
                                   message_uid=row.get("message_uid"), tool_call_uids=row.get("_tool_call_uids"),
                                   tool_call_uid=row.get("_tool_call_uid")))
    return tuple(outcomes)


class SQLiteMessageRepository:
    """See the ``persistence.conversation.messages`` protocol for the contract each method implements."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    # ── writes ──

    def append(self, session_id: SessionId, message: NewMessage, *, compression_lock_holder: Optional[str] = None,
               turn_lease_holder: Optional[str] = None, turn_lease_ttl_seconds: float = 300.0) -> MessageId:
        holder = self._store.holder()
        msg = _message_dict(message)
        with _refusals(holder, session_id):
            return MessageId(append_message(holder,
                session_id, message.role, content=message.content, tool_name=message.tool_name,
                tool_calls=msg.get("tool_calls"), tool_call_id=message.tool_call_id, token_count=message.token_count,
                finish_reason=message.finish_reason, reasoning=message.reasoning,
                reasoning_content=message.reasoning_content, reasoning_details=message.reasoning_details,
                codex_reasoning_items=message.codex_reasoning_items, codex_message_items=message.codex_message_items,
                platform_message_id=message.platform_message_id, observed=message.observed,
                effect_disposition=message.effect_disposition, _compressed_summary=message.compressed_summary,
                timestamp=message.timestamp, api_content=message.api_content, display_kind=message.display_kind,
                display_metadata=msg.get("display_metadata"), compression_lock_holder=compression_lock_holder,
                turn_lease_holder=turn_lease_holder, turn_lease_ttl_seconds=turn_lease_ttl_seconds,
                message_uid=message.message_uid))

    def append_batch(self, session_id: SessionId, messages: Sequence[NewMessage], *,
                     compression_lock_holder: Optional[str] = None, turn_lease_holder: Optional[str] = None,
                     chunk_rows: Optional[int] = None, turn_lease_ttl_seconds: float = 300.0) -> BatchAppendResult:
        holder = self._store.holder()
        dicts = [_message_dict(m) for m in messages]
        chunks = [dicts]
        if chunk_rows is not None and len(dicts) > chunk_rows:
            tool_uid_index: dict[str, str] = {}
            for msg in dicts:  # each chunk indexes only its own rows: pair results across chunk boundaries
                _stamp_tool_call_uids(msg, _parse_tool_calls(msg.get("tool_calls")), tool_uid_index)
            chunks = [dicts[start:start + chunk_rows] for start in range(0, len(dicts), chunk_rows)]
        committed: list[RowOutcome] = []
        try:
            with _refusals(holder, session_id):
                for chunk in chunks:  # one write transaction each
                    inserted = _append_chunk(holder,
                        session_id, chunk, compression_lock_holder=compression_lock_holder,
                        turn_lease_holder=turn_lease_holder, turn_lease_ttl_seconds=turn_lease_ttl_seconds)
                    committed.extend(_row_outcomes(chunk, inserted))
        except BaseException as exc:
            exc.batch_progress = BatchAppendProgress(outcomes=tuple(committed), uids=tuple(
                RowUids(message_uid=msg.get("message_uid") or None, tool_call_uids=msg.get("_tool_call_uids"),
                        tool_call_uid=msg.get("_tool_call_uid")) for msg in dicts))
            raise
        return BatchAppendResult(outcomes=tuple(committed))

    def append_delegation_delivery(self, session_id: SessionId, delivery: DelegationDelivery) -> MessageId:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return MessageId(append_delegation_delivery(holder, session_id, delivery.content, dict(delivery.metadata)))

    def append_completion_note(self, session_id: SessionId, admission_id: str, content: str, display_kind: str,
                               metadata: Mapping[str, JsonValue]) -> MessageId:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return MessageId(append_completion_note(holder, session_id, admission_id, content, display_kind,
                                                       dict(metadata)))

    def replace_transcript(self, session_id: SessionId, messages: Sequence[NewMessage], mode: ReplaceMode, *,
                           reject_active_turn_lease: bool = False) -> ReplaceResult:
        holder = self._store.holder()
        dicts = [_message_dict(m) for m in messages]
        for msg in dicts:  # a replace stamps no ownership version on a row it keeps
            msg.pop(DB_ROW_SNAPSHOT, None)
        with _refusals(holder, session_id):
            replace_messages(holder,
                session_id, dicts, active_only=mode is ReplaceMode.PRESERVE_EXISTING_ARCHIVE,
                archive_dropped=mode is ReplaceMode.ARCHIVE_DROPPED,
                reject_active_turn_lease=reject_active_turn_lease)
            kept = any(m.get("timestamp") is None or not m.get("message_uid") for m in dicts)
            stored = {int(r["id"]): r for r in get_messages(holder, session_id)} if kept else {}
        return ReplaceResult(rows=tuple(_stamp(m, stored) for m in dicts))

    def clear_transcript(self, session_id: SessionId) -> None:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            clear_messages(holder, session_id)

    def rewind_user_turn(self, session_id: SessionId, user_ordinal: int, *,
                         warm_history: Optional[Sequence[Mapping[str, Any]]] = None, require_retryable: bool = False,
                         require_composite: bool = False, adopt_row_ids: bool = False) -> UserTurnRewind:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            outcome = rewind_user_turn(holder,
                session_id, user_ordinal, warm_history=None if warm_history is None else [dict(m) for m in warm_history],
                require_retryable=require_retryable, require_composite=require_composite, adopt_row_ids=adopt_row_ids)
        return UserTurnRewind(prefix=tuple(outcome.prefix), live_view=outcome.live_view, live_text=outcome.live_text,
                              rewound_count=outcome.rewound_count, turns_undone=outcome.turns_undone)

    def retain_refused_turn_for_display(self, session_id: SessionId, *, notice: str,
                                        boundary_row_id: Optional[MessageId] = None) -> bool:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return retain_refused_turn_for_display(holder, session_id, notice=notice, boundary_row_id=boundary_row_id)

    def compact(self, session_id: SessionId, replacement: Sequence[NewMessage], *,
                watermark: Optional[Watermark] = None, carried_tail_count: int = 0,
                lock_holder: Optional[str] = None, model_config_patch: Optional[Mapping[str, JsonValue]] = None,
                carried: Sequence[HeldMessage] = (), coverage: Optional[HeldCoverage] = None,
                held_for_cas: Optional[Sequence[HeldMessage]] = None) -> CompactionResult:
        holder = self._store.holder()
        dicts = [_message_dict(m) for m in replacement]
        with _refusals(holder, session_id):
            active = archive_and_compact(holder,
                session_id, dicts, model_config_patch=None if model_config_patch is None else dict(model_config_patch),
                watermark=None if watermark is None else int(watermark.last_active or 0),
                lock_holder=lock_holder, tail_count=carried_tail_count,
                carried_messages=[_held_dict(m) for m in carried] or None,
                covered_ids=None if coverage is None else [int(i) for i in coverage.covered_ids],
                unresolved_held=None if coverage is None else [_held_dict(m) for m in coverage.unresolved],
                held_for_cas=None if held_for_cas is None else [_held_dict(m) for m in held_for_cas])
        return CompactionResult(active_count=active, rows=tuple(_stamp(m, {}) for m in dicts))

    def rewrite_pruned_rows(self, session_id: SessionId, changes: Sequence[PrunedRowChange], *,
                            model_config_patch: Optional[Mapping[str, JsonValue]] = None,
                            turn_lease_holder: Optional[str] = None,
                            turn_lease_ttl_seconds: float = 300.0) -> PruneRewriteResult:
        holder = self._store.holder()
        pairs = [(_pruned_original(c), _message_dict(c.replacement)) for c in changes]
        for _, replacement in pairs:
            replacement.pop(DB_ROW_SNAPSHOT, None)
        with _refusals(holder, session_id):
            rewritten = rewrite_pruned_rows(holder,
                session_id, pairs, model_config_patch=None if model_config_patch is None else dict(model_config_patch),
                turn_lease_holder=turn_lease_holder, turn_lease_ttl_seconds=turn_lease_ttl_seconds)
        return PruneRewriteResult(rewritten=rewritten,
                                  snapshots=tuple(replacement.get(DB_ROW_SNAPSHOT) for _, replacement in pairs))

    def purge_stale_tool_call_markers(self, *, dry_run: bool = False, backup: bool = True) -> MarkerPurgeReport:
        holder = self._store.holder()
        with _refusals(holder):
            report = purge_stale_tool_call_markers(holder, dry_run=dry_run, backup=backup)
        return MarkerPurgeReport(dry_run=report["dry_run"], rows_affected=report["rows_affected"],
                                 row_ids=tuple(MessageId(i) for i in report["row_ids"]),
                                 backup_path=report["backup_path"])

    def set_user_message_content(self, session_id: SessionId, message_id: MessageId, content: JsonValue, display_metadata: Optional[Mapping[str, JsonValue]] = None) -> bool:
        holder = self._store.holder()
        metadata = dict(display_metadata) if display_metadata is not None else None
        with _refusals(holder, session_id):
            return bool(set_user_message_content(holder, session_id, message_id, content, metadata))

    def deactivate_message(self, session_id: SessionId, message_id: MessageId) -> bool:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return bool(deactivate_message(holder, session_id, message_id))

    def deactivate_messages_by_display_kind(self, session_id: SessionId, display_kind: str) -> int:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return deactivate_messages_by_display_kind(holder, session_id, display_kind)

    def set_latest_matching_message_display_kind(self, session_id: SessionId, *, role: str, content: str,
                                                 display_kind: str,
                                                 display_metadata: Optional[Mapping[str, JsonValue]] = None) -> bool:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return set_latest_matching_message_display_kind(holder,
                session_id, role=role, content=content, display_kind=display_kind,
                display_metadata=None if display_metadata is None else dict(display_metadata))

    def set_latest_user_api_content(self, session_id: SessionId, content: JsonValue, api_content: str) -> bool:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return bool(set_latest_user_api_content(holder, session_id, content, api_content))

    def set_message_api_content(self, session_id: SessionId, message_id: MessageId, content: JsonValue,
                                api_content: str) -> bool:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return bool(set_message_api_content(holder, session_id, message_id, content, api_content))

    def record_compression_cooldown(self, session_id: SessionId, until: float, error: Optional[str] = None) -> None:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            record_compression_failure_cooldown(holder, session_id, until, error)

    def clear_compression_cooldown(self, session_id: SessionId) -> None:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            clear_compression_failure_cooldown(holder, session_id)

    def restore_compression_cooldown(self, session_id: SessionId, snapshot: CooldownSnapshot) -> None:
        holder = self._store.holder()
        row = {"session_exists": snapshot.present, "cooldown_until": snapshot.until, "error": snapshot.error}
        try:
            with _refusals(holder, session_id):
                restore_compression_failure_cooldown_row(holder, session_id, row)
        except RuntimeError as exc:
            if isinstance(exc, PersistenceError):
                raise
            raise CooldownRestoreRefused(session_id, str(exc)) from exc

    def set_reaction(self, session_id: SessionId, message_id: MessageId, emoji: Optional[str], *,
                     author: str) -> Optional[tuple[Reaction, ...]]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            reactions = set_message_reaction(holder, session_id, message_id, emoji, author=author)
        if reactions is None:
            return None
        return tuple(Reaction(emoji=r.get("emoji"), author=r.get("author"), at=float(r.get("at") or 0.0),
                              seen=bool(r.get("seen")))
                     for r in reactions)

    def take_unseen_reactions(self, session_id: SessionId, *, author: str = "user") -> tuple[UnseenReaction, ...]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            pending = take_unseen_reactions(holder, session_id, author=author)
        return tuple(UnseenReaction(row_id=MessageId(p["row_id"]), role=p["role"], emoji=p["emoji"], text=p["text"])
                     for p in pending)

    # ── reads ──

    def read_transcript(self, session_id: SessionId, request: TranscriptRead = TranscriptRead()) -> tuple[TranscriptRow, ...]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            rows = get_messages(holder,
                session_id, include_inactive=request.view is TranscriptView.ALL,
                include_compacted=request.view is TranscriptView.DISPLAY, limit=request.limit, offset=request.offset,
                latest=request.latest, after_id=None if request.after_id is None else int(request.after_id),
                include_ancestors=request.include_lineage)
        return tuple(_transcript_row(row) for row in rows)

    def active_message_ids(self, session_id: SessionId) -> tuple[MessageId, ...]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return tuple(MessageId(i) for i in get_active_message_ids(holder, session_id))

    def active_watermark(self, session_id: SessionId) -> Watermark:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            last = get_active_message_watermark(holder, session_id)
        return Watermark(MessageId(last) if last else None)

    def count_all(self) -> int:
        holder = self._store.holder()
        with _refusals(holder):
            return message_count(holder)

    def display_count(self, session_id: SessionId) -> int:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return display_message_count(holder, session_id)

    def conversation_revision(self, session_id: SessionId, *, include_lineage: bool = False) -> ConversationRevision:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            ids, count, max_id, revision = conversation_revision(holder, session_id, include_ancestors=include_lineage)
        return ConversationRevision(lineage_ids=tuple(SessionId(i) for i in ids), active_count=count,
                                    max_active_id=max_id, transcript_revision=revision)

    def read_conversation(self, session_id: SessionId,
                          request: ConversationRequest = ConversationRequest()) -> tuple[ConversationMessage, ...]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            messages = get_messages_as_conversation(holder,
                session_id, include_ancestors=request.include_lineage,
                include_inactive=request.view is TranscriptView.ALL, repair_alternation=request.repair_alternation,
                include_row_ids=request.include_row_ids, include_compacted=request.view is TranscriptView.DISPLAY)
        return _conversation(messages)

    def read_resume_conversations(self, session_id: SessionId) -> ResumeConversations:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            model, display = get_resume_conversations(holder, session_id)
        return ResumeConversations(model=_conversation(model), display=_conversation(display))

    def assert_resume_safe(self, session_id: SessionId, *, max_messages: int, tip_only: bool = False) -> int:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return assert_resume_safe(holder, session_id, max_messages, tip_only=tip_only)

    def read_around(self, session_id: SessionId, anchor: MessageId, *, window: int = 5) -> AnchoredWindow:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            around = get_messages_around(holder, session_id, anchor, window=window)
        return AnchoredWindow(window=tuple(_message_record(r) for r in around["window"]),
                              before=around["messages_before"], after=around["messages_after"])

    def read_anchored_view(self, session_id: SessionId, anchor: MessageId, *, window: int = 5, bookend: int = 3,
                           keep_roles: Optional[tuple[str, ...]] = ("user", "assistant")) -> AnchoredView:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            view = get_anchored_view(holder, session_id, anchor, window=window, bookend=bookend, keep_roles=keep_roles)
        return AnchoredView(
            window=tuple(_message_record(r) for r in view["window"]), before=view["messages_before"],
            after=view["messages_after"], bookend_start=tuple(_message_record(r) for r in view["bookend_start"]),
            bookend_end=tuple(_message_record(r) for r in view["bookend_end"]))

    def find_pr_url_hits(self, session_ids: Sequence[SessionId]) -> tuple[PrUrlHit, ...]:
        holder = self._store.holder()
        with _refusals(holder):
            hits = find_pr_url_messages(holder, list(session_ids))
        return tuple(PrUrlHit(session_id=SessionId(h["session_id"]), content=h["content"]) for h in hits)

    def latest_todo_result(self, session_id: SessionId, *, max_chars: int) -> Optional[str]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return get_latest_todo_result(holder, session_id, max_chars=max_chars)

    def read_session_timeline(self, session_id: SessionId, *, limit: int = 500,
                              after_row_id: int = 0) -> SessionTimeline:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            result = timeline.get_session_timeline(holder, session_id, limit=limit, after_row_id=after_row_id)
        page = result["pagination"]
        return SessionTimeline(
            entries=tuple(TimelineEntry(row_id=MessageId(e["row_id"]), preview=e["preview"], timestamp=e["timestamp"])
                          for e in result["entries"]),
            limit=page["limit"], after_row_id=page["after_row_id"], returned=page["returned"], total=page["total"],
            has_more=page["has_more"], next_cursor=page["next_cursor"])

    def read_prompt_page(self, session_id: SessionId, row_id: MessageId, *, limit: int = 120) -> Optional[PromptPage]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            result = timeline.get_session_messages_around(holder, session_id, row_id, limit=limit)
        if result is None:
            return None
        page = result["pagination"]
        return PromptPage(messages=tuple(_transcript_row(r) for r in result["messages"]), row_id=MessageId(page["row_id"]),
                          limit=page["limit"], returned=page["returned"], offset=page["offset"], total=page["total"],
                          has_older=page["has_older"], has_newer=page["has_newer"])

    def latest_conversation_role(self, session_id: SessionId) -> Optional[str]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return latest_conversation_role(holder, session_id)

    def latest_dialogue_message(self, session_id: SessionId) -> Optional[str]:
        # Read with the writer connection's 1 s lock wait.
        holder = self._store.holder()
        with _refusals(holder, session_id):
            with holder._read_ctx_with_timeout(1.0) as conn:
                row = conn.execute(
                    "SELECT content FROM messages"
                    " WHERE session_id = ? AND role IN ('user', 'assistant')"
                    " AND active = 1 AND content IS NOT NULL AND TRIM(content) != ''"
                    " ORDER BY id DESC LIMIT 1",
                    (session_id,)).fetchone()
        return row[0] if row else None

    def message_storage_state(self, message_id: MessageId) -> Optional[MessageStorageState]:
        if not message_id:
            return None
        # Read with the writer connection's 1 s lock wait.
        holder = self._store.holder()
        with _refusals(holder):
            with holder._read_ctx_with_timeout(1.0) as conn:
                row = conn.execute(
                    "SELECT session_id, active, compacted FROM messages WHERE id = ?", (message_id,)).fetchone()
        if row is None:
            return None
        return MessageStorageState(session_id=SessionId(row["session_id"]),
                                   visibility=_visibility(row["active"], row["compacted"]))

    def latest_row_id(self, session_id: SessionId, *, role: str = "user", offset: int = 0,
                      require_text: bool = True) -> Optional[MessageId]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            row_id = latest_message_row_id(holder, session_id, role=role, offset=offset, require_text=require_text)
        return None if row_id is None else MessageId(row_id)

    def message_role(self, session_id: SessionId, message_id: MessageId) -> Optional[str]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return get_message_role(holder, session_id, message_id)

    def resolve_active_row_id(self, session_id: SessionId, message_id: MessageId) -> Optional[MessageId]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            row_id = resolve_active_row_id(holder, session_id, message_id)
        return None if row_id is None else MessageId(row_id)

    def rewritten_held_row_ids(self, session_id: SessionId, held: Sequence[HeldMessage]) -> tuple[MessageId, ...]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return tuple(MessageId(i) for i in rewritten_held_row_ids(holder, session_id, [_held_dict(m) for m in held]))

    def plan_user_turn_rewind(self, session_id: SessionId, user_ordinal: int, *,
                              warm_history: Optional[Sequence[Mapping[str, Any]]] = None,
                              require_retryable: bool = False, require_composite: bool = False) -> RewindPlan:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            plan = plan_user_turn_rewind(holder,
                session_id, user_ordinal, warm_history=None if warm_history is None else [dict(m) for m in warm_history],
                require_retryable=require_retryable, require_composite=require_composite)
        return RewindPlan(
            target_row_id=MessageId(plan.target_row_id),
            expected_active_ids=tuple(MessageId(i) for i in plan.expected_active_ids),
            expected_target_content=plan.expected_target_content, scaffold=plan.scaffold, prefix=tuple(plan.prefix),
            durable_prefix=tuple(plan.durable_prefix), live_view=plan.live_view, live_text=plan.live_text,
            durable_user=tuple(plan.durable_user), user_ordinal=plan.user_ordinal)

    def has_platform_message_id(self, session_id: SessionId, platform_message_id: str) -> bool:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return has_platform_message_id(holder, session_id, platform_message_id)

    def has_gateway_input_owner(self, session_id: SessionId, owner: str) -> bool:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return has_gateway_input_owner(holder, session_id, owner)

    def latest_conversation_boundary(self, peer: ConversationPeer) -> Optional[int]:
        holder = self._store.holder()
        with _refusals(holder):
            return latest_conversation_boundary(holder, peer.session_key, peer.source)

    def read_compression_cooldown(self, session_id: SessionId, at: float) -> Optional[CompressionCooldown]:
        snapshot = self.read_compression_cooldown_snapshot(session_id)
        if snapshot.until is None or float(snapshot.until) <= at:
            return None
        return CompressionCooldown(until=float(snapshot.until), error=snapshot.error)

    def read_compression_cooldown_snapshot(self, session_id: SessionId) -> CooldownSnapshot:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            row = get_compression_failure_cooldown_row(holder, session_id)
        if not row["session_exists"]:
            return CooldownSnapshot(present=False)
        return CooldownSnapshot(present=True, until=row["cooldown_until"], error=row["error"])

    def terminal_tool_call_payloads(self, *, since_ts: float) -> Iterator[str]:
        holder = self._store.holder()
        with _refusals(holder):
            with holder._read_ctx() as conn:
                cur = conn.execute(
                    "SELECT tool_calls FROM messages WHERE role='assistant' AND tool_calls IS NOT NULL "
                    "AND tool_calls LIKE '%terminal%' AND timestamp >= ?",
                    (since_ts,),
                )
                while rows := cur.fetchmany(2000):
                    for (raw,) in rows:
                        yield raw

    def blocked_tool_result_payloads(self, *, since_ts: float) -> Iterator[tuple[str, Optional[str]]]:
        holder = self._store.holder()
        with _refusals(holder):
            with holder._read_ctx() as conn:
                cur = conn.execute(
                    "SELECT tool_call_id, content FROM messages "
                    "WHERE role='tool' AND tool_call_id IS NOT NULL AND timestamp >= ? "
                    "AND (content LIKE '%BLOCKED%' OR content LIKE '%approval%')",
                    (since_ts,),
                )
                while rows := cur.fetchmany(2000):
                    yield from ((tool_call_id, content) for tool_call_id, content in rows)

    def has_any_message(self) -> bool:
        holder = self._store.holder()
        with _refusals(holder):
            with holder._read_ctx() as conn:
                if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'").fetchone() is None:
                    return False
                return conn.execute("SELECT 1 FROM messages LIMIT 1").fetchone() is not None

    def spillover_reference_rows(self) -> tuple[tuple[Optional[str], Optional[str]], ...]:
        # At most 20001 rows: one past the reference set the spillover sweep trusts.
        holder = self._store.holder()
        with _refusals(holder):
            with holder._read_ctx() as conn:
                rows = conn.execute(
                    "SELECT content, api_content FROM messages WHERE "
                    "(instr(content, 'spillover') > 0 OR instr(api_content, 'spillover') > 0) LIMIT 20001"
                ).fetchall()
        return tuple((content, api_content) for content, api_content in rows)

    def read_snapshot(self) -> ContextManager[ReadCheck]:
        return self._store.holder().transcript_read_snapshot()

    def display_index_complete(self, session_id: SessionId) -> bool:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return stored_reads.transcript_display_index_complete(holder, session_id)

    def ensure_display_index(self, session_id: SessionId) -> bool:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return _ensure_display_order(holder, session_id)

    def display_generations(self, session_id: SessionId, lineage_ids: Sequence[SessionId]) -> tuple[PagedMessage, ...]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            found = stored_reads.display_generations(holder, session_id, [str(sid) for sid in lineage_ids])
        return tuple(PagedMessage(message=ConversationMessage.from_openai_message(message), position=int(durable),
                                  durable_id=MessageId(int(durable))) for message, durable in found)

    def transcript_segment_page(self, session_id: SessionId, *, latest: bool, after: Optional[tuple[int, int]], count: int,
                                inclusive: bool = False) -> SegmentPage:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            found, full, edge, start = stored_reads.segment_page_messages(
                holder, session_id, latest=latest, after=None if after is None else (int(after[0]), int(after[1])),
                count=count, inclusive=inclusive)
        return SegmentPage(messages=tuple(PagedMessage(message=ConversationMessage.from_openai_message(message),
                                                       position=int(position), durable_id=MessageId(int(durable)))
                                          for message, (position, durable) in found),
                           full=full, edge=None if edge is None else (int(edge[0]), int(edge[1])), start=start)

    def transcript_handoff_rows(self, shown_ids: Sequence[MessageId], segment_ids: Sequence[SessionId]) -> Mapping[MessageId, tuple[TranscriptRow, ...]]:
        holder = self._store.holder()
        with _refusals(holder):
            found = stored_reads.transcript_handoff_rows(holder, [int(i) for i in shown_ids], [str(s) for s in segment_ids])
        return {MessageId(shown): tuple(_transcript_row(row) for row in rows) for shown, rows in found.items()}

    def transcript_handoffs_shown_before(self, session_id: SessionId, before: int, handoffs: Sequence[tuple[Optional[str], JsonValue]]) -> tuple[TranscriptRow, ...]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            rows = stored_reads.transcript_handoffs_shown_before(holder, session_id, before, handoffs)
        return tuple(_transcript_row(row) for row in rows)

    def transcript_tool_calls_before(self, segment_ids: Sequence[SessionId], before_id: MessageId, *, limit: int = 16) -> tuple[tuple[Optional[str], Optional[str]], ...]:
        holder = self._store.holder()
        with _refusals(holder):
            return tuple(stored_reads.transcript_tool_calls_before(holder, [str(s) for s in segment_ids], int(before_id), limit))

    def transcript_cursor_state(self, session_id: SessionId, through_id: MessageId, lineage_ids: Sequence[SessionId]) -> tuple[int, int, int]:
        holder = self._store.holder()
        with _refusals(holder, session_id):
            return stored_reads.transcript_cursor_state(holder, session_id, int(through_id), [str(s) for s in lineage_ids])


register_repository("messages", SQLiteMessageRepository)
