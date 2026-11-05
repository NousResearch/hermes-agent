"""Transcript writes of the in-memory message reference: appends, detached results, replace, in-place row edits, reactions and the compression-failure cooldown."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Callable, Mapping, Optional, Sequence

from persistence import CooldownRestoreRefused, ProfileScope, RowVisibility
from persistence.conversation.messages import (
    BatchAppendProgress,
    BatchAppendResult,
    CooldownSnapshot,
    DelegationDelivery,
    MarkerPurgeReport,
    NewMessage,
    Reaction,
    ReplaceMode,
    ReplaceResult,
    RowDisposition,
    RowOutcome,
    RowUids,
    UnseenReaction,
)
from persistence.core.identity import JsonValue, MessageId, SessionId

from . import _turn_policy as policy
from .messages_base import MessageBase
from .transcript_store import (
    _LIVE_MISSING_METADATA,
    _METADATA_FIELDS,
    _IDENTITY_FIELDS,
    _OWNED_FIELDS,
    _Cooldown,
    _decoded_row,
    _payload,
    _row_outcome,
    _row_snapshot,
    _stamp_tool_call_uids,
    _stored_content,
    _StoredRow,
    _Transcript,
    compression_ancestors,
    insert_rows,
    resume_lineage,
    row_stamp,
    rows_of,
)
from .messages_compact import row_identity

_DELIVERY_KINDS = ("async_delegation_complete", "hidden")


def _resolve_target(target: _StoredRow, message: NewMessage) -> RowOutcome:
    """One batch row resolved onto the stored row it names (``agent.transcript_repair.resolve_and_repair_transcript_batch``)."""
    if message.db_row_snapshot is not None:
        if _row_snapshot(target.message) != message.db_row_snapshot:
            # Another writer changed the payload after the caller's flush: adopt its durable version, write nothing.
            return _row_outcome(target, RowDisposition.ADOPTED, _decoded_row(target.message))
        # Still what the caller committed: the caller's owned payload wins; the stored timestamp, other writers' metadata and the identity columns the caller does not carry stay.
        kept = {name: getattr(target.message, name) for name in _METADATA_FIELDS}
        kept.update({name: getattr(message, name) or getattr(target.message, name) for name in _IDENTITY_FIELDS})
        target.message = replace(_payload(message, keep_reasoning=message.role == "assistant"), role=target.message.role,
                                 timestamp=target.message.timestamp, **kept)
        lent: dict[str, JsonValue] = {name: getattr(target.message, name) for name in _LIVE_MISSING_METADATA
                                      if getattr(message, name) is None and getattr(target.message, name) is not None}
        return _row_outcome(target, RowDisposition.REPAIRED, {**lent, "_metadata_only": True} if lent else None)
    if message.role == "assistant" and policy.is_content_blank(target.message.content):
        # The provider streamed nothing before the first flush: fill the blank row's content in place, counters untouched.
        target.message = replace(target.message, content=message.content)
        return _row_outcome(target, RowDisposition.REPAIRED)
    if message.role == "assistant":
        # A concurrent winner already holds content the model saw: adopt its content only, never overwrite.
        return _row_outcome(target, RowDisposition.ADOPTED, {"content": target.message.content, "_content_only": True})
    return _row_outcome(target, RowDisposition.REPAIRED)


def _repair_target(transcript: _Transcript, message: NewMessage) -> Optional[_StoredRow]:
    """The same-role row ``message.repair_target`` names: itself while active, else the one active clone a compaction made of it, else the addressed row; ``None`` inserts fresh."""
    if message.repair_target is None:
        return None
    original = transcript.find(int(message.repair_target))
    if original is None or original.message.role != message.role:
        return None
    if original.visibility is RowVisibility.ACTIVE:
        return original
    clones = [r for r in transcript.rows if r.visibility is RowVisibility.ACTIVE and r.id != original.id and r.message == original.message]
    return clones[0] if len(clones) == 1 else original


# What a restore may leave off a copy of a stored row: the row's value stands in for a falsy one.
_RESTORE_MAY_OMIT = ("token_count", "compressed_summary", "tool_call_uids", "absorbed_message_uids")
# The columns a restored copy must replay exactly: the owned payload, its uid maps and the replay sidecars.
_REPLAY_FIELDS = (*_OWNED_FIELDS, "tool_call_uids", "absorbed_message_uids", "api_content", "codex_reasoning_items")


def _same_as_stored(message: NewMessage, stored: NewMessage) -> bool:
    """``message`` stored now would replay exactly as ``stored`` (``resolve_and_repair_transcript_batch``'s ``_same_as_stored``): the stored content seen as a load returns it, and the stored values standing in for what a restore may leave off."""
    candidate = _payload(message, keep_reasoning=message.role == "assistant")
    standing_in = {name: getattr(stored, name) for name in _RESTORE_MAY_OMIT if not getattr(message, name)}
    if candidate.content == _stored_content(policy.loaded_view(stored.role, stored.content)):
        standing_in["content"] = stored.content
    candidate = replace(candidate, **standing_in)
    return all(getattr(candidate, name) == getattr(stored, name) for name in _REPLAY_FIELDS)


def _restored_row(transcript: _Transcript, message: NewMessage) -> Optional[_StoredRow]:
    """The active row a restored batch row is a copy of (the durable-uid guard of ``resolve_and_repair_transcript_batch``): a ``restored_from_store`` row without a row address or version whose uid names an active row it replays exactly like; the newest active row of its role with that uid stands for it. ``None`` inserts fresh."""
    if not message.restored_from_store or message.repair_target is not None or message.db_row_snapshot is not None or not message.message_uid:
        return None
    active = [row for row in transcript.rows if row.visibility is RowVisibility.ACTIVE and row.message.message_uid == message.message_uid]
    if not any(_same_as_stored(message, row.message) for row in active):
        return None
    same_role = [row for row in active if row.message.role == message.role]
    return same_role[-1] if same_role else None


def _prestamped(messages: Sequence[NewMessage]) -> list[NewMessage]:
    """Tool-call uids stamped over the whole batch before it is chunked, so a result pairs with its call across a chunk boundary."""
    index: dict[str, str] = {}
    return [_stamp_tool_call_uids(message, index) for message in messages]


class MessageWritesMixin(MessageBase):
    """Appends, detached results, replace, row edits, reactions and the cooldown."""

    def append(self, session_id: SessionId, message: NewMessage, *, compression_lock_holder: Optional[str] = None,
               turn_lease_holder: Optional[str] = None, turn_lease_ttl_seconds: float = 300.0) -> MessageId:
        scope = self._backend.scope
        self._write_guards(scope, session_id, turn_lease_holder=turn_lease_holder, ttl=turn_lease_ttl_seconds)
        # Only the fields ``append_message`` takes are stored: given tool-call uids, absorbed uids and ``data_use`` are dropped.
        message = replace(message, tool_call_uids=None, tool_call_uid=None, absorbed_message_uids=(), data_use=())
        return MessageId(insert_rows(self._ns(scope), self._created(scope, session_id), [message], keep_reasoning=True)[0].id)

    def append_batch(self, session_id: SessionId, messages: Sequence[NewMessage], *, compression_lock_holder: Optional[str] = None,
                     turn_lease_holder: Optional[str] = None, chunk_rows: Optional[int] = None,
                     turn_lease_ttl_seconds: float = 300.0) -> BatchAppendResult:
        scope = self._backend.scope
        if not messages:
            return BatchAppendResult(outcomes=())
        stamped, chunks = list(messages), [messages]
        if chunk_rows is not None and len(messages) > chunk_rows:
            stamped = _prestamped(messages)
            chunks = [stamped[start:start + chunk_rows] for start in range(0, len(stamped), chunk_rows)]
        outcomes: list[RowOutcome] = []
        try:
            for chunk in chunks:
                outcomes.extend(self._append_chunk(scope, session_id, chunk, turn_lease_holder, turn_lease_ttl_seconds))
        except BaseException as exc:
            committed = [RowUids(o.message_uid, o.tool_call_uids, o.tool_call_uid) for o in outcomes]
            pending = [RowUids(m.message_uid, m.tool_call_uids, m.tool_call_uid) for m in stamped[len(outcomes):]]
            exc.batch_progress = BatchAppendProgress(outcomes=tuple(outcomes), uids=(*committed, *pending))  # type: ignore[attr-defined]
            raise
        return BatchAppendResult(outcomes=tuple(outcomes))

    def _append_chunk(self, scope: ProfileScope, session_id: SessionId, messages: Sequence[NewMessage], turn_lease_holder: Optional[str],
                      ttl: float) -> list[RowOutcome]:
        """One batch transaction: the guards, every row resolved onto the row it names, then the fresh rows inserted in batch order."""
        self._write_guards(scope, session_id, turn_lease_holder=turn_lease_holder, ttl=ttl)
        transcript = self._created(scope, session_id)
        resolved: list[Optional[RowOutcome]] = []
        fresh: list[NewMessage] = []
        for message in messages:
            target = _repair_target(transcript, message)
            if target is None and (restored := _restored_row(transcript, message)) is not None:
                # The writer's own durable row handed back: nothing is written.
                resolved.append(_row_outcome(restored, RowDisposition.REPAIRED))
                continue
            if target is None:
                resolved.append(None)
                fresh.append(message)
                continue
            calls_before, payload_before = len(target.message.tool_calls), target.message
            resolved.append(_resolve_target(target, message))
            if target.message is not payload_before:
                self._bump_transcript_revision(scope, session_id)
            if target.visibility is RowVisibility.ACTIVE:
                # A rewrite in place is not an insert: the counters keep the calls they counted.
                transcript.counter_skew_tool_calls += calls_before - len(target.message.tool_calls)
        inserted = iter(insert_rows(self._ns(scope), transcript, fresh))
        return [outcome if outcome is not None else _row_outcome(next(inserted), RowDisposition.APPENDED) for outcome in resolved]

    def _append_detached(self, scope: ProfileScope, session_id: SessionId, content: str, display_kind: str, metadata: Mapping[str, JsonValue],
                         match: Callable[[NewMessage], bool]) -> MessageId:
        """A result that arrived between turns recorded once as a user row (``SessionDetachedResultsMixin._append_detached_result``): a row ``match`` finds in the session or its compression ancestors, any visibility, is returned instead; a new insert refuses under a live turn lease or on a session closed by compression."""
        for _sid, row in rows_of(self._backend, scope, compression_ancestors(self._backend, scope, str(session_id))):
            if match(row.message):
                return MessageId(row.id)
        self._write_guards(scope, session_id, reject_live=("turn",))
        message = NewMessage(role="user", content=content, timestamp=time.time(), display_kind=display_kind, display_metadata=dict(metadata))
        return MessageId(insert_rows(self._ns(scope), self._created(scope, session_id), [message], keep_reasoning=True)[0].id)

    def append_delegation_delivery(self, session_id: SessionId, delivery: DelegationDelivery) -> MessageId:
        scope = self._backend.scope
        delegation_id, notice = delivery.identity
        kind = "hidden" if delivery.metadata.get("presentation_suppressed") else "async_delegation_complete"

        def match(message: NewMessage) -> bool:
            meta = message.display_metadata or {}
            stored_notice = meta.get("delivery_notice")
            return (message.display_kind in _DELIVERY_KINDS and meta.get("delegation_id") == delegation_id
                    and ("" if stored_notice is None else stored_notice) == notice)

        return self._append_detached(scope, session_id, delivery.content, kind, delivery.metadata, match)

    def append_completion_note(self, session_id: SessionId, admission_id: str, content: str, display_kind: str,
                               metadata: Mapping[str, JsonValue]) -> MessageId:
        scope = self._backend.scope
        return self._append_detached(scope, session_id, content, display_kind, {**metadata, "completion_admission": admission_id},
                                     lambda message: (message.display_metadata or {}).get("completion_admission") == admission_id)

    def replace_transcript(self, session_id: SessionId, messages: Sequence[NewMessage], mode: ReplaceMode, *,
                           reject_active_turn_lease: bool = False) -> ReplaceResult:
        scope = self._backend.scope
        self._write_guards(scope, session_id, reject_live=("compaction", "turn") if reject_active_turn_lease else (), insert=bool(messages))
        transcript = self._created(scope, session_id)
        # Every mode ends with the store's absolute counter SET over the rows left active.
        transcript.recount()
        kept: list[_StoredRow] = []
        if mode is ReplaceMode.DESTRUCTIVE:
            transcript.rows = []
        elif mode is ReplaceMode.PRESERVE_EXISTING_ARCHIVE:
            transcript.rows = [r for r in transcript.rows if r.visibility is not RowVisibility.ACTIVE]
        else:
            live = transcript.active()
            for row, message in zip(live, messages):
                m = row.message
                if row_identity(m.role, m.content, m.tool_call_id, m.tool_calls) != row_identity(message.role, message.content, message.tool_call_id, message.tool_calls):
                    break
                kept.append(row)
            for row in live[len(kept):]:
                row.visibility = RowVisibility.REWOUND
        inserted = insert_rows(self._ns(scope), transcript, messages[len(kept):])
        # A kept row is stamped with only its stored id and uid: its stamp has no snapshot.
        return ReplaceResult(rows=(*(replace(row_stamp(row), db_row_snapshot=None) for row in kept), *(row_stamp(row) for row in inserted)))

    def _active_row(self, scope: ProfileScope, session_id: SessionId, message_id: MessageId, role: Optional[str] = None) -> Optional[_StoredRow]:
        row = self._row_in_session(scope, session_id, message_id)
        if row is None or row.visibility is not RowVisibility.ACTIVE or (role is not None and row.message.role != role):
            return None
        return row

    def set_user_message_content(self, session_id: SessionId, message_id: MessageId, content: JsonValue, display_metadata: Optional[Mapping[str, JsonValue]] = None) -> bool:
        scope = self._backend.scope
        row = self._active_row(scope, session_id, message_id, "user")
        if row is None:
            return False
        if display_metadata is None:
            row.message = replace(row.message, content=_payload(replace(row.message, content=content)).content)
        else:
            stored = _payload(replace(row.message, content=content, display_metadata=dict(display_metadata)))
            row.message = replace(row.message, content=stored.content, display_metadata=stored.display_metadata, api_content=None)
        self._bump_transcript_revision(scope, session_id)
        return True

    def deactivate_message(self, session_id: SessionId, message_id: MessageId) -> bool:
        scope = self._backend.scope
        row = self._row_in_session(scope, session_id, message_id)
        if row is None:
            return False
        if row.visibility is RowVisibility.ACTIVE:
            self._deactivate(scope, session_id, row)
        return True

    def _deactivate(self, scope: ProfileScope, session_id: SessionId, row: _StoredRow) -> None:
        """Mark one active row rewound; the store's UPDATE touches no counter, so the counters still include the row and its calls."""
        row.visibility = RowVisibility.REWOUND
        transcript = self._created(scope, session_id)
        transcript.counter_skew_messages += 1
        transcript.counter_skew_tool_calls += len(row.message.tool_calls)

    def deactivate_messages_by_display_kind(self, session_id: SessionId, display_kind: str) -> int:
        scope = self._backend.scope
        transcript = self._transcript(scope, session_id)
        if transcript is None or not display_kind:
            return 0
        matched = [row for row in transcript.active() if row.message.display_kind == display_kind]
        for row in matched:
            self._deactivate(scope, session_id, row)
        return len(matched)

    def set_latest_matching_message_display_kind(self, session_id: SessionId, *, role: str, content: str, display_kind: str,
                                                 display_metadata: Optional[Mapping[str, JsonValue]] = None) -> bool:
        scope = self._backend.scope
        transcript = self._transcript(scope, session_id)
        if transcript is None or not session_id or not content or not display_kind:
            return False
        stored = _payload(NewMessage(role=role, content=content)).content
        for row in reversed(transcript.active()):
            if row.message.role == role and row.message.content == stored:
                metadata = dict(display_metadata) if display_metadata else None
                row.message = replace(row.message, display_kind=display_kind, display_metadata=metadata)
                return True
        return False

    def set_latest_user_api_content(self, session_id: SessionId, content: JsonValue, api_content: str) -> bool:
        scope = self._backend.scope
        transcript = self._transcript(scope, session_id)
        newest = next((row for row in reversed(transcript.active()) if row.message.role == "user"), None) if transcript else None
        return newest is not None and self._stamp_api_content(scope, session_id, newest, content, api_content)

    def set_message_api_content(self, session_id: SessionId, message_id: MessageId, content: JsonValue, api_content: str) -> bool:
        scope = self._backend.scope
        row = self._active_row(scope, session_id, message_id, "user")
        return row is not None and self._stamp_api_content(scope, session_id, row, content, api_content)

    def _stamp_api_content(self, scope: ProfileScope, session_id: SessionId, row: _StoredRow, content: JsonValue, api_content: str) -> bool:
        """Stamp the sidecar when the row still holds ``content`` as the store compares it (both sides encoded)."""
        if row.message.content != _payload(NewMessage(role="user", content=content)).content:
            return False
        row.message = replace(row.message, api_content=api_content)
        self._bump_transcript_revision(scope, session_id)
        return True

    def purge_stale_tool_call_markers(self, *, dry_run: bool = False, backup: bool = True) -> MarkerPurgeReport:
        scope = self._backend.scope
        owned = sorted(
            ((sid, row) for sid, transcript in self._ns(scope)["transcripts"].items() for row in transcript.rows
             if row.message.role == "assistant" and row.message.tool_calls and isinstance(row.message.content, str)
             and policy.STALE_TOOL_CALL_MARKER_RE.fullmatch(row.message.content.strip())),
            key=lambda pair: pair[1].id)
        matches = [row for _sid, row in owned]
        if not dry_run:
            for sid, row in owned:
                row.message = replace(row.message, content="")
                self._bump_transcript_revision(scope, SessionId(sid))
        # No physical snapshot facility on this backend: ``backup_path`` is None whatever ``backup`` says.
        return MarkerPurgeReport(dry_run=dry_run, rows_affected=len(matches), row_ids=tuple(MessageId(r.id) for r in matches), backup_path=None)

    def _visible_lineage_row(self, scope: ProfileScope, session_id: SessionId, message_id: MessageId) -> Optional[_StoredRow]:
        """The row a client can see under the session's display resume: an active or compaction-archived row of any segment of its resume lineage (``SessionMessagesMixin._reaction_row_query``)."""
        for _sid, row in rows_of(self._backend, scope, resume_lineage(self._backend, scope, str(session_id))):
            if row.id == int(message_id):
                return row if row.visibility is not RowVisibility.REWOUND else None
        return None

    def set_reaction(self, session_id: SessionId, message_id: MessageId, emoji: Optional[str], *, author: str) -> Optional[tuple[Reaction, ...]]:
        scope = self._backend.scope
        row = self._visible_lineage_row(scope, session_id, message_id) if session_id else None
        if row is None:
            return None
        previous = next((r for r in row.reactions if r.get("author") == author), None)
        kept = [r for r in row.reactions if r.get("author") != author]
        if emoji and (previous is None or previous.get("emoji") != emoji):
            kept.append({"emoji": emoji, "author": author, "at": time.time()})
        row.reactions = kept
        return tuple(Reaction(emoji=str(r.get("emoji") or ""), author=str(r.get("author") or ""), at=float(r.get("at") or 0.0), seen=bool(r.get("seen")))
                     for r in kept)

    def take_unseen_reactions(self, session_id: SessionId, *, author: str = "user") -> tuple[UnseenReaction, ...]:
        scope = self._backend.scope
        if not session_id:
            return ()
        taken: list[UnseenReaction] = []
        for _sid, row in rows_of(self._backend, scope, resume_lineage(self._backend, scope, str(session_id))):
            if row.visibility is RowVisibility.REWOUND or not row.reactions:
                continue
            marked: list[dict[str, Any]] = []
            for reaction in row.reactions:
                if reaction.get("author") == author and not reaction.get("seen"):
                    content = row.message.content
                    taken.append(UnseenReaction(row_id=MessageId(row.id), role=row.message.role, emoji=str(reaction.get("emoji") or ""),
                                                text=content if isinstance(content, str) else ""))
                    reaction = {**reaction, "seen": True}
                marked.append(reaction)
            if marked != row.reactions:
                row.reactions = marked
        return tuple(taken)

    def _cooldowns(self, scope: ProfileScope) -> dict[str, _Cooldown]:
        cooldowns: dict[str, _Cooldown] = self._ns(scope)["cooldowns"]
        return cooldowns

    def record_compression_cooldown(self, session_id: SessionId, until: float, error: Optional[str] = None) -> None:
        scope = self._backend.scope
        if not session_id or str(session_id) not in self._sessions(scope):
            return
        record = self._cooldowns(scope).setdefault(str(session_id), _Cooldown())
        if record.until is None or not record.until > until:
            record.until = until
        record.error = error

    def clear_compression_cooldown(self, session_id: SessionId) -> None:
        scope = self._backend.scope
        record = self._cooldowns(scope).get(str(session_id))
        if record is not None:
            record.until = record.error = None

    def restore_compression_cooldown(self, session_id: SessionId, snapshot: CooldownSnapshot) -> None:
        scope = self._backend.scope
        present = str(session_id) in self._sessions(scope)
        if not snapshot.present:
            if present:
                raise CooldownRestoreRefused(session_id, "cannot restore absent compression cooldown row: session now exists")
            return
        if not present:
            return
        self._cooldowns(scope)[str(session_id)] = _Cooldown(until=snapshot.until, error=snapshot.error)
