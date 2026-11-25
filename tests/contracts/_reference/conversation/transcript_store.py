"""Transcript storage of the in-memory reference: the stored row, a session's transcript, the scope namespace, and the row codecs and projections the message and lineage references share.

Rows live in the scope store under ``messages``; ids are minted per scope in insertion order, so id order is the append sequence. A stored row keeps its payload as a ``NewMessage`` whose ``timestamp`` and ``message_uid`` are always set.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime
from typing import Any, Iterable, Optional, Sequence

from persistence import ProfileScope, RowVisibility
from persistence.conversation.messages import (
    DB_PERSISTED_MARKER_KEY,
    MessageRecord,
    NewMessage,
    RowDisposition,
    RowOutcome,
    RowStamp,
    TranscriptRow,
)
from persistence.core.identity import JsonValue, MessageId, SessionId

from .._real_text import real_as_text
from ..backend import InMemoryBackend
from . import _turn_policy as policy

#: End reasons that advance a peer's conversation generation.
RESET_END_REASONS = ("session_reset", "session_switch", "idle", "daily", "suspended", "resume_pending_expired")

_SIDECARS = ("reasoning_details", "codex_reasoning_items", "codex_message_items")
_REASONING_FIELDS = ("reasoning", "reasoning_content", *_SIDECARS)
# Columns other same-process writers stamp after a flush, and the identity columns minted at insert: neither is part of the ownership version, and a snapshot-matched rewrite keeps the stored values.
_METADATA_FIELDS = ("display_kind", "display_metadata", "api_content", "codex_reasoning_items", "platform_message_id")
_IDENTITY_FIELDS = ("message_uid", "absorbed_message_uids", "tool_call_uids", "tool_call_uid")
_LIVE_MISSING_METADATA = ("display_kind", "display_metadata")
_UNOWNED_FIELDS = frozenset({"role", "timestamp", "repair_target", "db_row_snapshot", "data_use", "restored_from_store", *_METADATA_FIELDS, *_IDENTITY_FIELDS})
_OWNED_FIELDS = tuple(f.name for f in fields(NewMessage) if f.name not in _UNOWNED_FIELDS)
_PR_URL = re.compile("/pull/", re.IGNORECASE | re.ASCII)
# Multimodal content is stored NUL-sentinel-prefixed, and SQL ``length()`` counts characters before the first NUL.
_NUL = "\x00"


@dataclass
class _StoredRow:
    id: int
    message: NewMessage
    visibility: RowVisibility = RowVisibility.ACTIVE
    # The display slot an in-place prune kept for a rewritten row (its group's first row id), else None.
    display_slot: Optional[int] = None

    @property
    def reactions(self) -> list[dict[str, Any]]:
        stored = (self.message.display_metadata or {}).get("reactions")
        return [item for item in stored if isinstance(item, dict)] if isinstance(stored, list) else []

    @reactions.setter
    def reactions(self, values: list[dict[str, Any]]) -> None:
        metadata = dict(self.message.display_metadata or {})
        if values:
            metadata["reactions"] = values
        else:
            metadata.pop("reactions", None)
        self.message = replace(self.message, display_metadata=metadata or None)

    @property
    def epoch(self) -> float:
        return float(self.message.timestamp or 0.0)

    def as_record(self) -> MessageRecord:
        m = copy.deepcopy(self.message)
        return MessageRecord(
            id=MessageId(self.id), role=m.role, content=m.content, timestamp=self.epoch,
            tool_calls=m.tool_calls, tool_name=m.tool_name, tool_call_id=m.tool_call_id,
            display_kind=m.display_kind, compressed_summary=m.compressed_summary, visibility=self.visibility,
            api_content=m.api_content,
            display_metadata={key: value for key, value in (m.display_metadata or {}).items() if key != "reactions"} or None,
            platform_message_id=m.platform_message_id, observed=m.observed,
            effect_disposition=m.effect_disposition, finish_reason=m.finish_reason,
            reasoning=m.reasoning, reasoning_content=m.reasoning_content,
            reasoning_details=_record_sidecar(m.reasoning_details),
            codex_reasoning_items=_record_sidecar(m.codex_reasoning_items),
            codex_message_items=_record_sidecar(m.codex_message_items),
            message_uid=m.message_uid,
        )

    def as_transcript_row(self, session_id: str) -> TranscriptRow:
        m = copy.deepcopy(self.message)
        return TranscriptRow(
            id=MessageId(self.id), session_id=SessionId(session_id), role=m.role, content=m.content,
            timestamp=self.epoch, visibility=self.visibility,
            tool_call_id=m.tool_call_id, tool_calls=list(m.tool_calls) if m.tool_calls else None,
            tool_name=m.tool_name, effect_disposition=m.effect_disposition, token_count=m.token_count,
            finish_reason=m.finish_reason, reasoning=m.reasoning, reasoning_content=m.reasoning_content,
            reasoning_details=_sidecar_text(m.reasoning_details), codex_reasoning_items=_sidecar_text(m.codex_reasoning_items),
            codex_message_items=_sidecar_text(m.codex_message_items), platform_message_id=m.platform_message_id,
            observed=m.observed, compressed_summary=m.compressed_summary, api_content=m.api_content,
            display_kind=m.display_kind, display_metadata=dict(m.display_metadata) if m.display_metadata else None,
            message_uid=m.message_uid,
            absorbed_message_uids=json.dumps(list(m.absorbed_message_uids)) if m.absorbed_message_uids else None,
            tool_call_uids=json.dumps(dict(m.tool_call_uids), sort_keys=True) if m.tool_call_uids else None,
            tool_call_uid=m.tool_call_uid,
        )


@dataclass
class _Transcript:
    rows: list[_StoredRow] = field(default_factory=list)
    rewind_count: int = 0
    # The store keeps message/tool-call counters that inserts bump and recounting writes (replace, rewind, compaction) set to the active rows; deactivation and an owned rewrite leave them alone. The skew is what those non-counting writes moved since the last recount.
    counter_skew_messages: int = 0
    counter_skew_tool_calls: int = 0

    def active(self) -> list[_StoredRow]:
        return [r for r in self.rows if r.visibility is RowVisibility.ACTIVE]

    def counters(self) -> tuple[int, int]:
        """``(message_count, tool_call_count)`` as the store's per-operation counters read."""
        active = self.active()
        return (len(active) + self.counter_skew_messages,
                sum(len(r.message.tool_calls) for r in active) + self.counter_skew_tool_calls)

    def recount(self) -> None:
        """A write the store follows with an absolute counter SET over the active rows."""
        self.counter_skew_messages = self.counter_skew_tool_calls = 0

    def find(self, message_id: int) -> Optional[_StoredRow]:
        return next((row for row in self.rows if row.id == int(message_id)), None)


@dataclass
class _Cooldown:
    until: Optional[float] = None
    error: Optional[str] = None


def _messages_namespace(backend: InMemoryBackend, scope: ProfileScope) -> dict[str, Any]:
    ns: dict[str, Any] = backend.scope_store(scope).setdefault("messages", {"seq": 0, "transcripts": {}, "cooldowns": {}})
    ns.setdefault("generations", {})
    return ns


def transcripts_of(backend: InMemoryBackend, scope: ProfileScope) -> dict[str, _Transcript]:
    transcripts: dict[str, _Transcript] = _messages_namespace(backend, scope)["transcripts"]
    return transcripts


def mint_row_id(namespace: dict[str, Any]) -> int:
    namespace["seq"] += 1
    return int(namespace["seq"])


def _epoch(message: NewMessage) -> float:
    """The stored epoch seconds of a row's payload."""
    return float(message.timestamp or 0.0)


def _stored_content(content: Any) -> Any:
    """Content as the store keeps it: lone surrogates replaced, structured values copied, other scalars as their TEXT spelling. A string carrying the reserved ``\\x00json:`` prefix reads back verbatim."""
    if isinstance(content, str):
        return re.sub(r"[\ud800-\udfff]", "\ufffd", content)
    if isinstance(content, (dict, list, tuple)):
        return json.loads(json.dumps(content))
    if isinstance(content, bool):
        return str(int(content))
    if isinstance(content, int):
        return str(content)
    if isinstance(content, float):
        return real_as_text(content)
    return content


def _payload(message: NewMessage, *, keep_reasoning: bool = True) -> NewMessage:
    """The stored form of a written row, a copy that shares no value with the caller's: content normalized, ``data_use`` folded into ``display_metadata``, empty sidecars absent, reasoning dropped unless kept, and the caller's resolution bookkeeping (target id, snapshot, restored mark) never stored."""
    message = copy.deepcopy(message)
    metadata = dict(message.display_metadata) if isinstance(message.display_metadata, dict) and message.display_metadata else None
    if message.data_use:
        metadata = {**(metadata or {}), "data_use": list(message.data_use)}
    reasoning: dict[str, Any] = {name: None for name in _REASONING_FIELDS} if not keep_reasoning else {
        name: getattr(message, name) or None for name in _SIDECARS}
    return replace(message, content=_stored_content(message.content), display_metadata=metadata, data_use=(),
                   repair_target=None, db_row_snapshot=None, restored_from_store=False, **reasoning)


def _sidecar_text(value: Any) -> Optional[str]:
    """A JSON sidecar as its stored text: absent when falsy, a string kept as given."""
    if not value:
        return None
    return value if isinstance(value, str) else json.dumps(value)


def _replayed_sidecar(value: Any) -> Any:
    """A stored sidecar as the replay decodes it: text parsed, undecodable text ``None``."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _record_sidecar(value: Any) -> Any:
    """A stored sidecar on the typed row record: text parsed, undecodable text carried as stored."""
    if not value:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _text_length(content: Any) -> int:
    """SQL ``length()`` of the stored content: characters before the first NUL for text, 0 for NULL and for sentinel-prefixed structured content."""
    if content is None or isinstance(content, (list, dict)):
        return 0
    text = content if isinstance(content, str) else str(content)
    return len(text.split(_NUL, 1)[0])


def _hashable(value: Any) -> Any:
    """A column value as stored for identity: scalars as they are, structured values by their serialized text."""
    return value if isinstance(value, (str, int, float, bytes, type(None))) else json.dumps(value)


def _row_snapshot(message: NewMessage) -> str:
    """The reference's opaque ownership version: a digest of the owned payload only."""
    owned = [[name, getattr(message, name)] for name in _OWNED_FIELDS]
    encoded = json.dumps(owned, sort_keys=True, default=repr, ensure_ascii=False).encode("utf-8", "surrogatepass")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _identity_keys(message: NewMessage) -> dict[str, JsonValue]:
    """The stored identity columns under their live keys."""
    out: dict[str, JsonValue] = {}
    if message.message_uid:
        out["message_uid"] = message.message_uid
    if message.tool_call_uids:
        out["_tool_call_uids"] = dict(message.tool_call_uids)
    if message.tool_call_uid:
        out["_tool_call_uid"] = message.tool_call_uid
    if message.absorbed_message_uids:
        out["_absorbed_message_uids"] = list(message.absorbed_message_uids)
    return out


def _decoded_row(message: NewMessage) -> dict[str, JsonValue]:
    """A stored row decoded for a live dict (``SessionMessagesMixin._decoded_repair_row``: present keys only, reasoning on assistant rows)."""
    message = copy.deepcopy(message)
    row: dict[str, JsonValue] = {"role": message.role, "content": message.content}
    for key in ("tool_call_id", "tool_name", "effect_disposition", "token_count", "finish_reason"):
        if getattr(message, key) is not None:
            row[key] = getattr(message, key)
    if message.tool_calls:
        row["tool_calls"] = list(message.tool_calls)
    if message.platform_message_id is not None:
        row["message_id"] = row["platform_message_id"] = message.platform_message_id
    if message.observed:
        row["observed"] = True
    if message.compressed_summary:
        row["_compressed_summary"] = True
    row.update(_identity_keys(message))
    for key in ("api_content", "display_kind", "display_metadata"):
        if getattr(message, key) is not None:
            row[key] = getattr(message, key)
    if message.role == "assistant":
        for key in ("reasoning", "reasoning_content"):
            if getattr(message, key) is not None:
                row[key] = getattr(message, key)
        for key in _SIDECARS:
            if getattr(message, key):
                row[key] = _replayed_sidecar(getattr(message, key))
    return row


def row_stamp(row: _StoredRow) -> RowStamp:
    m = row.message
    return RowStamp(id=MessageId(row.id), timestamp=row.epoch, db_row_snapshot=_row_snapshot(m), message_uid=str(m.message_uid),
                    tool_call_uids=copy.deepcopy(dict(m.tool_call_uids)) if m.tool_call_uids else None, tool_call_uid=m.tool_call_uid)


def _row_outcome(row: _StoredRow, disposition: RowDisposition, canonical: Optional[dict[str, JsonValue]] = None) -> RowOutcome:
    stamp = row_stamp(row)
    return RowOutcome(id=stamp.id, disposition=disposition, timestamp=stamp.timestamp, db_row_snapshot=stamp.db_row_snapshot,
                      canonical_row=copy.deepcopy(canonical), message_uid=stamp.message_uid, tool_call_uids=stamp.tool_call_uids,
                      tool_call_uid=stamp.tool_call_uid)


def _stamp_tool_call_uids(message: NewMessage, index: dict[str, str]) -> NewMessage:
    """Per-occurrence tool-call identity at insert (``SessionMessagesMixin._stamp_tool_call_uids``): an assistant row mints a uid for every call without one; a result row without one takes the uid its batch's assistant row named."""
    if message.role == "assistant" and message.tool_calls:
        uids = dict(message.tool_call_uids or {})
        for call in message.tool_calls:
            call_id = policy.coalesce_tool_call_id(call)
            if call_id and call_id not in uids:
                uids[call_id] = policy.mint_uid()
        policy.index_tool_call_uids(index, {"tool_calls": list(message.tool_calls), "_tool_call_uids": uids})
        return replace(message, tool_call_uids=uids or None)
    if message.role == "user":
        index.clear()
    elif message.role == "tool" and not message.tool_call_uid:
        uid = policy.resolve_tool_call_uid(index, message.tool_call_id)
        if uid:
            return replace(message, tool_call_uid=uid)
    return message


def insert_rows(namespace: dict[str, Any], transcript: _Transcript, messages: Sequence[NewMessage], *,
                keep_reasoning: Optional[bool] = None, now: Optional[float] = None) -> list[_StoredRow]:
    """Insert ``messages`` as fresh active rows (``SessionMessagesMixin._insert_message_rows``): a missing or invalid timestamp takes the clock, kept strictly increasing across the batch; a missing ``message_uid`` is minted; reasoning is kept on assistant rows unless ``keep_reasoning`` says otherwise."""
    now_ts = time.time() if now is None else now
    index: dict[str, str] = {}
    inserted: list[_StoredRow] = []
    for message in messages:
        stamp = policy.coerce_epoch(message.timestamp) if message.timestamp is not None else None
        stamp = now_ts if stamp is None else (0.0 if stamp == 0.0 else stamp)
        keep = message.role == "assistant" if keep_reasoning is None else keep_reasoning
        stored = replace(_payload(message, keep_reasoning=keep), timestamp=stamp, message_uid=message.message_uid or policy.mint_uid())
        row = _StoredRow(id=mint_row_id(namespace), message=_stamp_tool_call_uids(stored, index))
        transcript.rows.append(row)
        inserted.append(row)
        now_ts = max(now_ts, stamp) + 1e-6
    return inserted


def stored_payload(message: NewMessage) -> NewMessage:
    """A row payload as an import or a seeded write stores it: normalized, with a minted ``message_uid`` when it carries none; the caller supplies the timestamp."""
    return replace(_payload(message), message_uid=message.message_uid or policy.mint_uid())


def clone_rows(namespace: dict[str, Any], transcript: _Transcript, originals: Iterable[_StoredRow]) -> list[_StoredRow]:
    """Column clones of ``originals`` as fresh active rows: the payload byte-exact, a new id (``SessionMessagesMixin._clone_message_rows``)."""
    clones = [_StoredRow(id=mint_row_id(namespace), message=row.message) for row in originals]
    transcript.rows.extend(clones)
    return clones


def _display_key(row: _StoredRow) -> tuple[Any, ...]:
    """A row's display identity (``SessionMessagesMixin._display_dedupe_key``): a tool-calling assistant row by its call ids, a user carrier by its live view, every other row by role, content, timestamp and tool columns."""
    m = row.message
    call_ids = tuple(policy.coalesce_tool_call_id(call) for call in m.tool_calls)
    if m.role == "assistant" and call_ids and all(call_ids):
        return ("assistant", None, m.timestamp, m.tool_call_id, call_ids, m.tool_name)
    content: Any = m.content
    if m.role == "user":
        handoff, live = policy.split_user_originated_turn(
            {"role": "user", "content": m.content, "display_kind": m.display_kind, "display_metadata": m.display_metadata})
        if handoff is not None and live is not None:
            content = live.get("content")
    return (m.role, _hashable(content), m.timestamp, m.tool_call_id, _hashable(list(m.tool_calls)) if m.tool_calls else None, m.tool_name)


def _fold_summary_carrier(row: _StoredRow, key: tuple[Any, ...], first_by_uid: dict[str, tuple[Any, ...]]) -> tuple[Any, ...]:
    """A compaction carrier keeps its message's uid but not its content, so it shows as the first row with that uid (``SessionMessagesMixin._fold_summary_carrier``)."""
    uid = row.message.message_uid
    if not uid:
        return key
    if row.message.compressed_summary:
        return first_by_uid.get(uid, key)
    first_by_uid.setdefault(uid, key)
    return key


def _model_only(row: _StoredRow) -> bool:
    return bool((row.message.display_metadata or {}).get(policy.MODEL_ONLY_KEY))


def _rank(row: _StoredRow) -> tuple[bool, int]:
    return (row.visibility is RowVisibility.ACTIVE, row.id)


def _placed(pairs: Sequence[tuple[str, _StoredRow]]) -> list[tuple[int, str, _StoredRow]]:
    """``(slot, sid, row)`` for each row a display read can show, in input order; the slot is the row's identity's first row id in its session, or the slot an in-place prune kept."""
    slots: dict[tuple[str, tuple[Any, ...]], int] = {}
    first_by_uid: dict[str, dict[str, tuple[Any, ...]]] = {}
    placed = []
    for sid, row in pairs:
        if row.visibility is RowVisibility.REWOUND or _model_only(row):
            continue
        key = _fold_summary_carrier(row, _display_key(row), first_by_uid.setdefault(sid, {}))
        placed.append((slots.setdefault((sid, key), row.display_slot if row.display_slot is not None else row.id), sid, row))
    return placed


def _later_tool_copies(placed: Sequence[tuple[int, str, _StoredRow]]) -> set[int]:
    """Ids of the tool rows a display read skips so a compacted tool result paints once: a copy in a later slot than its archived original (a compaction-archived tool row of the same session, uid and timestamp)."""
    def owner(sid: str, row: _StoredRow) -> Optional[tuple[str, str, float]]:
        m = row.message
        return (sid, m.message_uid, m.timestamp) if m.role == "tool" and m.message_uid and m.timestamp is not None else None

    first: dict[tuple[str, str, float], int] = {}
    for slot, sid, row in placed:
        if (key := owner(sid, row)) and row.visibility is RowVisibility.COMPACTION_ARCHIVED:
            first[key] = min(first.get(key, slot), slot)
    return {row.id for slot, sid, row in placed if (key := owner(sid, row)) and first.get(key, slot) < slot}


def display_slots(pairs: Sequence[tuple[str, _StoredRow]]) -> list[tuple[int, tuple[str, _StoredRow]]]:
    """One session's display set as the store's display index serves it: each logical message once in its slot (its identity's first row id, or the slot an in-place prune kept), the live row preferred, then the newest; ``(slot, pair)`` in slot order. A later copy of a compacted tool result is not shown."""
    chosen: dict[int, tuple[str, _StoredRow]] = {}
    placed = _placed(pairs)
    later = _later_tool_copies(placed)
    for slot, sid, row in placed:
        if row.id in later:
            continue
        current = chosen.get(slot)
        if current is None or _rank(row) > _rank(current[1]):
            chosen[slot] = (sid, row)
    return [(slot, chosen[slot]) for slot in sorted(chosen)]


def dedupe_display(pairs: Sequence[tuple[str, _StoredRow]], *, drop_removed: bool = True,
                   first_ids: Optional[dict[int, int]] = None) -> list[tuple[str, _StoredRow]]:
    """Generations collapsed by display identity (``SessionMessagesMixin._dedupe_display_generations``): rows in id order in, each logical message once out (live row preferred, then newest, ordered by its first row). With ``drop_removed`` the input holds every visibility, a message whose newest copy was removed by a rewind or a replacement is left out and a later copy of a compacted tool result is not shown; without it every given row counts. ``first_ids`` gets each kept row's id -> its message's first durable row id, a removed copy's included."""
    seen: dict[tuple[Any, ...], tuple[str, _StoredRow]] = {}
    first_id: dict[tuple[Any, ...], int] = {}
    first_by_uid: dict[str, tuple[Any, ...]] = {}
    removed: dict[Any, bool] = {}
    removed_first: dict[Any, int] = {}
    identities: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    later = _later_tool_copies(_placed(pairs)) if drop_removed else set()
    for sid, row in pairs:
        gone = row.visibility is RowVisibility.REWOUND
        m = row.message
        copies = (("key", _display_key(row)), *(("uid", uid) for uid in (m.message_uid, *m.absorbed_message_uids) if uid))
        if drop_removed:
            removed.update(dict.fromkeys(copies, gone))
            if gone:
                for copy_key in copies:
                    removed_first.setdefault(copy_key, row.id)
                continue
        if _model_only(row) or row.id in later:
            continue
        key = _fold_summary_carrier(row, _display_key(row), first_by_uid)
        current = seen.get(key)
        if current is None or _rank(row) > _rank(current[1]):
            seen[key] = (sid, row)
            identities[key] = copies
        first_id[key] = min(first_id.get(key, row.id), row.id)
    keys = [key for key in sorted(seen, key=first_id.__getitem__) if not (drop_removed and any(removed[copy] for copy in identities[key]))]
    if first_ids is not None:
        first_ids.update((seen[key][1].id, min([first_id[key], *(removed_first[copy_key] for copy_key in identities[key]
                                                               if copy_key in removed_first)])) for key in keys)
    return [seen[key] for key in keys]


def _replay_dict(row: _StoredRow, *, include_row_ids: bool, include_summary_markers: bool, stamp_snapshot: bool) -> dict[str, Any]:
    """One stored row as the replay dict (``SessionMessagesMixin._rows_to_conversation``): born durable, present keys only, reasoning on assistant rows."""
    m = copy.deepcopy(row.message)
    msg: dict[str, Any] = {"role": m.role, "content": policy.loaded_view(m.role, m.content), DB_PERSISTED_MARKER_KEY: True}
    if stamp_snapshot:
        msg["_db_row_snapshot"] = _row_snapshot(m)
    if include_row_ids:
        msg["_row_id"] = row.id
    msg.update(_identity_keys(m))
    msg.update((key, getattr(m, key)) for key in ("api_content", "display_kind") if getattr(m, key))
    if m.display_metadata:
        msg["display_metadata"] = dict(m.display_metadata)
        if m.display_metadata.get("data_use"):
            msg["data_use"] = list(m.display_metadata["data_use"])
    if include_summary_markers and m.compressed_summary:
        msg["_compressed_summary"] = True
    msg.update((key, getattr(m, key)) for key in ("timestamp", "tool_call_id", "tool_name", "effect_disposition") if getattr(m, key))
    if m.tool_calls:
        msg["tool_calls"] = list(m.tool_calls)
    if m.platform_message_id:
        msg["message_id"] = m.platform_message_id
    if m.observed:
        msg["observed"] = True
    if m.role == "assistant":
        msg.update((key, getattr(m, key)) for key in ("finish_reason", "reasoning") if getattr(m, key))
        if m.reasoning_content is not None:
            msg["reasoning_content"] = m.reasoning_content
        msg.update((key, _replayed_sidecar(getattr(m, key))) for key in _SIDECARS if getattr(m, key))
    return msg


def conversation_dicts(rows: Sequence[_StoredRow], *, include_lineage: bool, repair_alternation: bool, include_row_ids: bool,
                       include_summary_markers: bool) -> list[dict[str, Any]]:
    """Rows (id order, already filtered) as the replay projection, after the shared projection policy: a result row without a stored tool-call uid takes the one its assistant row named, a lineage read drops replayed user copies, the review harness and stale markers go, and live replay repairs role alternation."""
    stamp_snapshot = repair_alternation and not include_row_ids
    messages: list[dict[str, Any]] = []
    clones: dict[Any, dict[str, Any]] = {}
    index: dict[str, str] = {}
    owners: list[dict[str, Any]] = []
    for row in rows:
        msg = _replay_dict(row, include_row_ids=include_row_ids, include_summary_markers=include_summary_markers, stamp_snapshot=stamp_snapshot)
        role = row.message.role
        if role == "assistant" and msg.get("tool_calls"):
            owners.append(msg)
        elif role == "user":
            index.clear()
            owners.clear()
        elif role == "tool" and not row.message.tool_call_uid and row.message.tool_call_id:
            for owner in owners:
                policy.index_tool_call_uids(index, owner)
            owners.clear()
            uid = policy.resolve_tool_call_uid(index, row.message.tool_call_id)
            if uid:
                msg["_tool_call_uid"] = uid
        if include_lineage:
            skip, clone_key = policy.dedupe_replayed_user(messages, msg, clones)
            if skip:
                continue
            if clone_key is not None:
                clones[clone_key] = msg
        messages.append(msg)
    messages = policy.strip_stale_tool_call_markers(policy.strip_background_review_harness(messages))
    if repair_alternation and messages:
        policy.repair_message_sequence(messages)
    return messages


def rows_of(backend: InMemoryBackend, scope: ProfileScope, session_ids: Sequence[str]) -> list[tuple[str, _StoredRow]]:
    """Every stored row of ``session_ids`` as ``(session_id, row)`` in id order."""
    transcripts = transcripts_of(backend, scope)
    out = [(sid, row) for sid in dict.fromkeys(session_ids) if sid in transcripts for row in transcripts[sid].rows]
    out.sort(key=lambda pair: pair[1].id)
    return out


def resume_lineage(backend: InMemoryBackend, scope: ProfileScope, session_id: str) -> list[str]:
    """The sessions a display resume reads, root first: the session and its verified compression ancestors, or the session alone for a fork child (``SessionMessagesMixin._resume_lineage_ids``)."""
    table: dict[str, dict[str, Any]] = backend.scope_store(scope).get("sessions", {})
    return list(reversed(backend.sessions._resume_lineage_ids(table, session_id)))


def compression_ancestors(backend: InMemoryBackend, scope: ProfileScope, session_id: str) -> list[str]:
    """The session plus its parents followed only across parents ended by compression (the store's delivery match), cycle-safe."""
    sessions: dict[str, dict[str, Any]] = backend.scope_store(scope).get("sessions", {})
    chain = [session_id]
    current = session_id
    while True:
        parent_id = (sessions.get(current) or {}).get("parent_session_id")
        parent = sessions.get(parent_id) if parent_id else None
        if parent is None or parent.get("end_reason") != "compression" or parent_id in chain:
            return chain
        chain.append(parent_id)
        current = parent_id


def advance_conversation_generation(backend: InMemoryBackend, scope: ProfileScope, session_id: str, end_reason: str) -> None:
    """Advance the peer named by the session row's ``(source, session_key)`` when ``end_reason`` is a reset boundary; the call the session-ending references make in the ending step. The counter never goes down."""
    if end_reason not in RESET_END_REASONS:
        return
    row = backend.scope_store(scope).get("sessions", {}).get(session_id)
    if row is None:
        return
    source = str(row.get("source") or "").strip()
    session_key = str(row.get("session_key") or "").strip()
    if not source or not session_key:
        return
    generations: dict[tuple[str, str], int] = _messages_namespace(backend, scope)["generations"]
    generations[(source, session_key)] = generations.get((source, session_key), 0) + 1


def newest_transcript_timestamp(backend: InMemoryBackend, scope: ProfileScope, session_id: str) -> Optional[datetime]:
    """The newest stored row timestamp of ``session_id`` over every visibility, or None without rows (the transcript arm of a listing's ``last_active``)."""
    transcript = transcripts_of(backend, scope).get(session_id)
    if transcript is None or not transcript.rows:
        return None
    return datetime.fromtimestamp(max(row.epoch for row in transcript.rows), UTC)


def pr_url_content(content: Any) -> bool:
    return isinstance(content, str) and bool(_PR_URL.search(content))
