"""Stored-session reads of the in-memory message reference: one read generation, the display-index pages with their durable keys, the compaction handoff lookups, the poller's cursor state and the tool calls a page's results name."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, ContextManager, Iterator, Mapping, Optional, Sequence

from persistence import RowVisibility
from persistence.conversation.messages import (
    ConversationMessage, PagedMessage, ReadCheck, SegmentPage, TranscriptRow,
)
from persistence.core.identity import JsonValue, MessageId, SessionId

from .messages_base import MessageBase
from .transcript_store import (
    _display_key, _placed, _StoredRow, conversation_dicts, dedupe_display, display_slots, rows_of,
)

# The text markers a stored row that can hold a compaction handoff carries (the store's ``instr`` probes).
_HANDOFF_MARKERS = ("[CONTEXT COMPACTION", "[CONTEXT SUMMARY]")


def _stored_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _holds_handoff(row: _StoredRow) -> bool:
    m = row.message
    return m.role in ("user", "assistant") and (
        m.compressed_summary or any(marker in _stored_text(m.content) for marker in _HANDOFF_MARKERS))


def _paged(message: dict[str, Any], position: int, durable: int) -> PagedMessage:
    return PagedMessage(message=ConversationMessage.from_openai_message(message), position=position,
                        durable_id=MessageId(durable))


class StoredReadsMixin(MessageBase):
    """The reads ``session.transcript`` and the picker make without attaching; none changes a stored row."""

    def read_snapshot(self) -> ContextManager[ReadCheck]:
        return self._snapshot()

    @contextmanager
    def _snapshot(self) -> Iterator[ReadCheck]:
        # No read transaction to pin: every read is its own, checked against the scope's state at the start.
        scope = self._backend.scope
        start = repr(self._backend.scope_store(scope))
        yield ReadCheck(lambda: repr(self._backend.scope_store(scope)) == start, pinned=False)

    def display_index_complete(self, session_id: SessionId) -> bool:
        return True

    def ensure_display_index(self, session_id: SessionId) -> bool:
        return True

    def display_generations(self, session_id: SessionId, lineage_ids: Sequence[SessionId]) -> tuple[PagedMessage, ...]:
        first_ids: dict[int, int] = {}
        rows = [row for _sid, row in dedupe_display(rows_of(self._backend, self._backend.scope, [str(s) for s in lineage_ids]),
                                                    first_ids=first_ids)]
        messages = conversation_dicts(rows, include_lineage=True, repair_alternation=False, include_row_ids=True,
                                      include_summary_markers=False)
        return tuple(_paged(m, first_ids.get(m["_row_id"], m["_row_id"]), first_ids.get(m["_row_id"], m["_row_id"]))
                     for m in messages)

    def _durable_ids(self, pairs: Sequence[tuple[str, _StoredRow]], slots: Sequence[tuple[int, tuple[str, _StoredRow]]]) -> dict[int, int]:
        """``{row id: first durable id}`` of the shown rows: the slot, a removed original with the same identity, and for a summary carrier the first removed row with its uid."""
        by_key: dict[Any, int] = {}
        for _sid, row in pairs:
            if row.visibility is RowVisibility.REWOUND:
                by_key.setdefault(("key", _display_key(row)), row.id)
                if row.message.message_uid:
                    by_key.setdefault(("uid", row.message.message_uid), row.id)
        out = {}
        for slot, (_sid, row) in slots:
            uid = row.message.message_uid
            carrier = by_key.get(("uid", uid), row.id) if row.message.compressed_summary and uid else row.id
            out[row.id] = min(slot, by_key.get(("key", _display_key(row)), row.id), carrier)
        return out

    def _cursor_slot(self, pairs: Sequence[tuple[str, _StoredRow]], durable: int) -> Optional[int]:
        row = next((row for _sid, row in pairs if row.id == durable), None)
        if row is None:
            return None
        shown = {r.id: slot for slot, _sid, r in _placed(pairs)}
        key = _display_key(row)
        copies = [r for _sid, r in pairs if r.id in shown and _display_key(r) == key]
        if copies:
            return shown[max(copies, key=lambda r: (r.visibility is RowVisibility.ACTIVE, r.id)).id]
        uid = row.message.message_uid
        carriers = [r for _sid, r in pairs if uid and r.id in shown and r.visibility is RowVisibility.ACTIVE
                    and r.message.compressed_summary and r.message.message_uid == uid]
        if carriers:
            return shown[max(r.id for r in carriers)]
        return shown.get(row.id)

    def transcript_segment_page(self, session_id: SessionId, *, latest: bool, after: Optional[tuple[int, int]], count: int,
                                inclusive: bool = False) -> SegmentPage:
        pairs = rows_of(self._backend, self._backend.scope, [str(session_id)])
        slots = display_slots(pairs)
        position = None
        if after is not None:
            found = self._cursor_slot(pairs, int(after[1]))
            position = int(after[0]) if found is None else found
            if inclusive:
                position += 1 if latest else -1
        past = [item for item in slots if position is None or (item[0] < position if latest else item[0] > position)]
        raw = past[::-1][:count][::-1] if latest else past[:count]
        durable = self._durable_ids(pairs, slots)
        before = [item for item in slots if raw and item[0] < raw[0][0]][-1:]
        keys = {pair[1].id: (slot, durable[pair[1].id]) for slot, pair in raw}
        context = {pair[1].id for _slot, pair in before}
        messages = [m for m in conversation_dicts([pair[1] for _slot, pair in (*before, *raw)], include_lineage=False,
                                                  repair_alternation=False, include_row_ids=True, include_summary_markers=False)
                    if m["_row_id"] not in context]
        edge = None if not raw else keys[raw[0][1][1].id if latest else raw[-1][1][1].id]
        return SegmentPage(messages=tuple(_paged(m, *keys[m["_row_id"]]) for m in messages), full=len(raw) == count,
                           edge=edge, start=raw[0][0] if raw else None)

    def transcript_handoff_rows(self, shown_ids: Sequence[MessageId], segment_ids: Sequence[SessionId]) -> Mapping[MessageId, tuple[TranscriptRow, ...]]:
        if not shown_ids:
            return {}
        segments = {str(s) for s in segment_ids}
        every = rows_of(self._backend, self._backend.scope, list(self._sessions(self._backend.scope)))
        by_id = {row.id: (sid, row) for sid, row in every}
        found: dict[MessageId, tuple[TranscriptRow, ...]] = {}
        for shown in sorted({int(i) for i in shown_ids}):
            if shown not in by_id:
                continue
            sid, row = by_id[shown]
            uid = row.message.message_uid
            held = [(s, r) for s, r in every if uid and s in segments and r.message.message_uid == uid and r.id < shown
                    and _holds_handoff(r)]
            if sid in segments and _holds_handoff(row):
                held.append((sid, row))
            if held:
                found[MessageId(shown)] = tuple(r.as_transcript_row(s) for s, r in held)
        return found

    def transcript_handoffs_shown_before(self, session_id: SessionId, before: int, handoffs: Sequence[tuple[Optional[str], JsonValue]]) -> tuple[TranscriptRow, ...]:
        uids = {uid for uid, _ in handoffs if uid}
        contents = [content for uid, content in handoffs if not uid and isinstance(content, (str, list, dict))]
        pairs = rows_of(self._backend, self._backend.scope, [str(session_id)])
        chosen = {pair[1].id: slot for slot, pair in display_slots(pairs)}
        out = []
        for sid, g in pairs:
            m = g.message
            if not _holds_handoff(g) or not ((m.message_uid in uids) or (not m.message_uid and m.content in contents)):
                continue
            if any(chosen[w.id] < before for _s, w in pairs if w.id in chosen
                   and (w.id == g.id or (m.message_uid and w.message.message_uid == m.message_uid and w.id > g.id))):
                out.append(g.as_transcript_row(sid))
        return tuple(out)

    def transcript_tool_calls_before(self, segment_ids: Sequence[SessionId], before_id: MessageId, *, limit: int = 16) -> tuple[tuple[Optional[str], Optional[str]], ...]:
        pairs = rows_of(self._backend, self._backend.scope, [str(s) for s in segment_ids])
        found = [row.as_transcript_row(sid) for sid, row in pairs
                 if row.message.role == "assistant" and row.message.tool_calls and row.id < int(before_id)]
        return tuple((json.dumps(row.tool_calls), row.tool_call_uids) for row in found[::-1][:limit])

    def transcript_cursor_state(self, session_id: SessionId, through_id: MessageId, lineage_ids: Sequence[SessionId]) -> tuple[int, int, int]:
        scope = self._backend.scope
        ids = [str(s) for s in lineage_ids] or [str(session_id)]
        pairs = rows_of(self._backend, scope, ids)
        live = sum(1 for sid, row in pairs if row.visibility is RowVisibility.ACTIVE and (sid != session_id or row.id <= int(through_id)))
        inactive = sum(1 for _sid, row in pairs if row.visibility is not RowVisibility.ACTIVE)
        sessions = self._sessions(scope)
        revision = sum(int(sessions[sid].get("transcript_revision", 0)) for sid in dict.fromkeys(ids) if sid in sessions)
        return live, inactive, revision
