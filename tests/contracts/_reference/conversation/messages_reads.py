"""Transcript reads of the in-memory message reference: the stored-row and replay projections, resume reads, anchored windows, the prompt timeline and the probes the gateway and CLI guards read."""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional, Sequence

from persistence import ProfileScope, RowVisibility
from persistence.conversation.messages import (
    CONTENT_JSON_PREFIX,
    AnchoredView,
    AnchoredWindow,
    CompressionCooldown,
    ConversationMessage,
    ConversationPeer,
    ConversationRequest,
    ConversationRevision,
    CooldownSnapshot,
    MessageRecord,
    MessageStorageState,
    PromptPage,
    PrUrlHit,
    ResumeConversations,
    ResumeTooLarge,
    SessionTimeline,
    TimelineEntry,
    TranscriptRead,
    TranscriptRow,
    TranscriptView,
    Watermark,
)
from persistence.core.identity import MessageId, SessionId

from .._authority import authority_state
from . import _turn_policy as policy
from .messages_base import MessageBase
from .transcript_store import (
    _display_key,
    _StoredRow,
    _text_length,
    conversation_dicts,
    dedupe_display,
    display_slots,
    pr_url_content,
    resume_lineage,
    rows_of,
)

_IN_VIEW = {
    TranscriptView.ACTIVE: lambda row: row.visibility is RowVisibility.ACTIVE,
    TranscriptView.DISPLAY: lambda row: row.visibility is not RowVisibility.REWOUND,
    TranscriptView.ALL: lambda row: True,
}


def _page(items: list[Any], *, limit: Optional[int], offset: int, latest: bool) -> list[Any]:
    """``limit`` items after skipping ``offset``, from the newest end when ``latest``, always in chronological order."""
    if latest:
        items = items[::-1][offset:]
        return (items if limit is None else items[:limit])[::-1]
    items = items[offset:]
    return items if limit is None else items[:limit]


def _messages(dicts: list[dict[str, Any]]) -> tuple[ConversationMessage, ...]:
    return tuple(ConversationMessage.from_openai_message(message) for message in dicts)


class MessageReadsMixin(MessageBase):
    """Every read verb; none changes a stored row."""

    def _pairs(self, scope: ProfileScope, session_ids: Sequence[str]) -> list[tuple[str, _StoredRow]]:
        return rows_of(self._backend, scope, session_ids)

    def _ids(self, scope: ProfileScope, session_id: SessionId, include_lineage: bool) -> list[str]:
        return resume_lineage(self._backend, scope, str(session_id)) if include_lineage else [str(session_id)]

    def read_transcript(self, session_id: SessionId, request: TranscriptRead = TranscriptRead()) -> tuple[TranscriptRow, ...]:
        scope = self._backend.scope
        ids = self._ids(scope, session_id, request.include_lineage)
        pairs = self._pairs(scope, ids)
        keep = _IN_VIEW[request.view]
        if len(ids) > 1:
            # A merged lineage dedupes generations, then pages (per-session display slots do not compare across segments).
            display = request.view is TranscriptView.DISPLAY
            chosen = dedupe_display(pairs if display else [p for p in pairs if keep(p[1])], drop_removed=display)
        elif request.view is TranscriptView.DISPLAY:
            chosen = [pair for _slot, pair in display_slots(pairs)]
        else:
            chosen = [p for p in pairs if keep(p[1]) and (request.after_id is None or p[1].id > int(request.after_id))]
        page = _page(chosen, limit=request.limit, offset=request.offset, latest=request.latest)
        return tuple(row.as_transcript_row(sid) for sid, row in page)

    def active_message_ids(self, session_id: SessionId) -> tuple[MessageId, ...]:
        scope = self._backend.scope
        transcript = self._transcript(scope, session_id)
        return tuple(MessageId(row.id) for row in transcript.active()) if transcript else ()

    def active_watermark(self, session_id: SessionId) -> Watermark:
        ids = self.active_message_ids(session_id)
        return Watermark(last_active=ids[-1] if ids else None)

    def count_all(self) -> int:
        # A deleted session's transcript goes with it: the sessions reference retires the id and leaves this namespace alone.
        scope = self._backend.scope
        retired = authority_state(self._backend, scope).snapshot["retired"]
        appended = sum(len(t.rows) for sid, t in self._ns(scope)["transcripts"].items() if sid not in retired)
        imported = sum(len(rows) for sid, rows in self._backend.scope_store(scope).get("transcripts", {}).items() if sid not in retired)
        return appended + imported

    def display_count(self, session_id: SessionId) -> int:
        scope = self._backend.scope
        return len(display_slots(self._pairs(scope, [str(session_id)])))

    def conversation_revision(self, session_id: SessionId, *, include_lineage: bool = False) -> ConversationRevision:
        scope = self._backend.scope
        ids = self._ids(scope, session_id, include_lineage)
        active = [row.id for _sid, row in self._pairs(scope, ids) if row.visibility is RowVisibility.ACTIVE]
        sessions = self._sessions(scope)
        revision = sum(int(sessions[sid].get("transcript_revision", 0)) for sid in dict.fromkeys(ids) if sid in sessions)
        return ConversationRevision(lineage_ids=tuple(SessionId(sid) for sid in ids), active_count=len(active),
                                    max_active_id=max(active, default=0), transcript_revision=revision)

    def _conversation_rows(self, scope: ProfileScope, session_id: SessionId, view: TranscriptView, include_lineage: bool) -> list[_StoredRow]:
        pairs = self._pairs(scope, self._ids(scope, session_id, include_lineage))
        if view is TranscriptView.DISPLAY:
            return [row for _sid, row in dedupe_display(pairs)]
        return [row for _sid, row in pairs if _IN_VIEW[view](row)]

    def read_conversation(self, session_id: SessionId, request: ConversationRequest = ConversationRequest()) -> tuple[ConversationMessage, ...]:
        scope = self._backend.scope
        rows = self._conversation_rows(scope, session_id, request.view, request.include_lineage)
        return _messages(conversation_dicts(rows, include_lineage=request.include_lineage, repair_alternation=request.repair_alternation,
                                            include_row_ids=request.include_row_ids, include_summary_markers=request.repair_alternation))

    def read_resume_conversations(self, session_id: SessionId) -> ResumeConversations:
        scope = self._backend.scope
        pairs = self._pairs(scope, resume_lineage(self._backend, scope, str(session_id)))
        # The model projection is the tip's working context: active rows, without the gateway's session_meta bookkeeping row.
        model = [row for sid, row in pairs if sid == str(session_id) and row.visibility is RowVisibility.ACTIVE and row.message.role != "session_meta"]
        display = [row for _sid, row in dedupe_display(pairs)]
        return ResumeConversations(
            model=_messages(conversation_dicts(model, include_lineage=False, repair_alternation=True, include_row_ids=True, include_summary_markers=True)),
            display=_messages(conversation_dicts(display, include_lineage=True, repair_alternation=False, include_row_ids=True, include_summary_markers=False)))

    def assert_resume_safe(self, session_id: SessionId, *, max_messages: int, tip_only: bool = False) -> int:
        scope = self._backend.scope
        if max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        if max_messages == 0:
            return 0
        ids, view = ([str(session_id)], TranscriptView.ACTIVE) if tip_only else (resume_lineage(self._backend, scope, str(session_id)), TranscriptView.DISPLAY)
        count = 0
        for _sid, row in self._pairs(scope, ids):
            if _IN_VIEW[view](row):
                count += 1
                if count > max_messages:
                    raise ResumeTooLarge(count, max_messages, scope="in its tip segment" if tip_only else "across its lineage")
        return count

    def read_around(self, session_id: SessionId, anchor: MessageId, *, window: int = 5) -> AnchoredWindow:
        scope = self._backend.scope
        window = max(window, 0)
        transcript = self._transcript(scope, session_id)
        rows = transcript.rows if transcript else []
        if transcript is None or transcript.find(int(anchor)) is None:
            return AnchoredWindow(window=(), before=0, after=0)
        before_rows = [r for r in rows if r.id <= int(anchor)][-(window + 1):]
        after_rows = [r for r in rows if r.id > int(anchor)][:window]
        return AnchoredWindow(window=tuple(r.as_record() for r in (*before_rows, *after_rows)), before=len(before_rows) - 1, after=len(after_rows))

    def read_anchored_view(self, session_id: SessionId, anchor: MessageId, *, window: int = 5, bookend: int = 3,
                           keep_roles: Optional[tuple[str, ...]] = ("user", "assistant")) -> AnchoredView:
        scope = self._backend.scope
        bookend = max(bookend, 0)
        primitive = self.read_around(session_id, anchor, window=window)
        if not primitive.window:
            return AnchoredView(window=(), before=0, after=0, bookend_start=(), bookend_end=())
        kept = primitive.window if keep_roles is None else tuple(m for m in primitive.window if int(m.id) == int(anchor) or m.role in keep_roles)
        start: tuple[MessageRecord, ...] = ()
        end: tuple[MessageRecord, ...] = ()
        if bookend > 0:
            transcript = self._transcript(scope, session_id)
            rows = transcript.rows if transcript else []
            eligible = [r for r in rows if (keep_roles is None or r.message.role in keep_roles) and _text_length(r.message.content) > 0]
            first_id, last_id = int(primitive.window[0].id), int(primitive.window[-1].id)
            start = tuple(r.as_record() for r in [r for r in eligible if r.id < first_id][:bookend])
            end = tuple(r.as_record() for r in [r for r in eligible if r.id > last_id][-bookend:])
        return AnchoredView(window=kept, before=primitive.before, after=primitive.after, bookend_start=start, bookend_end=end)

    def find_pr_url_hits(self, session_ids: Sequence[SessionId]) -> tuple[PrUrlHit, ...]:
        scope = self._backend.scope
        ids = [str(s) for s in session_ids if s]
        return tuple(PrUrlHit(session_id=SessionId(sid), content=row.message.content) for sid, row in self._pairs(scope, ids)
                     if row.message.role == "tool" and pr_url_content(row.message.content))

    def latest_todo_result(self, session_id: SessionId, *, max_chars: int) -> Optional[str]:
        scope = self._backend.scope
        if not session_id:
            return None
        lineage = resume_lineage(self._backend, scope, str(session_id))
        removed = _removed_todo_copies(self._pairs(scope, lineage))
        for sid in reversed(lineage):
            rows = [row for _sid, row in self._pairs(scope, [sid]) if row.visibility is not RowVisibility.REWOUND]
            for index in range(len(rows) - 1, -1, -1):
                if any(removed.get(copy, 0) > rows[index].id for copy in _todo_copies(rows[index])):
                    continue
                content = _todo_content(rows, index, max_chars)
                if content is not None:
                    return content
        return None

    def _timeline_slots(self, scope: ProfileScope, session_id: SessionId) -> list[tuple[int, _StoredRow, str]]:
        """The session's displayed human prompts: ``(slot, representative row, preview)`` in display order."""
        out = []
        for slot, (_sid, row) in display_slots(self._pairs(scope, [str(session_id)])):
            if row.message.role != "user":
                continue
            preview = policy.prompt_preview(row.message.content, row.message.display_kind, row.message.compressed_summary)
            if preview:
                out.append((slot, row, preview))
        return out

    def read_session_timeline(self, session_id: SessionId, *, limit: int = 500, after_row_id: int = 0) -> SessionTimeline:
        scope = self._backend.scope
        prompts = self._timeline_slots(scope, session_id)
        following = [entry for entry in prompts if entry[0] > after_row_id][:limit + 1]
        has_more = len(following) > limit
        page = following[:limit]
        return SessionTimeline(
            entries=tuple(TimelineEntry(row_id=MessageId(row.id), preview=preview, timestamp=row.epoch) for _slot, row, preview in page),
            limit=limit, after_row_id=after_row_id, returned=len(page), total=len(prompts), has_more=has_more,
            next_cursor=page[-1][0] if has_more else None)

    def read_prompt_page(self, session_id: SessionId, row_id: MessageId, *, limit: int = 120) -> Optional[PromptPage]:
        scope = self._backend.scope
        anchor = self._row_in_session(scope, session_id, row_id)
        if anchor is None or anchor.message.role != "user" or anchor.visibility is RowVisibility.REWOUND:
            return None
        if not policy.prompt_preview(anchor.message.content, anchor.message.display_kind, anchor.message.compressed_summary):
            return None
        slots = display_slots(self._pairs(scope, [str(session_id)]))
        start = next((slot for slot, (_sid, row) in slots if row.id == int(row_id)), None)
        if start is None:
            return None
        offset = sum(1 for slot, _pair in slots if slot < start)
        page = [pair for slot, pair in slots if slot >= start][:limit]
        return PromptPage(messages=tuple(row.as_transcript_row(sid) for sid, row in page), row_id=row_id, limit=limit, returned=len(page),
                          offset=offset, total=len(slots), has_older=offset > 0, has_newer=offset + len(page) < len(slots))

    def latest_conversation_role(self, session_id: SessionId) -> Optional[str]:
        scope = self._backend.scope
        transcript = self._transcript(scope, session_id)
        return next((row.message.role for row in reversed(transcript.active()) if row.message.role not in ("session_meta", "system")), None) if transcript else None

    def latest_dialogue_message(self, session_id: SessionId) -> Optional[str]:
        scope = self._backend.scope
        transcript = self._transcript(scope, session_id)
        for row in reversed(transcript.active() if transcript else []):
            # The store's predicate: TRIM strips spaces only.
            if row.message.role in ("user", "assistant") and _has_text(row.message.content):
                return _content_text(row.message.content)
        return None

    def message_storage_state(self, message_id: MessageId) -> Optional[MessageStorageState]:
        scope = self._backend.scope
        for session_id, transcript in self._ns(scope)["transcripts"].items():
            row = transcript.find(int(message_id))
            if row is not None:
                return MessageStorageState(session_id=SessionId(session_id), visibility=row.visibility)
        return None

    def latest_row_id(self, session_id: SessionId, *, role: str = "user", offset: int = 0,
                      require_text: bool = True) -> Optional[MessageId]:
        scope = self._backend.scope
        transcript = self._transcript(scope, session_id)
        if transcript is None or not session_id or role not in ("user", "assistant") or offset < 0:
            return None
        candidates = [row for row in reversed(transcript.active()) if row.message.role == role and (not require_text or _has_text(row.message.content))]
        return MessageId(candidates[offset].id) if offset < len(candidates) else None

    def message_role(self, session_id: SessionId, message_id: MessageId) -> Optional[str]:
        scope = self._backend.scope
        row = self._row_in_session(scope, session_id, message_id) if session_id else None
        return row.message.role if row is not None and row.visibility is RowVisibility.ACTIVE else None

    def resolve_active_row_id(self, session_id: SessionId, message_id: MessageId) -> Optional[MessageId]:
        scope = self._backend.scope
        transcript = self._transcript(scope, session_id)
        origin = transcript.find(int(message_id)) if transcript is not None and session_id else None
        if transcript is None or origin is None:
            return None
        if origin.visibility is RowVisibility.ACTIVE:
            return message_id
        o = origin.message
        clones = [row.id for row in transcript.active() if row.id > origin.id and row.message.role == o.role
                  and row.message.content == o.content and row.message.timestamp == o.timestamp]
        return MessageId(clones[0]) if len(clones) == 1 else None

    def has_platform_message_id(self, session_id: SessionId, platform_message_id: str) -> bool:
        scope = self._backend.scope
        transcript = self._transcript(scope, session_id)
        return transcript is not None and any(row.message.platform_message_id == platform_message_id for row in transcript.rows)

    def has_gateway_input_owner(self, session_id: SessionId, owner: str) -> bool:
        scope = self._backend.scope
        transcript = self._transcript(scope, session_id)
        return transcript is not None and any(
            row.message.role == "user" and not row.message.observed and row.visibility is not RowVisibility.REWOUND
            and (row.message.display_metadata or {}).get("gateway_input_owner") == owner
            for row in transcript.rows)

    def latest_conversation_boundary(self, peer: ConversationPeer) -> Optional[int]:
        scope = self._backend.scope
        if not peer.session_key or not peer.source:
            return None
        generation = int(self._ns(scope)["generations"].get((peer.source, peer.session_key), 0))
        return generation if generation > 0 else None

    def read_compression_cooldown(self, session_id: SessionId, at: float) -> Optional[CompressionCooldown]:
        scope = self._backend.scope
        record = self._ns(scope)["cooldowns"].get(str(session_id))
        if record is None or record.until is None or record.until <= at or str(session_id) not in self._sessions(scope):
            return None
        return CompressionCooldown(until=record.until, error=record.error)

    def read_compression_cooldown_snapshot(self, session_id: SessionId) -> CooldownSnapshot:
        scope = self._backend.scope
        if not session_id or str(session_id) not in self._sessions(scope):
            return CooldownSnapshot(present=False)
        record = self._ns(scope)["cooldowns"].get(str(session_id))
        return CooldownSnapshot(present=True, until=None if record is None else record.until, error=None if record is None else record.error)

    def terminal_tool_call_payloads(self, *, since_ts: float) -> Iterator[str]:
        scope = self._backend.scope
        rows = sorted((row for transcript in self._ns(scope)["transcripts"].values() for row in transcript.rows
                       if row.message.role == "assistant" and row.message.tool_calls and row.epoch >= since_ts), key=lambda row: row.id)
        payloads = [json.dumps(list(row.message.tool_calls)) for row in rows]
        # The store's ``tool_calls LIKE '%terminal%'``: ASCII-case-insensitive, over the whole stored payload.
        return iter([payload for payload in payloads if "terminal" in payload.lower()])

    def blocked_tool_result_payloads(self, *, since_ts: float) -> Iterator[tuple[str, Optional[str]]]:
        scope = self._backend.scope
        # Only text content: the store keeps structured content behind a NUL sentinel, where its LIKE stops.
        hits = [(row.message.tool_call_id, row.message.content)
                for transcript in self._ns(scope)["transcripts"].values() for row in transcript.rows
                if row.message.role == "tool" and row.message.tool_call_id is not None and row.epoch >= since_ts
                and isinstance(row.message.content, str)
                and ("blocked" in row.message.content.lower() or "approval" in row.message.content.lower())]
        return iter(hits)

    def has_any_message(self) -> bool:
        scope = self._backend.scope
        return any(transcript.rows for transcript in self._ns(scope)["transcripts"].values())

    def spillover_reference_rows(self) -> tuple[tuple[Optional[str], Optional[str]], ...]:
        scope = self._backend.scope
        rows = sorted((row for transcript in self._ns(scope)["transcripts"].values() for row in transcript.rows), key=lambda row: row.id)
        hits = [(_content_text(row.message.content), row.message.api_content) for row in rows
                if "spillover" in (_content_text(row.message.content) or "") or "spillover" in (row.message.api_content or "")]
        return tuple(hits[:_SPILLOVER_ROW_CAP])


# The literal row cap on the spillover reference scan.
_SPILLOVER_ROW_CAP = 20001


def _has_text(content: Any) -> bool:
    """The store's ``content IS NOT NULL AND TRIM(content) != ''`` (spaces only)."""
    return content is not None and (_content_text(content) or "").strip(" ") != ""


def _content_text(content: Any) -> Optional[str]:
    """The stored content column: text as is, structured content as its sentinel-prefixed JSON."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, (list, dict)):
        return CONTENT_JSON_PREFIX + json.dumps(content)
    return str(content)


def _is_todo_call(call: Any, call_id: str) -> bool:
    """A todo tool call (direct, or batched inside ``tool_call``) with this id (``SessionTodoResultMixin.get_latest_todo_result``)."""
    if not isinstance(call, dict) or call.get("id") != call_id:
        return False
    function = call.get("function")
    if not isinstance(function, dict):
        return False
    if function.get("name") in ("todo_list", "todo"):
        return True
    if function.get("name") != "tool_call":
        return False
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    inner = arguments.get("calls") if isinstance(arguments, dict) else None
    return isinstance(inner, list) and any(isinstance(item, dict) and item.get("name") in ("todo_list", "todo") for item in inner)


def _todo_copies(row: _StoredRow) -> tuple[Any, ...]:
    """The copy identities a display read gives a row: its display identity, its uid and the uids it absorbed."""
    m = row.message
    return (("key", _display_key(row)), *(("uid", uid) for uid in (m.message_uid, *m.absorbed_message_uids) if uid))


def _removed_todo_copies(pairs: Sequence[tuple[str, _StoredRow]]) -> dict[Any, int]:
    """Copy identity to the newest rewound Todo result row id among ``pairs`` (``SessionTodoResultMixin._removed_todo_copies``): a result with a newer removed copy is hidden from display, so the resume must not restore it."""
    removed: dict[Any, int] = {}
    for _sid, row in pairs:
        if row.visibility is RowVisibility.REWOUND and row.message.role == "tool" and row.message.tool_name in ("todo_list", "todo"):
            for copy in _todo_copies(row):
                removed[copy] = max(removed.get(copy, 0), row.id)
    return removed


def _todo_content(rows: Sequence[_StoredRow], index: int, max_chars: int) -> Optional[str]:
    """The row's content when it is a valid todo result paired with the assistant call right before it, else None."""
    message = rows[index].message
    content = message.content
    if message.role != "tool" or message.tool_name not in ("todo_list", "todo") or not message.tool_call_id:
        return None
    if _text_length(content) > max_chars or not isinstance(content, str):
        return None
    prior = next((row.message for row in reversed(rows[:index]) if row.message.role in ("assistant", "user", "system")), None)
    if prior is None or prior.role != "assistant":
        return None
    try:
        if not any(_is_todo_call(call, message.tool_call_id) for call in prior.tool_calls):
            return None
        data = json.loads(content)
        if not isinstance(data, dict) or not isinstance(data.get("todos"), list):
            return None
        int(data.get("revision") or 0)
    except (ValueError, TypeError):
        return None
    return content
