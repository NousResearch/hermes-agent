"""Contract suite for the transcript reads of ``MessageRepository``.

Backend-parametrized cases exercise the ten transcript read verbs (``read_conversation``, ``read_resume_conversations``, ``assert_resume_safe``, ``read_around``, ``read_anchored_view``, ``find_pr_url_hits``, ``latest_conversation_role``, ``has_platform_message_id``, ``has_gateway_input_owner``, ``latest_conversation_boundary``). Record checks, reference refusal injection, and SQLite read-only paging also have backend-independent or backend-specific cases.

The common scripts seed through public protocol operations (``append``, ``deactivate_message``, ``compact``, ``SessionRepository.create``/``end``); the counter-read case calls a reference-only helper and the reference refusal case replaces methods. Whole-projection checks use exact sequence equality.
"""

from __future__ import annotations

import time
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

import pytest

from persistence import RowVisibility, ProfileScope
from persistence.core.identity import MessageId, SessionId
from persistence.conversation.messages import (
    DB_PERSISTED_MARKER_KEY,
    ConversationMessage,
    ConversationPeer,
    ConversationRequest,
    NewMessage,
    ResumeTooLarge,
    TranscriptRead,
    TranscriptView,
)
from persistence.conversation.sessions import NewSession, SessionLineageKind

SESSION = SessionId("reads-s1")
OTHER = SessionId("reads-s2")
CHILD = SessionId("reads-child")
T0 = datetime(2026, 9, 11, 10, 0, 0, tzinfo=UTC)


def ts(i: int) -> float:
    return (T0 + timedelta(seconds=i)).timestamp()


def new(role: str, content: Any, i: int, **kwargs: Any) -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=ts(i), **kwargs)


class Totals(NamedTuple):
    """The transcript counters the session row carries."""

    message_count: int
    tool_call_count: int
    rewind_count: int


def totals(backend, scope: ProfileScope, sid: SessionId) -> Totals:
    """The session row's transcript counters, all zero for an absent session."""
    store = backend.open(scope)
    record = store.sessions.get(sid)
    return Totals(0, 0, 0) if record is None else Totals(record.message_count, record.tool_call_count, record.rewind_count)


def rewind_from(repo, sid: SessionId, target: MessageId) -> None:
    """Mark ``target`` and every later active row of the session rewound, the visibility a user-turn rewind leaves behind."""
    for row_id in repo.active_message_ids(sid):
        if row_id >= target:
            repo.deactivate_message(sid, row_id)


def dumps(messages: Iterable[Any]) -> str:
    return json.dumps([m.as_openai_message() if hasattr(m, "as_openai_message") else m for m in messages], sort_keys=True, ensure_ascii=False)


# ── fixture script: one description, replayed onto a backend ──


class Script:
    """A seeding script replayed onto a backend through its public verbs."""

    def __init__(self) -> None:
        self.ops: list[tuple] = []

    def session(self, sid: SessionId, source: str = "cli", parent: Optional[SessionId] = None, *,
                branch: bool = False, session_key: Optional[str] = None) -> Script:
        self.ops.append(("session", sid, source, parent, branch, session_key)); return self

    def msg(self, sid: SessionId, role: str, content: Any, i: int, **kwargs: Any) -> Script:
        self.ops.append(("msg", sid, role, content, i, kwargs)); return self

    def rewind(self, sid: SessionId, ordinal: int) -> Script:
        """Rewind to the ordinal-th appended row of the whole script (0-based)."""
        self.ops.append(("rewind", sid, ordinal)); return self

    def compact(self, sid: SessionId, replacement: list[NewMessage]) -> Script:
        self.ops.append(("compact", sid, replacement)); return self

    def end(self, sid: SessionId, reason: str) -> Script:
        self.ops.append(("end", sid, reason)); return self

    def apply(self, backend, scope: ProfileScope) -> list[MessageId]:
        store = backend.open(scope)
        ids: list[MessageId] = []
        steps = {
            "session": lambda sid, source, parent, branch, key: store.sessions.create(NewSession(
                session_id=sid, source=source, parent_session_id=parent,
                lineage_kind=SessionLineageKind.BRANCH if branch else SessionLineageKind.NONE, session_key=key)),
            "msg": lambda sid, role, content, i, kwargs: ids.append(store.messages.append(sid, new(role, content, i, **kwargs))),
            "rewind": lambda sid, ordinal: rewind_from(backend.open(scope).messages, sid, ids[ordinal]),
            "compact": lambda sid, replacement: store.messages.compact(sid, replacement),
            "end": lambda sid, reason: store.sessions.end(sid, reason),
        }
        for kind, *args in self.ops:
            steps[kind](*args)
        return ids



def projection_script() -> Script:
    """The replay fixture list: leading assistant, consecutive user rows, an orphan tool row, api_content beside content, a rewound tail, sidecar columns."""
    return (Script().session(SESSION)
            .msg(SESSION, "assistant", "leading assistant", 0, finish_reason="stop", reasoning="why",
                 reasoning_details=[{"type": "text", "text": "r"}], codex_message_items=[{"id": "m1"}])
            .msg(SESSION, "user", "first ask", 1, api_content="first ask [exact api bytes]", platform_message_id="tg-1",
                 display_metadata={"gateway_input_owner": "peer-A"})
            .msg(SESSION, "user", "second ask in a row", 2)
            .msg(SESSION, "tool", "orphan tool result", 3, tool_call_id="call-orphan", tool_name="grep")
            .msg(SESSION, "assistant", "reply", 4, tool_calls=({"id": "call-1", "type": "function", "function": {"name": "f", "arguments": "{}"}},))
            .msg(SESSION, "tool", "answer for call-1", 5, tool_call_id="call-1", tool_name="f")
            .msg(SESSION, "assistant", "final", 6, reasoning_content="", display_kind="final",
                 reasoning_details="[]", codex_reasoning_items='[{"id": "r1"}]', codex_message_items="")
            .msg(SESSION, "user", [{"type": "text", "text": "multimodal"}, {"type": "image_url", "image_url": {"url": "data:,x"}}], 7)
            .msg(SESSION, "assistant", "rewound reply", 8)
            .rewind(SESSION, 7))


def lineage_script() -> Script:
    return (Script().session(SESSION).session(CHILD, parent=SESSION)
            .msg(SESSION, "user", "root u1", 0).msg(SESSION, "assistant", "root a1", 1).end(SESSION, "compression")
            .msg(CHILD, "user", "root a1 follow-up", 2)
            .msg(CHILD, "user", "child u2", 3)
            .msg(CHILD, "assistant", "child a2", 4)
            .msg(CHILD, "assistant", {"a": 1, "b": 2}, 5)
            .msg(CHILD, "assistant", {"b": 2, "a": 1}, 5)
            .compact(CHILD, [new("user", "SUMMARY", 10, compressed_summary=True), new("user", "child u2", 3), new("assistant", "child a2", 4),
                             new("assistant", {"b": 2, "a": 1}, 5)]))



# ── read_conversation ────────────────────────────────────────────────────────



def test_read_conversation_projection_policy(backend, scope_a):
    """CONTRACT: active rows only, born-durable marker on every dict, api_content verbatim, present-only sidecars, no row ids without opt-in; the repair heals the LIST and leaves the rows."""
    projection_script().apply(backend, scope_a)
    repo = backend.open(scope_a).messages
    out = repo.read_conversation(SESSION)
    assert [m.content for m in out] == [
        "leading assistant", "first ask", "second ask in a row", "orphan tool result", "reply", "answer for call-1", "final"]
    wire = [m.as_openai_message() for m in out]
    assert all(w[DB_PERSISTED_MARKER_KEY] is True and "_row_id" not in w for w in wire)
    assert (wire[1]["api_content"], wire[1]["message_id"], wire[1]["display_metadata"]) == (
        "first ask [exact api bytes]", "tg-1", {"gateway_input_owner": "peer-A"})
    assert wire[0]["reasoning_details"] == [{"type": "text", "text": "r"}] and wire[0]["codex_message_items"] == [{"id": "m1"}]
    assert wire[6]["reasoning_content"] == "" and "reasoning_content" not in wire[0]
    # Serialized sidecars replay decoded; a stored "[]" is present as []; an empty string is absent (stored NULL).
    assert (wire[6]["reasoning_details"], wire[6]["codex_reasoning_items"]) == ([], [{"id": "r1"}])
    assert "codex_message_items" not in wire[6]
    everything = repo.read_conversation(SESSION, ConversationRequest(view=TranscriptView.ALL))
    assert everything[7].content == [{"type": "text", "text": "multimodal"}, {"type": "image_url", "image_url": {"url": "data:,x"}}]
    repaired = repo.read_conversation(SESSION, ConversationRequest(repair_alternation=True))
    assert [m.role for m in repaired] == ["assistant", "user", "assistant", "tool", "assistant"]
    assert repaired[1].content == "first ask\n\nsecond ask in a row" and repaired[1].api_content is None
    assert [m.role for m in repo.read_conversation(SESSION)][1:3] == ["user", "user"]
    assert repo.read_conversation(SessionId("never")) == ()


def test_read_conversation_lineage_walk_display_dedupe_and_branch_isolation(backend, scope_a):
    """CONTRACT: ancestors root-to-tip with the cloned user row once (child carrier wins); DISPLAY shows each logical message once, live row preferred, ordered by first row; a /branch copy stands alone."""
    script = lineage_script().session(SessionId("reads-branch"), parent=SESSION, branch=True).msg(SessionId("reads-branch"), "assistant", "branch reply", 20)
    ids = script.apply(backend, scope_a)
    repo = backend.open(scope_a).messages
    display = repo.read_conversation(CHILD, ConversationRequest(view=TranscriptView.DISPLAY, include_lineage=True, include_row_ids=True))
    assert [m.content for m in display] == [
        "root u1", "root a1", "root a1 follow-up", "child u2", "child a2", {"a": 1, "b": 2}, {"b": 2, "a": 1}, "SUMMARY"]
    assert display[3].row_id is not None and display[3].row_id > int(ids[6])  # the carried live copy, not the archived original
    # Structured content dedupes by stored bytes: same keys in another order is another message (both survive).
    assert [m.row_id for m in display[5:7]] == [int(ids[5]), int(ids[6]) + 4]  # the archived {a,b} row; the carried live {b,a} copy
    assert [m.content for m in repo.read_conversation(CHILD)] == ["SUMMARY", "child u2", "child a2", {"b": 2, "a": 1}]
    assert [m.content for m in repo.read_conversation(CHILD, ConversationRequest(include_lineage=True))] == [
        "root u1", "root a1", "SUMMARY", "child u2", "child a2", {"b": 2, "a": 1}]
    assert [m.content for m in repo.read_conversation(SessionId("reads-branch"), ConversationRequest(include_lineage=True))] == ["branch reply"]


def test_read_conversation_dedupes_the_rotation_cloned_user_row_and_strips_harness_turns(backend, scope_a):
    """CONTRACT: an exact (timestamp, content) user clone in the child appears once as the child's row; harness turns drop; a bare [marker] tool-call content is blanked."""
    store_a = backend.open(scope_a)
    script = (Script().session(SESSION).session(CHILD, parent=SESSION)
              .msg(SESSION, "user", "cloned ask", 0).end(SESSION, "compression")
              .msg(CHILD, "user", "cloned ask", 0)
              .msg(CHILD, "assistant", "[memory]", 1, tool_calls=({"id": "c", "type": "function", "function": {"name": "memory", "arguments": "{}"}},))
              .msg(CHILD, "tool", "saved", 2, tool_call_id="c", tool_name="memory")
              .msg(CHILD, "user", "Review the conversation above and consider saving to memory. Whatever.", 3)
              .msg(CHILD, "assistant", "curator reply", 4))
    ids = script.apply(backend, scope_a)
    out = store_a.messages.read_conversation(CHILD, ConversationRequest(include_lineage=True, include_row_ids=True))
    assert [(m.content, m.row_id) for m in out] == [("cloned ask", int(ids[1])), ("", int(ids[2])), ("saved", int(ids[3]))]


# ── read_resume_conversations / assert_resume_safe ───────────────────────────


def test_read_resume_conversations_equal_the_separate_reads(backend, scope_a):
    """CONTRACT: (model, display) equal read(ACTIVE, repair, row_ids) on the tip and read(DISPLAY, lineage, row_ids)."""
    lineage_script().apply(backend, scope_a)
    repo = backend.open(scope_a).messages
    pair = repo.read_resume_conversations(CHILD)
    assert dumps(pair.model) == dumps(repo.read_conversation(CHILD, ConversationRequest(repair_alternation=True, include_row_ids=True)))
    assert dumps(pair.display) == dumps(repo.read_conversation(CHILD, ConversationRequest(view=TranscriptView.DISPLAY, include_lineage=True, include_row_ids=True)))
    assert pair.model[0].compressed_summary is True and pair.display[-1].compressed_summary is False
    empty = repo.read_resume_conversations(SessionId("never"))
    assert (empty.model, empty.display) == ((), ())


def test_assert_resume_safe_scopes_bounds_and_refusals(backend, scope_a):
    """CONTRACT: default scope = lineage active+compaction-archived (rewound excluded); tip_only = tip's active rows; count bounded at limit+1; 0 disables; negative refuses; unknown counts 0."""
    script = lineage_script().msg(CHILD, "user", "rewound", 30).rewind(CHILD, 7)
    script.apply(backend, scope_a)
    repo = backend.open(scope_a).messages
    assert repo.assert_resume_safe(CHILD, max_messages=100) == 11  # root 2 + child 5 archived + 4 live; the rewound row excluded
    assert repo.assert_resume_safe(CHILD, max_messages=100, tip_only=True) == 4
    assert repo.assert_resume_safe(CHILD, max_messages=0) == 0
    with pytest.raises(ResumeTooLarge) as exc:
        repo.assert_resume_safe(CHILD, max_messages=4)
    assert (exc.value.message_count, exc.value.limit, exc.value.scope) == (5, 4, "across its lineage")
    with pytest.raises(ResumeTooLarge) as exc:
        repo.assert_resume_safe(CHILD, max_messages=2, tip_only=True)
    assert (exc.value.message_count, exc.value.scope) == (3, "in its tip segment")
    with pytest.raises(ValueError):
        repo.assert_resume_safe(CHILD, max_messages=-1)
    assert repo.assert_resume_safe(SessionId("never"), max_messages=5) == 0


# ── anchored reads ───────────────────────────────────────────────────────────


def scroll_script() -> Script:
    script = Script().session(SESSION).session(OTHER)
    for i in range(10):
        role = ("user", "assistant", "tool")[i % 3]
        content: Any = "" if i == 8 else f"m{i}"
        if i == 1:
            content = [{"type": "text", "text": "m1 as a media part"}]  # QUIRK(nul-length): never a bookend
        script.msg(SESSION, role, content, i, tool_call_id=f"c{i}" if role == "tool" else None)
    return script.rewind(SESSION, 9)


def test_read_around_window_counts_and_foreign_anchor(backend, scope_a):
    """CONTRACT: up to window rows either side in id order over EVERY visibility; before/after counted strictly; a foreign anchor is the typed empty; a negative window is the anchor alone."""
    ids = scroll_script().apply(backend, scope_a)
    repo = backend.open(scope_a).messages
    view = repo.read_around(SESSION, ids[4], window=2)
    assert ([m.id for m in view.window], view.before, view.after) == (ids[2:7], 2, 2)
    head = repo.read_around(SESSION, ids[0], window=3)
    assert (head.before, head.after) == (0, 3)
    tail = repo.read_around(SESSION, ids[9], window=3)
    assert ([m.id for m in tail.window], tail.after) == (ids[6:], 0)
    foreign = repo.read_around(OTHER, ids[4])
    assert (foreign.window, foreign.before, foreign.after) == ((), 0, 0)
    assert [m.id for m in (repo.read_around(SESSION, ids[4], window=-3)).window] == [ids[4]]


def test_read_anchored_view_role_filter_and_bookends(backend, scope_a):
    """CONTRACT: the window keeps keep_roles plus the anchor; bookends are the first/last `bookend` kept-role non-empty rows strictly outside the unfiltered window; keep_roles=None keeps every role; bookend<=0 reads empty bookends."""
    ids = scroll_script().apply(backend, scope_a)
    repo = backend.open(scope_a).messages
    view = repo.read_anchored_view(SESSION, ids[5], window=1, bookend=2)
    assert [m.id for m in view.window] == ids[4:7]
    assert (view.before, view.after) == (1, 1)
    assert [m.id for m in view.bookend_start] == [ids[0], ids[3]]  # the multimodal row 1 does not qualify
    assert [m.id for m in view.bookend_end] == [ids[7], ids[9]]
    narrow = repo.read_anchored_view(SESSION, ids[4], window=1, bookend=2)
    assert [m.id for m in narrow.window] == [ids[3], ids[4]]
    everything = repo.read_anchored_view(SESSION, ids[5], window=1, bookend=2, keep_roles=None)
    assert [m.id for m in everything.bookend_end] == [ids[7], ids[9]]
    assert (repo.read_anchored_view(SESSION, ids[5], window=1, bookend=0)).bookend_start == ()
    assert (repo.read_anchored_view(OTHER, ids[5])).window == ()


# ── pull-request mentions ────────────────────────────────────────────────────


def test_find_pr_url_hits_loose_scan(backend, scope_a):
    """CONTRACT: tool rows whose string content contains /pull/ ASCII case-insensitively, any visibility, insertion order across the named sessions; empty and duplicate ids collapse; non-tool rows never match."""
    store_a = backend.open(scope_a)
    script = (Script().session(SESSION).session(OTHER)
              .msg(SESSION, "user", "see https://github.com/o/r/pull/1", 0)
              .msg(SESSION, "tool", "opened https://github.com/o/r/pull/2", 1, tool_call_id="a")
              .msg(OTHER, "tool", "OPENED HTTPS://GITHUB.COM/O/R/PULL/3", 2, tool_call_id="b")
              .msg(SESSION, "tool", "no link here", 3, tool_call_id="c")
              .msg(SESSION, "user", "x", 4)
              .msg(SESSION, "tool", "https://github.com/o/r/pull/4 after", 5, tool_call_id="d")
              .rewind(SESSION, 4))
    script.apply(backend, scope_a)
    hits = store_a.messages.find_pr_url_hits([SESSION, SessionId(""), OTHER, SESSION])
    assert [(h.session_id, h.content) for h in hits] == [
        (SESSION, "opened https://github.com/o/r/pull/2"),
        (OTHER, "OPENED HTTPS://GITHUB.COM/O/R/PULL/3"),
        (SESSION, "https://github.com/o/r/pull/4 after"),
    ]
    assert store_a.messages.find_pr_url_hits([]) == ()


# ── gateway guards ───────────────────────────────────────────────────────────


def test_latest_conversation_role_skips_bookkeeping_and_inactive_rows(backend, scope_a):
    """CONTRACT: the newest ACTIVE row's role ignoring session_meta/system; None when nothing qualifies."""
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    store_a.sessions.create(NewSession(session_id=SESSION, source="cli"))
    assert repo.latest_conversation_role(SESSION) is None
    repo.append(SESSION, new("user", "ask", 0))
    repo.append(SESSION, new("assistant", "reply", 1))
    target = repo.append(SESSION, new("user", "open tail", 2))
    repo.append(SESSION, new("session_meta", "{}", 3))
    repo.append(SESSION, new("system", "note", 4))
    assert repo.latest_conversation_role(SESSION) == "user"
    rewind_from(repo, SESSION, target)
    assert repo.latest_conversation_role(SESSION) == "assistant"


def test_has_platform_message_id_counts_every_visibility_within_the_session(backend, scope_a):
    """CONTRACT: any stored row with the platform id dedupes, rewound included; scoped to the session."""
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    store_a.sessions.create(NewSession(session_id=SESSION, source="cli"))
    store_a.sessions.create(NewSession(session_id=OTHER, source="cli"))
    assert repo.has_platform_message_id(SESSION, "tg-9") is False
    row = repo.append(SESSION, new("user", "hi", 0, platform_message_id="tg-9"))
    assert repo.has_platform_message_id(SESSION, "tg-9") is True
    assert repo.has_platform_message_id(OTHER, "tg-9") is False
    rewind_from(repo, SESSION, row)
    assert repo.has_platform_message_id(SESSION, "tg-9") is True


def test_has_gateway_input_owner_predicate(backend, scope_a):
    """CONTRACT: a user row, not observed, active or compaction-archived (never rewound), whose display_metadata.gateway_input_owner equals the owner."""
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    store_a.sessions.create(NewSession(session_id=SESSION, source="cli"))
    repo.append(SESSION, new("user", "a", 0, display_metadata={"gateway_input_owner": "peer-A"}))
    assert repo.has_gateway_input_owner(SESSION, "peer-A") is True
    assert repo.has_gateway_input_owner(SESSION, "peer-B") is False
    repo.append(SESSION, new("user", "b", 1, observed=True, display_metadata={"gateway_input_owner": "peer-B"}))
    assert repo.has_gateway_input_owner(SESSION, "peer-B") is False
    repo.append(SESSION, new("assistant", "c", 2, display_metadata={"gateway_input_owner": "peer-C"}))
    assert repo.has_gateway_input_owner(SESSION, "peer-C") is False
    repo.compact(SESSION, [new("user", "S", 9, compressed_summary=True)])
    assert repo.has_gateway_input_owner(SESSION, "peer-A") is True
    rewound = repo.append(SESSION, new("user", "d", 3, display_metadata={"gateway_input_owner": "peer-D"}))
    rewind_from(repo, SESSION, rewound)
    assert repo.has_gateway_input_owner(SESSION, "peer-D") is False


def test_latest_conversation_boundary_is_the_peer_generation(backend, backend_name, scope_a, scope_b, request):
    """CONTRACT: None until a reset end; +1 per reset end of a session routed to (session_key, source); compression ends and repeated ends of an ended session do not advance; the peer is (key, source); empty components read None; profile-scoped; survives deleting the sessions."""
    store_b = backend.open(scope_b)
    peer = ConversationPeer(session_key="chat-1", source="telegram")
    repo, sessions = backend.open(scope_a).messages, backend.open(scope_a).sessions
    assert repo.latest_conversation_boundary(peer) is None
    sessions.create(NewSession(session_id=SessionId("gw-1"), source="telegram", session_key="chat-1"))
    sessions.end(SessionId("gw-1"), "compression")
    assert repo.latest_conversation_boundary(peer) is None
    sessions.create(NewSession(session_id=SessionId("gw-2"), source="telegram", session_key="chat-1"))
    sessions.end(SessionId("gw-2"), "session_reset")
    assert repo.latest_conversation_boundary(peer) == 1
    sessions.end(SessionId("gw-2"), "idle")
    assert repo.latest_conversation_boundary(peer) == 1
    sessions.create(NewSession(session_id=SessionId("gw-3"), source="telegram", session_key="chat-1"))
    sessions.end(SessionId("gw-3"), "idle")
    assert repo.latest_conversation_boundary(peer) == 2
    assert repo.latest_conversation_boundary(ConversationPeer(session_key="chat-1", source="api")) is None
    assert repo.latest_conversation_boundary(ConversationPeer(session_key="", source="telegram")) is None
    assert store_b.messages.latest_conversation_boundary(peer) is None
    for sid in ("gw-1", "gw-2", "gw-3"):
        sessions.delete(SessionId(sid), media_root=MEDIA_ROOT)
    assert repo.latest_conversation_boundary(peer) == 2


def test_latest_conversation_boundary_reads_the_counter_the_end_transition_advances(backend, backend_name, scope_a, scope_b):
    """CONTRACT (read half, both backends): the generation is a per-peer counter, None at 0, keyed by (key, source), profile-scoped; on the reference the write is driven through the seam the sessions family wires into ``end``."""
    store_b = backend.open(scope_b)
    peer = ConversationPeer(session_key="chat-1", source="telegram")
    repo, sessions = backend.open(scope_a).messages, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=SessionId("gw-1"), source="telegram", session_key="chat-1"))
    sessions.create(NewSession(session_id=SessionId("gw-2"), source="telegram", session_key="chat-1"))
    assert repo.latest_conversation_boundary(peer) is None

    def end(sid: str, reason: str) -> None:
        if backend_name == "in_memory":
            from tests.contracts._reference.conversation.transcript_store import advance_conversation_generation

            advance_conversation_generation(backend, scope_a, sid, reason)
        else:
            sessions.end(SessionId(sid), reason)

    end("gw-1", "compression")
    assert repo.latest_conversation_boundary(peer) is None
    end("gw-2", "session_reset")
    assert repo.latest_conversation_boundary(peer) == 1
    sessions.create(NewSession(session_id=SessionId("gw-3"), source="telegram", session_key="chat-1"))
    end("gw-3", "idle")
    assert repo.latest_conversation_boundary(peer) == 2
    assert repo.latest_conversation_boundary(ConversationPeer(session_key="chat-1", source="api")) is None
    assert store_b.messages.latest_conversation_boundary(peer) is None
    for sid in ("gw-1", "gw-2", "gw-3"):
        sessions.delete(SessionId(sid), media_root=MEDIA_ROOT)
    assert repo.latest_conversation_boundary(peer) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Transcript writes
#
# Cases for eight after-the-fact write verbs. Store cases seed through public protocol operations and observe through public reads (``read_transcript`` for stored columns, ``read_conversation`` for the replay projection); record-only cases need no store, and the reference refusal case replaces methods to provoke post-read failures.
# ═══════════════════════════════════════════════════════════════════════════

from persistence import CompressionClosed, CompressionInProgress, Conflict, RewindRefused, TurnLeaseLost
from persistence.conversation.sessions import END_REASON_COMPRESSION
from tests.contracts.conftest import require_capability
from persistence.conversation.messages import (
    DelegationDelivery,
    MarkerPurgeReport,
    ReplaceMode,
    RewindTargetUnavailable,
    UnseenReaction,
    UserTurnRewind,
)
from tests.contracts._scopes import MEDIA_ROOT, profile_scope
from tests.contracts._waiting import wait_past

W = SessionId("writes-s1")
W_CHILD = SessionId("writes-child")


def _seed_dialogue(backend, scope, sid=W):
    """u1 a1 u2 a2 u3 a3 (ids returned in order)."""
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=sid, source="cli"))
    ids = []
    for i, (role, content) in enumerate([("user", "u1"), ("assistant", "a1"), ("user", "u2"), ("assistant", "a2"), ("user", "u3"), ("assistant", "a3")]):
        ids.append(store.messages.append(sid, new(role, content, i)))
    return ids


# ── append_delegation_delivery ───────────────────────────────────────────────


def test_delegation_delivery_record_validates_identity_before_any_store_access():
    """CONTRACT: a missing or empty delegation_id is ValueError at construction; any other value keys the delivery as given, an absent notice reading ""."""
    ok = DelegationDelivery(content="done", metadata={"delegation_id": "d-1", "delivery_notice": "n"})
    assert ok.identity == ("d-1", "n")
    assert DelegationDelivery(content="done", metadata={"delegation_id": "d-1"}).identity == ("d-1", "")
    assert DelegationDelivery(content="x", metadata={"delegation_id": 7, "delivery_notice": None}).identity == (7, None)
    for meta in ({}, {"delegation_id": ""}, {"delegation_id": None}):
        with pytest.raises(ValueError):
            DelegationDelivery(content="x", metadata=meta)


def test_append_delegation_delivery_is_idempotent_by_identity(backend, scope_a):
    """CONTRACT: one ACTIVE user row with display_kind async_delegation_complete and the metadata; the same identity returns the same id and writes nothing; a different notice is a new row."""
    ids = _seed_dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    delivery = DelegationDelivery(content="[delegation d-1 finished]", metadata={"delegation_id": "d-1", "kind": "api"})
    first = repo.append_delegation_delivery(W, delivery)
    again = repo.append_delegation_delivery(W, DelegationDelivery(content="different text, same unit", metadata={"delegation_id": "d-1"}))
    assert again == first
    rows = repo.read_transcript(W)
    assert [r.id for r in rows] == [*ids, first]
    row = rows[-1]
    assert (row.role, row.content, row.display_kind) == ("user", "[delegation d-1 finished]", "async_delegation_complete")
    assert row.display_metadata == {"delegation_id": "d-1", "kind": "api"}
    assert (totals(backend, scope_a, W)).message_count == 7
    noticed = repo.append_delegation_delivery(W, DelegationDelivery(content="x", metadata={"delegation_id": "d-1", "delivery_notice": "retry"}))
    assert noticed != first
    assert len(repo.read_transcript(W)) == 8
    # Identity survives the row leaving the active set.
    rewind_from(repo, W, first)
    assert repo.append_delegation_delivery(W, delivery) == first
    assert len(repo.read_transcript(W, TranscriptRead(view=TranscriptView.ALL))) == 8


def test_append_delegation_delivery_finds_the_ancestor_row_and_refuses_a_compression_closed_session(backend, scope_a):
    """CONTRACT: a delivery replayed on the continuation finds the row the compression-ended parent persisted; delivering INTO the closed parent is CompressionClosed naming the continuation."""
    store_a = backend.open(scope_a)
    _seed_dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    delivery = DelegationDelivery(content="unit", metadata={"delegation_id": "d-rot"})
    parent_row = repo.append_delegation_delivery(W, delivery)
    store_a.sessions.create(NewSession(session_id=W_CHILD, source="cli", parent_session_id=W))
    store_a.sessions.end(W, "compression")
    assert repo.append_delegation_delivery(W_CHILD, delivery) == parent_row
    assert repo.read_transcript(W_CHILD) == ()
    # Existence first: the replayed identity still returns its row on the now-closed parent; only a NEW insert is refused there.
    assert repo.append_delegation_delivery(W, delivery) == parent_row
    with pytest.raises(CompressionClosed) as closed:
        repo.append_delegation_delivery(W, DelegationDelivery(content="late", metadata={"delegation_id": "d-late"}))
    assert closed.value.tip == W_CHILD
    assert len(repo.read_transcript(W)) == 7


def test_append_delegation_delivery_searches_only_compression_ancestors(backend, scope_a):
    """CONTRACT: a parent that is still open (or ended for another reason) is not part of the delivery lineage: the child gets its own row; a grandparent behind a compression edge IS searched."""
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    grand, parent, child = SessionId("writes-grand"), SessionId("writes-parent"), SessionId("writes-leaf")
    store_a.sessions.create(NewSession(session_id=grand, source="cli"))
    store_a.sessions.create(NewSession(session_id=parent, source="cli", parent_session_id=grand))
    store_a.sessions.create(NewSession(session_id=child, source="cli", parent_session_id=parent))
    delivery = DelegationDelivery(content="unit", metadata={"delegation_id": "d-open"})
    in_parent = repo.append_delegation_delivery(parent, delivery)
    in_child = repo.append_delegation_delivery(child, delivery)  # parent is open: not searched
    assert in_child != in_parent
    assert [r.id for r in repo.read_transcript(child)] == [in_child]
    deep = DelegationDelivery(content="unit", metadata={"delegation_id": "d-deep"})
    in_grand = repo.append_delegation_delivery(grand, deep)
    store_a.sessions.end(grand, "compression")
    store_a.sessions.end(parent, "compression")
    assert repo.append_delegation_delivery(child, deep) == in_grand
    store_a.sessions.create(NewSession(session_id=SessionId("writes-idle-child"), source="cli", parent_session_id=child))
    store_a.sessions.end(child, "idle")
    assert repo.append_delegation_delivery(SessionId("writes-idle-child"), deep) != in_grand  # the idle edge breaks the chain


# ── take_unseen_reactions ────────────────────────────────────────────────────


def test_take_unseen_reactions_surfaces_each_reaction_exactly_once(backend, scope_a):
    """CONTRACT: the author's unseen reactions on active or compaction-archived rows of the resume lineage, in row order, with role/emoji/text; a take after a take is empty; other authors, rewound rows and unknown sessions yield nothing; the rows themselves never move."""
    ids = _seed_dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    repo.set_reaction(W, ids[1], "👍", author="user")
    repo.set_reaction(W, ids[3], "❤️", author="user")
    repo.set_reaction(W, ids[3], "👀", author="agent")
    repo.set_reaction(W, ids[5], "😂", author="user")
    rewind_from(repo, W, ids[4])  # a3 leaves the active set with its reaction
    before = [(m.role, m.content, m.row_id) for m in repo.read_conversation(W, ConversationRequest(include_row_ids=True))]
    taken = repo.take_unseen_reactions(W, author="user")
    assert taken == (
        UnseenReaction(row_id=ids[1], role="assistant", emoji="👍", text="a1"),
        UnseenReaction(row_id=ids[3], role="assistant", emoji="❤️", text="a2"),
    )
    assert repo.take_unseen_reactions(W, author="user") == ()
    assert repo.take_unseen_reactions(W, author="agent") == (
        UnseenReaction(row_id=ids[3], role="assistant", emoji="👀", text="a2"),)
    assert repo.take_unseen_reactions(SessionId("nobody"), author="user") == ()
    assert repo.take_unseen_reactions(W, author="") == ()
    # The take marks presentation metadata only: the model-visible rows are unchanged.
    assert [(m.role, m.content, m.row_id) for m in repo.read_conversation(W, ConversationRequest(include_row_ids=True))] == before
    # A fresh reaction after the take surfaces on the next take; a re-set of the same emoji is a clear (tapback), not a new unseen one.
    repo.set_reaction(W, ids[1], "🔥", author="user")
    repo.set_reaction(W, ids[3], "❤️", author="user")
    assert repo.take_unseen_reactions(W, author="user") == (
        UnseenReaction(row_id=ids[1], role="assistant", emoji="🔥", text="a1"),)


def test_take_unseen_reactions_text_is_the_plain_string_content_only(backend, scope_a):
    """CONTRACT: structured content surfaces as an empty text."""
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(session_id=W, source="cli"))
    rid = store_a.messages.append(W, new("user", [{"type": "text", "text": "multimodal"}], 0))
    store_a.messages.set_reaction(W, rid, "👍", author="user")
    assert store_a.messages.take_unseen_reactions(W) == (UnseenReaction(row_id=rid, role="user", emoji="👍", text=""),)


# ── set_latest_matching_message_display_kind ─────────────────────────────────


def test_set_latest_matching_message_display_kind_stamps_the_newest_active_match_only(backend, scope_a):
    """CONTRACT: newest ACTIVE row with equal role and string content gets display_kind and the WHOLE display_metadata; nothing else moves; no match, or an empty content/display_kind, is False."""
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(session_id=W, source="tui"))
    repo = backend.open(scope_a).messages
    a = repo.append(W, new("assistant", "same text", 0, display_metadata={"old": True}))
    b = repo.append(W, new("assistant", "same text", 1, display_metadata={"old": True}))
    c = repo.append(W, new("assistant", "same text", 2))
    u = repo.append(W, new("user", "same text", 3))
    rewind_from(repo, W, u)  # u and everything after leave the active set; c stays (it is before u)
    assert repo.set_latest_matching_message_display_kind(W, role="assistant", content="same text", display_kind="synthesized", display_metadata={"producer": "cron"}) is True
    rows = {r.id: r for r in repo.read_transcript(W, TranscriptRead(view=TranscriptView.ALL))}
    assert (rows[c].display_kind, rows[c].display_metadata) == ("synthesized", {"producer": "cron"})
    assert (rows[a].display_kind, rows[a].display_metadata) == (None, {"old": True})
    assert (rows[b].display_kind, rows[b].display_metadata) == (None, {"old": True})
    assert rows[u].display_kind is None
    # Replace-wholesale: None clears the metadata; the stamp moves to the newest match again (c).
    assert repo.set_latest_matching_message_display_kind(W, role="assistant", content="same text", display_kind="plain") is True
    rows = {r.id: r for r in repo.read_transcript(W, TranscriptRead(view=TranscriptView.ALL))}
    assert (rows[c].display_kind, rows[c].display_metadata) == ("plain", None)
    assert repo.set_latest_matching_message_display_kind(W, role="assistant", content="no such text", display_kind="x") is False
    assert repo.set_latest_matching_message_display_kind(W, role="user", content="same text", display_kind="x") is False  # only a rewound user row matches
    assert repo.set_latest_matching_message_display_kind(SessionId("nobody"), role="assistant", content="same text", display_kind="x") is False
    for content, kind in (("", "x"), ("same text", "")):
        assert repo.set_latest_matching_message_display_kind(W, role="assistant", content=content, display_kind=kind) is False
    assert (rows[c].display_kind, rows[c].display_metadata) == ("plain", None)
    # The stamp is presentation only: the replay projection's role/content bytes are unchanged.
    assert [(m.role, m.content) for m in repo.read_conversation(W)] == [("assistant", "same text")] * 3


# ── api_content sidecar stamps ───────────────────────────────────────────────


def test_set_message_api_content_is_a_guarded_row_addressed_stamp(backend, scope_a):
    """CONTRACT: stamped iff (session, user role, ACTIVE, equal stored content); False and no write otherwise; None clears; the projection replays the sidecar VERBATIM."""
    ids = _seed_dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    assert repo.set_message_api_content(W, ids[2], "u2", "u2 [exact api bytes]") is True
    assert [m.api_content for m in repo.read_conversation(W)] == [None, None, "u2 [exact api bytes]", None, None, None]
    assert repo.set_message_api_content(W, ids[2], "u2 (edited)", "other") is False  # content guard
    assert repo.set_message_api_content(W, ids[1], "a1", "x") is False  # not a user row
    assert repo.set_message_api_content(SessionId("elsewhere"), ids[2], "u2", "x") is False  # foreign session
    assert repo.set_message_api_content(W, MessageId(999_999), "u2", "x") is False  # unknown row
    rewind_from(repo, W, ids[4])
    assert repo.set_message_api_content(W, ids[4], "u3", "x") is False  # rewound
    assert [m.api_content for m in repo.read_conversation(W)] == [None, None, "u2 [exact api bytes]", None]
    assert repo.set_message_api_content(W, ids[2], "u2", None) is True
    assert [m.api_content for m in repo.read_conversation(W)] == [None] * 4


def test_set_user_message_content_rewrites_one_active_user_row(backend, scope_a, scope_b):
    """CONTRACT: rewritten iff (session, user role, ACTIVE); False and no write otherwise; only content moves (the api_content sidecar stays); list content round-trips; other scopes are untouched."""
    store_b = backend.open(scope_b)
    ids = _seed_dialogue(backend, scope_a)
    _seed_dialogue(backend, scope_b)
    repo = backend.open(scope_a).messages
    assert repo.set_message_api_content(W, ids[2], "u2", "u2 [api]") is True
    parts = [{"type": "text", "text": "u2 expanded"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    assert repo.set_user_message_content(W, ids[2], parts) is True
    conv = repo.read_conversation(W)
    assert [m.content for m in conv] == ["u1", "a1", parts, "a2", "u3", "a3"]
    assert conv[2].api_content == "u2 [api]"
    assert repo.set_user_message_content(W, ids[0], "u1 expanded") is True
    assert (repo.read_conversation(W))[0].content == "u1 expanded"
    assert repo.set_user_message_content(W, ids[1], "x") is False  # not a user row
    assert repo.set_user_message_content(SessionId("elsewhere"), ids[2], "x") is False  # foreign session
    assert repo.set_user_message_content(W, MessageId(999_999), "x") is False  # unknown row
    rewind_from(repo, W, ids[4])
    assert repo.set_user_message_content(W, ids[4], "x") is False  # rewound
    assert [m.content for m in store_b.messages.read_conversation(W)] == ["u1", "a1", "u2", "a2", "u3", "a3"]


def test_set_user_message_content_with_display_metadata_merges_it_and_clears_api_content(backend, scope_a):
    """CONTRACT: with ``display_metadata`` the rewrite also merge-patches it into the row's display metadata (its keys win, a ``None`` value removes the key, a key another writer added such as a reaction stays, an empty mapping changes nothing) and clears its api_content sidecar; the guards are the same as without it."""
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(session_id=W, source="tui"))
    repo = store_a.messages
    typed = repo.append(W, new("user", "typed", 0, display_metadata={"old": 1, "typed_ask": "typed"}))
    reply = repo.append(W, new("assistant", "reply", 1))
    assert repo.set_message_api_content(W, typed, "typed", "typed [api]") is True
    assert [r.emoji for r in repo.set_reaction(W, typed, "👍", author="user")] == ["👍"]
    withheld = {"withheld_prompt": "typed", "typed_ask": None}
    assert repo.set_user_message_content(W, typed, "[withheld]", display_metadata=withheld) is True

    def stored():
        row = repo.read_transcript(W)[0]
        return row.content, row.api_content, {k: v for k, v in (row.display_metadata or {}).items() if k != "reactions"}

    assert stored() == ("[withheld]", None, {"old": 1, "withheld_prompt": "typed"})
    assert repo.read_transcript(W)[1].display_metadata is None
    assert sorted(r.emoji for r in repo.set_reaction(W, typed, "👀", author="agent")) == sorted(["👍", "👀"])
    assert repo.set_user_message_content(W, reply, "x", display_metadata=withheld) is False  # not a user row
    assert repo.set_user_message_content(SessionId("elsewhere"), typed, "x", display_metadata=withheld) is False
    assert repo.set_user_message_content(W, typed, "[withheld]", display_metadata={}) is True
    assert stored() == ("[withheld]", None, {"old": 1, "withheld_prompt": "typed"})


def test_set_latest_user_api_content_addresses_only_the_newest_active_user_row(backend, scope_a):
    """CONTRACT: the positional form stamps the newest ACTIVE user row when its content matches, never an older one; a rewind moves which row is newest."""
    ids = _seed_dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    assert repo.set_latest_user_api_content(W, "u2", "x") is False  # u3 is the newest active user row
    assert repo.set_latest_user_api_content(W, "u3", "u3 [api]") is True
    assert [m.api_content for m in repo.read_conversation(W)] == [None, None, None, None, "u3 [api]", None]
    rewind_from(repo, W, ids[4])
    assert repo.set_latest_user_api_content(W, "u2", "u2 [api]") is True
    assert [m.api_content for m in repo.read_conversation(W)] == [None, None, "u2 [api]", None]
    assert repo.set_latest_user_api_content(SessionId("nobody"), "u2", "x") is False
    # Identical repeated turns make the newest row compare equal even when the caller meant the older one.
    repo.append(W, new("user", "u2", 9))
    assert repo.set_latest_user_api_content(W, "u2", "meant for the older u2") is True
    assert [m.api_content for m in repo.read_conversation(W)][-1] == "meant for the older u2"


# ── purge_stale_tool_call_markers ────────────────────────────────────────────


def _seed_markers(backend, scope):
    """Two sessions; returns the ids the purge must address (assistant + tool_calls + marker content, any visibility) and the ids it must leave alone."""
    store = backend.open(scope)
    call = ({"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},)
    store.sessions.create(NewSession(session_id=W, source="cli"))
    store.sessions.create(NewSession(session_id=OTHER, source="cli"))
    repo = backend.open(scope).messages
    hit_1 = repo.append(W, new("assistant", "[terminal]", 0, tool_calls=call))
    keep_tool = repo.append(W, new("tool", "[terminal]", 1, tool_call_id="c1", tool_name="terminal"))
    keep_no_calls = repo.append(W, new("assistant", "[terminal]", 2))
    keep_prose = repo.append(W, new("assistant", "ran [terminal] for you", 3, tool_calls=call))
    hit_padded = repo.append(OTHER, new("assistant", "  [read_file.v2-x]\n", 4, tool_calls=call))
    keep_structured = repo.append(OTHER, new("assistant", [{"type": "text", "text": "[terminal]"}], 5, tool_calls=call))
    rewound_target = repo.append(OTHER, new("user", "u", 6))
    hit_rewound = repo.append(OTHER, new("assistant", "[grep]", 7, tool_calls=call))
    rewind_from(repo, OTHER, rewound_target)
    return [hit_1, hit_padded, hit_rewound], [keep_tool, keep_no_calls, keep_prose, keep_structured]


def test_purge_stale_tool_call_markers_dry_run_reports_without_writing(backend, scope_a):
    """CONTRACT: dry_run lists the matching rows across every session in the scope, each once in no defined order (compared as a set), and blanks nothing; no backup is taken."""
    store_a = backend.open(scope_a)
    hits, _ = _seed_markers(backend, scope_a)
    report = store_a.messages.purge_stale_tool_call_markers(dry_run=True)
    assert (report.dry_run, report.rows_affected, len(report.row_ids), report.backup_path) == (True, 3, 3, None)
    assert set(report.row_ids) == set(hits)
    stored = {r.id: r.content for sid in (W, OTHER) for r in store_a.messages.read_transcript(sid, TranscriptRead(view=TranscriptView.ALL))}
    assert [stored[i] for i in hits] == ["[terminal]", "  [read_file.v2-x]\n", "[grep]"]


def test_purge_stale_tool_call_markers_blanks_only_the_matching_content(backend, scope_a):
    """CONTRACT: matching rows get content ""; tool_calls, visibility and every other row are untouched; a second run finds nothing; the backend's snapshot (when it has one) exists before the write and is a real store copy."""
    hits, keeps = _seed_markers(backend, scope_a)
    repo = backend.open(scope_a).messages
    before = {r.id: r for sid in (W, OTHER) for r in repo.read_transcript(sid, TranscriptRead(view=TranscriptView.ALL))}
    totals_before = (totals(backend, scope_a, W), totals(backend, scope_a, OTHER))
    report = repo.purge_stale_tool_call_markers()
    assert (report.dry_run, report.rows_affected, len(report.row_ids)) == (False, 3, 3)
    assert set(report.row_ids) == set(hits)
    after = {r.id: r for sid in (W, OTHER) for r in repo.read_transcript(sid, TranscriptRead(view=TranscriptView.ALL))}
    for rid in hits:
        assert after[rid].content == ""
        assert (after[rid].tool_calls, after[rid].visibility, after[rid].role) == (before[rid].tool_calls, before[rid].visibility, "assistant")
    for rid in keeps:
        assert after[rid] == before[rid]
    assert (totals(backend, scope_a, W), totals(backend, scope_a, OTHER)) == totals_before
    assert repo.purge_stale_tool_call_markers() == MarkerPurgeReport(dry_run=False, rows_affected=0, row_ids=(), backup_path=None)
    if report.backup_path is not None:
        snapshot = Path(report.backup_path)
        assert snapshot.exists() and snapshot.stat().st_size > 0
    else:
        assert backend.capabilities.cross_process is False, "a file-backed backend must snapshot before the write"


def test_purge_stale_tool_call_markers_backup_false_takes_no_snapshot(backend, scope_a):
    """CONTRACT: backup=False writes without a snapshot; nothing to purge means no snapshot either way."""
    store_a = backend.open(scope_a)
    _seed_markers(backend, scope_a)
    report = store_a.messages.purge_stale_tool_call_markers(backup=False)
    assert (report.rows_affected, report.backup_path) == (3, None)


# ── rewind_user_turn ─────────────────────────────────────────────────────────


def _carrier(ask: str = "REAL ASK") -> str:
    """A composite compaction carrier: summary + live ask in one user row."""
    from tests.contracts._reference.conversation._turn_policy import composite_carrier

    return composite_carrier("old task", ask)


def _replay(repo, sid=W):
    return [m.as_openai_message() for m in repo.read_conversation(sid, ConversationRequest(include_row_ids=True))]


def test_rewind_user_turn_addresses_the_ordinal_over_the_active_projection(backend, scope_a):
    """CONTRACT: ordinal -1 is the newest user-originated turn; the target and everything after leave the active set; the prefix is the durable replay before the target (row ids, born-durable marker); live_text is the flattened ask; turns_undone counts the ordinal span; rewind_count increments once."""
    ids = _seed_dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    before = _replay(repo)
    out = repo.rewind_user_turn(W, -1)
    assert isinstance(out, UserTurnRewind)
    assert list(out.prefix) == before[:4]
    assert all(m.get("_row_id") == ids[i] and m.get(DB_PERSISTED_MARKER_KEY) is True for i, m in enumerate(out.prefix))
    assert (out.live_view["content"], out.live_text, out.rewound_count, out.turns_undone) == ("u3", "u3", 2, 1)
    assert [m["content"] for m in _replay(repo)] == ["u1", "a1", "u2", "a2"]
    assert [r.visibility for r in repo.read_transcript(W, TranscriptRead(view=TranscriptView.ALL))][4:] == [RowVisibility.REWOUND] * 2
    # Positive ordinals count from the oldest; a negative ordinal past the oldest clamps to it.
    out = repo.rewind_user_turn(W, -5)
    assert (out.live_text, out.turns_undone, out.rewound_count, list(out.prefix)) == ("u1", 2, 4, [])
    assert _replay(repo) == []
    assert (totals(backend, scope_a, W)).rewind_count == 2
    with pytest.raises(RewindTargetUnavailable):
        repo.rewind_user_turn(W, -1)


def test_rewind_user_turn_accepts_stored_text_the_projection_sanitizes(backend, scope_a):
    """CONTRACT: the durable-vs-warm agreement is checked on the comparison projection; stored bytes that the replay projection strips (surrounding whitespace) still rewind, with or without a warm history."""
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(session_id=W, source="cli"))
    repo = backend.open(scope_a).messages
    repo.append(W, new("user", "  hello  ", 0))
    repo.append(W, new("assistant", "hi", 1))
    out = repo.rewind_user_turn(W, -1, warm_history=[{"role": "user", "content": "  hello  "}, {"role": "assistant", "content": "hi"}])
    assert (out.live_text.strip(), out.rewound_count, list(out.prefix)) == ("hello", 2, [])
    repo.append(W, new("user", "  again  ", 2))
    assert (repo.rewind_user_turn(W, -1)).rewound_count == 1


def test_rewind_user_turn_refusals_write_nothing(backend, scope_a):
    """CONTRACT: out-of-range ordinal and require_composite on a plain turn are RewindTargetUnavailable; require_retryable on media content is ValueError; every refusal leaves the transcript untouched."""
    _seed_dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    repo.append(W, new("user", [{"type": "image_url", "image_url": {"url": "data:,x"}}], 6))
    before = _replay(repo)
    with pytest.raises(RewindTargetUnavailable):
        repo.rewind_user_turn(W, 4)
    with pytest.raises(RewindTargetUnavailable):
        repo.rewind_user_turn(W, -1, require_composite=True)
    with pytest.raises(ValueError) as excinfo:
        repo.rewind_user_turn(W, -1, require_retryable=True)
    assert not isinstance(excinfo.value, RewindTargetUnavailable)
    with pytest.raises(RewindTargetUnavailable):
        repo.rewind_user_turn(SessionId("nobody"), -1)
    assert _replay(repo) == before
    assert (totals(backend, scope_a, W)).rewind_count == 0


def test_rewind_user_turn_validates_the_warm_history_against_the_durable_transcript(backend, scope_a):
    """CONTRACT: a warm history with a different user-turn count or a different addressed turn is RewindRefused("history_changed") and nothing changes; an agreeing warm history (ephemeral scaffolding stripped) is returned as the prefix, keeping the warm dicts' own keys, with durable row ids adopted on request."""
    ids = _seed_dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    before = _replay(repo)
    warm = [{"role": r, "content": c, "extra": True} for r, c in [("user", "u1"), ("assistant", "a1"), ("user", "u2"), ("assistant", "a2"), ("user", "u3"), ("assistant", "a3")]]
    for disagreeing in (warm[:4], [*warm[:4], {"role": "user", "content": "u3 edited"}, warm[5]]):
        with pytest.raises(RewindRefused) as refused:
            repo.rewind_user_turn(W, -1, warm_history=disagreeing)
        assert (refused.value.reason, refused.value.entity, refused.value.entity_id) == ("history_changed", "transcript", str(W))
    assert _replay(repo) == before
    scaffolded = [*warm[:4], {"role": "user", "content": "internal", "_empty_recovery_synthetic": True}, *warm[4:]]
    out = repo.rewind_user_turn(W, -1, warm_history=scaffolded, adopt_row_ids=True)
    assert [m["_row_id"] for m in out.prefix] == ids[:4]
    assert all(m["extra"] for m in out.prefix)  # the warm dicts' own keys survive: it is the warm history that comes back, not the durable replay
    assert (out.live_text, out.rewound_count) == ("u3", 2)
    # Without adopt_row_ids the warm dicts are returned untouched.
    ids2 = _seed_dialogue(backend, scope_a, OTHER)
    out = repo.rewind_user_turn(OTHER, -1, warm_history=[dict(m) for m in warm])
    assert all("_row_id" not in m for m in out.prefix)
    assert [m["content"] for m in _replay(repo, OTHER)] == ["u1", "a1", "u2", "a2"]
    assert ids2[4] not in {r.id for r in repo.read_transcript(OTHER)}


def test_rewind_user_turn_keeps_a_composite_carrier_handoff_as_the_new_head(backend, scope_a):
    """CONTRACT: a composite carrier target is archived and its hidden handoff scaffold becomes the new active head; the prefix ends with that scaffold carrying the replacement row id and the born-durable marker; live_text is the ask alone; require_composite accepts it."""
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(session_id=W, source="tui"))
    repo = backend.open(scope_a).messages
    older = repo.append(W, new("user", "older ask", 0))
    a0 = repo.append(W, new("assistant", "older reply", 1))
    carrier = repo.append(W, new("user", _carrier(), 2))
    failed = repo.append(W, new("assistant", "failed", 3))
    out = repo.rewind_user_turn(W, -1, require_composite=True, require_retryable=True)
    assert (out.live_text, out.rewound_count, out.turns_undone) == ("REAL ASK", 2, 1)
    active = _replay(repo)
    assert [m["_row_id"] for m in active[:2]] == [older, a0]
    head = active[-1]
    assert head["_row_id"] not in {older, a0, carrier, failed}
    assert head["role"] == "user" and head["display_kind"] == "hidden"
    head_record = next(r for r in repo.read_transcript(W) if r.id == head["_row_id"])
    assert head_record.compressed_summary is True
    assert "REAL ASK" not in head["content"] and "old task" in head["content"]
    assert out.prefix[-1]["_row_id"] == head["_row_id"] and out.prefix[-1][DB_PERSISTED_MARKER_KEY] is True
    assert out.prefix[-1]["content"] == head["content"]
    visibility = {r.id: r.visibility for r in repo.read_transcript(W, TranscriptRead(view=TranscriptView.ALL))}
    assert visibility[carrier] is not RowVisibility.ACTIVE and visibility[failed] is not RowVisibility.ACTIVE
    assert (totals(backend, scope_a, W)).rewind_count == 1


def _rewind_state(backend, scope, sid=W):
    store = backend.open(scope)
    return [(r.id, r.visibility) for r in store.messages.read_transcript(sid, TranscriptRead(view=TranscriptView.ALL))], (totals(backend, scope, sid)).rewind_count


def test_rewind_user_turn_refuses_under_a_live_lease_and_on_a_closed_session(backend, scope_a):
    """CONTRACT: the admission refusals are typed and write nothing. A live turn lease on the conversation is TurnLeaseLost; a live compaction lease is CompressionInProgress and is checked first; a session ended by compression is CompressionClosed naming its live tip, checked after both leases. The lease checks are holder-blind and leave the leases with their holders; the warm-history check runs before them."""
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    _seed_dialogue(backend, scope_a)
    repo, coordination = backend.open(scope_a).messages, backend.open(scope_a).coordination
    before = _rewind_state(backend, scope_a)
    assert coordination.try_acquire_turn_lease(session_id=W, holder="live-turn", ttl_seconds=300.0) is True
    with pytest.raises((TurnLeaseLost, CompressionInProgress)) as refused:
        repo.rewind_user_turn(W, -1)
    assert isinstance(refused.value, TurnLeaseLost) and refused.value.entity_id == str(W)
    with pytest.raises(RewindRefused) as warm_first:
        repo.rewind_user_turn(W, -1, warm_history=[{"role": "user", "content": "other"}])
    assert warm_first.value.reason == "history_changed"
    assert coordination.try_acquire_compression_lock(session_id=SessionId(str(W)), holder="compressor", ttl_seconds=300.0) is True
    with pytest.raises((TurnLeaseLost, CompressionInProgress)) as refused:
        repo.rewind_user_turn(W, -1)
    assert isinstance(refused.value, CompressionInProgress)
    assert _rewind_state(backend, scope_a) == before
    assert coordination.refresh_turn_lease(session_id=W, holder="live-turn", ttl_seconds=300.0) is True
    assert coordination.get_compression_lock_holder(session_id=SessionId(str(W))) == "compressor"
    coordination.release_compression_lock(session_id=SessionId(str(W)), holder="compressor")
    store_a.sessions.end(W, END_REASON_COMPRESSION)
    with pytest.raises((TurnLeaseLost, CompressionInProgress)) as refused:
        repo.rewind_user_turn(W, -1)
    assert isinstance(refused.value, TurnLeaseLost)
    coordination.release_turn_lease(session_id=W, holder="live-turn")
    with pytest.raises(CompressionClosed) as closed:
        repo.rewind_user_turn(W, -1)
    assert (closed.value.entity_id, closed.value.tip) == (str(W), W)
    assert _rewind_state(backend, scope_a) == before


def test_rewind_user_turn_reclaims_a_stale_turn_lease_and_proceeds(backend, scope_a):
    """CONTRACT: an expired turn lease does not refuse; the rewind reclaims it, so its holder can no longer renew."""
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    _seed_dialogue(backend, scope_a)
    assert store_a.coordination.try_acquire_turn_lease(session_id=W, holder="dead-turn", ttl_seconds=0.1) is True
    wait_past(time.time() + 0.1)
    assert (store_a.messages.rewind_user_turn(W, -1)).rewound_count == 2
    assert store_a.coordination.refresh_turn_lease(session_id=W, holder="dead-turn", ttl_seconds=300.0) is False


@pytest.mark.parametrize("reason, text", [
    ("history_changed", "session history changed before the rewind could be persisted"),
    ("target_unidentified", "rewind target has no durable row identity"),
    ("target_changed", "rewind target changed before it could be persisted"),
    ("handoff_lost", "rewind did not retain its compaction handoff"),
])
def test_rewind_refused_is_a_transcript_conflict_carrying_its_reason_and_legacy_text(reason, text):
    """CONTRACT: each rewind precondition refusal is a Conflict on the transcript entity, distinct from the plain race Conflict by type and reason, whose message is the text callers show."""
    refused = RewindRefused(W, reason)
    assert isinstance(refused, Conflict) and type(refused) is not Conflict
    assert (refused.reason, refused.entity, refused.entity_id, str(refused)) == (reason, "transcript", str(W), text)


def test_reference_rewind_user_turn_classifies_the_post_read_refusals():
    """The reference arm of the refusals no public operation can provoke (a durable turn without row identity, a carrier rewind whose handoff did not come back, a target that vanishes or is edited between the snapshot and the write); the SQLite arm is pinned end to end in tests/tui_gateway/test_composite_carrier_rewind.py."""
    from dataclasses import replace as dc_replace

    from tests.contracts._reference.backend import open_reference_backend

    scope = profile_scope("profile-a")
    with open_reference_backend() as backend:
        store = backend.open(scope)
        repo = store.messages
        store.sessions.create(NewSession(session_id=W, source="tui"))
        repo.append(W, new("user", _carrier(), 0))
        repo.append(W, new("assistant", "failed", 1))
        read_conversation, rewind_to = repo.read_conversation, repo._rewind_to

        def _unidentified(*args, **kwargs):
            return tuple(dc_replace(m, row_id=None) for m in read_conversation(*args, **kwargs))

        repo.read_conversation = _unidentified
        with pytest.raises(RewindRefused) as refused:
            repo.rewind_user_turn(W, -1)
        assert refused.value.reason == "target_unidentified"
        assert (totals(backend, scope, W)).rewind_count == 0
        repo.read_conversation = read_conversation

        def _without_handoff(*args, **kwargs):
            return rewind_to(*args, **kwargs)[0], None

        repo._rewind_to = _without_handoff
        with pytest.raises(RewindRefused) as refused:
            repo.rewind_user_turn(W, -1)
        assert refused.value.reason == "handoff_lost"
        # Checked after the write: the rewind itself committed.
        assert (totals(backend, scope, W)).rewind_count == 1
        del repo._rewind_to

        # The target vanishes between the snapshot and the transaction: the active-id pin (and, ahead of it, a live lease) refuses before the missing target is looked up.
        repo.append(W, new("user", "u2", 2))
        repo.append(W, new("assistant", "a2", 3))

        def _then_wipe(sid, request=ConversationRequest()):
            snapshot = read_conversation(sid, request)
            if request.repair_alternation:  # the plan's last read
                repo.replace_transcript(W, [new("user", "fresh", 0)], ReplaceMode.DESTRUCTIVE)
            return snapshot

        repo.read_conversation = _then_wipe
        with pytest.raises(RewindRefused) as race:
            repo.rewind_user_turn(W, -1)
        assert (race.value.reason, race.value.entity) == ("history_changed", "transcript")
        assert backend.open(scope).coordination.try_acquire_turn_lease(session_id=W, holder="live-turn", ttl_seconds=300.0) is True
        with pytest.raises((TurnLeaseLost, CompressionInProgress)):
            repo.rewind_user_turn(W, -1)
        backend.open(scope).coordination.release_turn_lease(session_id=W, holder="live-turn")

        # The whole session vanishes (unadmitted discard): still the pin, not an unavailable target.
        def _then_discard(sid, request=ConversationRequest()):
            snapshot = read_conversation(sid, request)
            if request.repair_alternation:
                assert backend.open(scope).sessions.discard_unadmitted(W, media_root=MEDIA_ROOT) == (W,)
            return snapshot

        repo.read_conversation = _then_discard
        with pytest.raises(RewindRefused) as race:
            repo.rewind_user_turn(W, -1)
        assert (race.value.reason, race.value.entity) == ("history_changed", "transcript")

        # The target's live payload is edited in place between the snapshot and the write: target_changed, nothing rewound.
        backend.open(scope).sessions.create(NewSession(session_id=W, source="tui"))
        target_id = repo.append(W, new("user", "ask", 0))
        repo.append(W, new("assistant", "answer", 1))

        def _then_edit(sid, request=ConversationRequest()):
            snapshot = read_conversation(sid, request)
            if request.repair_alternation:
                assert repo.set_user_message_content(W, target_id, "edited ask") is True
            return snapshot

        repo.read_conversation = _then_edit
        with pytest.raises(RewindRefused) as refused:
            repo.rewind_user_turn(W, -1)
        assert refused.value.reason == "target_changed"
        assert [r.visibility for r in repo.read_transcript(W, TranscriptRead(view=TranscriptView.ALL))] == [RowVisibility.ACTIVE] * 2


# ── read_transcript (display paging) ──



def test_read_transcript_display_shows_each_carried_message_once(backend, scope_a):
    """CONTRACT: DISPLAY carries compaction history plus the live set, each logical message once with the live row preferred; ACTIVE is the live set only; the summary marker rides as ``_compressed_summary``."""
    store_a = backend.open(scope_a)
    lineage_script().apply(backend, scope_a)
    display = store_a.messages.read_transcript(CHILD, TranscriptRead(view=TranscriptView.DISPLAY))
    active = store_a.messages.read_transcript(CHILD, TranscriptRead())
    contents = [r.content for r in display]
    assert contents.count("child u2") == 1 and contents.count("child a2") == 1
    assert next(r for r in display if r.content == "child u2").visibility is RowVisibility.ACTIVE
    assert any(r.visibility is RowVisibility.COMPACTION_ARCHIVED for r in display)
    assert all(r.visibility is RowVisibility.ACTIVE for r in active)
    summary = next(r for r in active if r.content == "SUMMARY").as_wire_row()
    assert summary["_compressed_summary"] is True and list(summary)[-1] == "_compressed_summary"
    assert "_compressed_summary" not in next(r for r in active if r.content == "child u2").as_wire_row()


def test_read_transcript_unknown_session_and_request_validation(backend, scope_a):
    """CONTRACT: an unknown session reads as an empty page; keyset paging refuses latest/offset paging, the display view and the merged lineage at the record, and pages the ALL view."""
    store_a = backend.open(scope_a)
    assert store_a.messages.read_transcript(SessionId("no-such"), TranscriptRead()) == ()
    assert TranscriptRead(view=TranscriptView.ALL, after_id=MessageId(1)).after_id == MessageId(1)
    for bad in (dict(latest=True), dict(offset=1), dict(view=TranscriptView.DISPLAY), dict(include_lineage=True)):
        with pytest.raises(ValueError):
            TranscriptRead(after_id=MessageId(1), **bad)


def test_deactivate_message_rewinds_one_row_and_is_idempotent(backend, scope_a, scope_b):
    """CONTRACT: the addressed ACTIVE row leaves the active transcript (kept, REWOUND); repeating it, or naming a row of the session that is already inactive, is True and moves nothing; unknown and foreign rows are False; other scopes are untouched."""
    store_b = backend.open(scope_b)
    ids = _seed_dialogue(backend, scope_a)
    _seed_dialogue(backend, scope_b)
    repo = backend.open(scope_a).messages
    assert repo.deactivate_message(W, ids[2]) is True
    assert [m.content for m in repo.read_conversation(W)] == ["u1", "a1", "a2", "u3", "a3"]
    everything = repo.read_transcript(W, TranscriptRead(view=TranscriptView.ALL))
    assert [(m.content, m.visibility) for m in everything if m.id == ids[2]] == [("u2", RowVisibility.REWOUND)]
    assert repo.deactivate_message(W, ids[2]) is True
    assert repo.deactivate_message(SessionId("elsewhere"), ids[3]) is False
    assert repo.deactivate_message(W, MessageId(999_999)) is False
    assert [m.content for m in repo.read_conversation(W)] == ["u1", "a1", "a2", "u3", "a3"]
    assert len(store_b.messages.read_conversation(W)) == 6


def test_read_transcript_after_id_pages_active_rows_forward(backend, scope_a):
    """CONTRACT: ``after_id`` returns only the ACTIVE rows appended after the anchor, chronologically, honouring ``limit``; a rewound anchor still anchors; the newest anchor reads empty."""
    ids = _seed_dialogue(backend, scope_a)
    read = backend.open(scope_a).messages.read_transcript
    page = read(W, TranscriptRead(after_id=ids[1]))
    assert [r.content for r in page] == ["u2", "a2", "u3", "a3"]
    assert [r.content for r in read(W, TranscriptRead(after_id=ids[1], limit=2))] == ["u2", "a2"]
    assert read(W, TranscriptRead(after_id=ids[-1])) == ()
    rewind_from(backend.open(scope_a).messages, W, ids[4])
    assert [r.content for r in read(W, TranscriptRead(after_id=ids[1]))] == ["u2", "a2"]
    assert [r.content for r in read(W, TranscriptRead(after_id=ids[4]))] == []




# ── replay key order: the dict is serialised into provider requests ─────────────

# The per-row emission order: identity columns, display columns, row columns, assistant sidecars, then the result row's resolved tool-call uid.
_M1_REPLAY_ORDER = [
    "role", "content", DB_PERSISTED_MARKER_KEY, "_db_row_snapshot", "_row_id", "message_uid", "_tool_call_uids", "_absorbed_message_uids",
    "api_content", "display_kind", "display_metadata", "data_use", "_compressed_summary", "timestamp", "tool_call_id", "tool_name",
    "effect_disposition", "tool_calls", "message_id", "observed", "finish_reason", "reasoning", "reasoning_content", "reasoning_details",
    "codex_reasoning_items", "codex_message_items", "_tool_call_uid",
]


def test_replay_dict_keys_follow_the_emission_order():
    message = ConversationMessage(
        role="assistant", content="c", db_row_snapshot="s", row_id=MessageId(1), message_uid="u", tool_call_uids={"c1": "t1"},
        tool_call_uid="t0", absorbed_message_uids=("a",), api_content="api", display_kind="k", display_metadata={"m": 1}, data_use=("d",),
        compressed_summary=True, timestamp=1.0, tool_call_id="c0", tool_name="n", effect_disposition="e", tool_calls=({"id": "c1"},),
        message_id="p", observed=True, finish_reason="stop", reasoning="r", reasoning_content="rc", reasoning_details=[1],
        codex_reasoning_items=[2], codex_message_items=[3],
    )
    # CONTRACT: the keys come in the emission order above, so a request body serialised from these dicts has stable bytes.
    assert list(message.as_openai_message()) == _M1_REPLAY_ORDER


def test_a_result_row_replays_its_resolved_tool_call_uid_last(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("order-s1")
    store_a.sessions.create(NewSession(sid, "cli"))
    store_a.messages.append(sid, new("assistant", "", 1, tool_calls=({"id": "c1", "type": "function", "function": {"name": "grep", "arguments": "{}"}},)))
    store_a.messages.append(sid, new("tool", "out", 2, tool_call_id="c1", tool_name="grep"))
    replayed = [m.as_openai_message() for m in store_a.messages.read_conversation(sid)]
    # CONTRACT: a result appended alone has no stored uid; the replay resolves it from the assistant row and emits it last.
    assert list(replayed[1]) == ["role", "content", DB_PERSISTED_MARKER_KEY, "message_uid", "timestamp", "tool_call_id", "tool_name", "_tool_call_uid"]
    assert replayed[1]["_tool_call_uid"] == replayed[0]["_tool_call_uids"]["c1"]


def test_a_result_row_replays_its_stored_tool_call_uid_with_its_identity(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("order-s2")
    store_a.sessions.create(NewSession(sid, "cli"))
    store_a.messages.append(sid, new("assistant", "", 1, tool_calls=({"id": "c1", "type": "function", "function": {"name": "grep", "arguments": "{}"}},)))
    store_a.messages.append_batch(sid, [new("tool", "out", 2, tool_call_id="c1", tool_name="grep", tool_call_uid="tu-stored")])
    replayed = [m.as_openai_message() for m in store_a.messages.read_conversation(sid)]
    # CONTRACT: a stored uid is restored with the row's identity columns, ahead of the per-row columns.
    assert list(replayed[1]) == ["role", "content", DB_PERSISTED_MARKER_KEY, "message_uid", "_tool_call_uid", "timestamp", "tool_call_id", "tool_name"]
    assert replayed[1]["_tool_call_uid"] == "tu-stored"


def test_a_result_row_appended_alone_does_not_store_a_given_tool_call_uid(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("order-s3")
    store_a.sessions.create(NewSession(sid, "cli"))
    store_a.messages.append(sid, new("assistant", "", 1, tool_calls=({"id": "c1", "type": "function", "function": {"name": "grep", "arguments": "{}"}},)))
    store_a.messages.append(sid, new("tool", "out", 2, tool_call_id="c1", tool_name="grep", tool_call_uid="tu-given"))
    replayed = [m.as_openai_message() for m in store_a.messages.read_conversation(sid)]
    # CONTRACT: the single append does not store a tool-call uid, so the given uid is dropped and the replay resolves the assistant row's, emitted last; append_batch stores it (the case above).
    assert list(replayed[1]) == ["role", "content", DB_PERSISTED_MARKER_KEY, "message_uid", "timestamp", "tool_call_id", "tool_name", "_tool_call_uid"]
    assert replayed[1]["_tool_call_uid"] == replayed[0]["_tool_call_uids"]["c1"]


def test_the_replay_record_keeps_where_its_dict_carried_the_tool_call_uid():
    stored = {"role": "tool", "content": "out", DB_PERSISTED_MARKER_KEY: True, "message_uid": "u", "_tool_call_uid": "t", "timestamp": 1.0, "tool_call_id": "c1"}
    resolved = {"role": "tool", "content": "out", DB_PERSISTED_MARKER_KEY: True, "message_uid": "u", "timestamp": 1.0, "tool_call_id": "c1", "_tool_call_uid": "t"}
    for message in (stored, resolved):
        # CONTRACT: the record emits the keys in the order the dict carried them, either of the two uid positions.
        assert list(ConversationMessage.from_openai_message(message).as_openai_message()) == list(message)
    # CONTRACT: the position is not part of the record's value.
    assert ConversationMessage.from_openai_message(stored) == ConversationMessage.from_openai_message(resolved)


def test_a_single_append_drops_data_use_and_a_batch_folds_it(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("data-use-s1")
    store_a.sessions.create(NewSession(sid, "cli"))
    store_a.messages.append(sid, new("assistant", "alone", 1, display_metadata={"m": 1}, data_use=("d",)))
    store_a.messages.append_batch(sid, [new("assistant", "batched", 2, display_metadata={"m": 1}, data_use=("d",))])
    rows = store_a.messages.read_conversation(sid)
    # CONTRACT: the single append drops data_use; the batch folds it into display_metadata.
    assert [(m.display_metadata, m.data_use) for m in rows] == [({"m": 1}, ()), ({"m": 1, "data_use": ["d"]}, ("d",))]
