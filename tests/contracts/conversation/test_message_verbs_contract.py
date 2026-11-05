"""Contract cases for the ``MessageRepository`` verbs the older suites do not reach: the detached completion note, kind-wide deactivation, the display count, the conversation fingerprint, the timeline and prompt pages, clone resolution, the held-row staleness probe, the rewind plan, the in-place prune rewrite, chunked batches and the logical readers other modules ran their own SQL for.

Every case seeds through public protocol operations and observes through public reads; refusals are proven non-mutating against re-read state.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from persistence import CompressionClosed, HeldRowRewritten, PruneRowStale, PruneRowUnresolved, RowVisibility, TurnLeaseLost
from persistence.core.identity import MessageId, SessionId
from persistence.conversation.messages import (
    HeldMessage,
    NewMessage,
    PrunedRowChange,
    RewindPlan,
    RewindTargetUnavailable,
    TranscriptRead,
    TranscriptView,
    Watermark,
)
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession
from tests.contracts.conftest import require_capability
from tests.contracts.conversation.test_messages import rewind_from, totals

S = SessionId("verbs-s1")
CHILD = SessionId("verbs-child")
OTHER = SessionId("verbs-s2")
CALL = {"id": "call-1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}


def ts(i: int) -> float:
    return datetime(2026, 9, 11, 10, 0, i, tzinfo=UTC).timestamp()


def msg(content, i: int = 0, role: str = "user", **kwargs) -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=ts(i), **kwargs)


def dialogue(backend, scope, sid: SessionId = S) -> list[MessageId]:
    """u1 a1 u2 a2 u3 a3 (ids in order)."""
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=sid, source="cli"))
    turns = [("user", "u1"), ("assistant", "a1"), ("user", "u2"), ("assistant", "a2"), ("user", "u3"), ("assistant", "a3")]
    return [store.messages.append(sid, msg(content, i, role)) for i, (role, content) in enumerate(turns)]


def visibility(backend, scope, sid: SessionId = S) -> list[tuple[MessageId, RowVisibility]]:
    store = backend.open(scope)
    return [(r.id, r.visibility) for r in store.messages.read_transcript(sid, TranscriptRead(view=TranscriptView.ALL))]


def test_active_message_ids_are_the_live_rows_in_order(backend, scope_a):
    ids = dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    assert repo.active_message_ids(S) == tuple(ids)
    rewind_from(repo, S, ids[4])
    assert repo.active_message_ids(S) == tuple(ids[:4])
    summary = repo.compact(S, [msg("SUMMARY", 9, "assistant")])
    assert repo.active_message_ids(S) == summary.row_ids
    assert repo.active_message_ids(SessionId("nobody")) == ()


def test_append_completion_note_is_once_per_admission_over_the_lineage(backend, scope_a):
    store_a = backend.open(scope_a)
    dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    note = repo.append_completion_note(S, "adm-1", "job finished", "completion_note", {"job": "j"})
    assert repo.append_completion_note(S, "adm-1", "other text", "other_kind", {}) == note
    row = repo.read_transcript(S)[-1]
    assert (row.id, row.role, row.content, row.display_kind) == (note, "user", "job finished", "completion_note")
    assert row.display_metadata == {"job": "j", "completion_admission": "adm-1"}
    # The compression continuation finds the parent's note; a new admission into the closed parent refuses.
    store_a.sessions.end(S, END_REASON_COMPRESSION)
    store_a.sessions.create(NewSession(session_id=CHILD, source="cli", parent_session_id=S))
    assert repo.append_completion_note(CHILD, "adm-1", "replay", "completion_note", {}) == note
    with pytest.raises(CompressionClosed) as closed:
        repo.append_completion_note(S, "adm-2", "late", "completion_note", {})
    assert closed.value.tip == CHILD
    assert repo.read_transcript(CHILD) == ()


def test_append_completion_note_refuses_a_new_note_under_a_live_turn(backend, scope_a):
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    note = repo.append_completion_note(S, "adm-1", "done", "completion_note", {})
    assert store_a.coordination.try_acquire_turn_lease(session_id=S, holder="turn", ttl_seconds=300.0) is True
    assert repo.append_completion_note(S, "adm-1", "done", "completion_note", {}) == note
    with pytest.raises(TurnLeaseLost):
        repo.append_completion_note(S, "adm-2", "new", "completion_note", {})
    assert len(repo.read_transcript(S)) == 7


def test_deactivate_messages_by_display_kind_rewinds_only_live_rows_of_the_kind(backend, scope_a):
    dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    marked = [repo.append(S, msg(f"switch {i}", 10 + i, display_kind="model_switch")) for i in range(2)]
    rewind_from(repo, S, marked[1])
    counted = totals(backend, scope_a, S)
    assert repo.deactivate_messages_by_display_kind(S, "model_switch") == 1
    assert dict(visibility(backend, scope_a))[marked[0]] is RowVisibility.REWOUND
    assert [r.content for r in repo.read_transcript(S)] == ["u1", "a1", "u2", "a2", "u3", "a3"]
    assert repo.deactivate_messages_by_display_kind(S, "model_switch") == 0
    assert repo.deactivate_messages_by_display_kind(S, "") == 0
    assert repo.deactivate_messages_by_display_kind(SessionId("nobody"), "model_switch") == 0
    # The counters keep the rewound row, as a deactivation writes no counter.
    assert totals(backend, scope_a, S) == counted


def test_display_count_counts_each_logical_message_once(backend, scope_a):
    dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    assert repo.display_count(S) == 6
    # A compaction that carries a3 forward archives the rest: the display set is the archive, the summary and a3 once.
    repo.compact(S, [msg("SUMMARY", 9, "assistant"), msg("a3", 5, "assistant")], carried_tail_count=1)
    display = repo.read_transcript(S, TranscriptRead(view=TranscriptView.DISPLAY))
    assert [r.content for r in display] == ["u1", "a1", "u2", "a2", "u3", "SUMMARY", "a3"]
    assert repo.display_count(S) == len(display) == 7
    assert repo.display_count(SessionId("nobody")) == 0


def test_conversation_revision_moves_on_every_transcript_change_and_only_then(backend, scope_a):
    ids = dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    first = repo.conversation_revision(S)
    assert (first.lineage_ids, first.active_count, first.max_active_id) == ((S,), 6, ids[-1])
    seen = {first.as_tuple()}

    def changed() -> None:
        current = repo.conversation_revision(S).as_tuple()
        assert current not in seen
        seen.add(current)

    repo.append(S, msg("u4", 6))
    changed()
    repo.set_user_message_content(S, ids[0], "u1 expanded")
    changed()
    assert repo.set_message_api_content(S, ids[2], "u2", "u2 [api]")
    changed()
    repo.deactivate_message(S, ids[1])
    changed()
    # Presentation-only writes leave the fingerprint alone.
    before = repo.conversation_revision(S)
    repo.set_reaction(S, ids[3], "👍", author="user")
    repo.set_latest_matching_message_display_kind(S, role="assistant", content="a3", display_kind="final")
    assert repo.conversation_revision(S) == before


def test_conversation_revision_with_lineage_covers_the_compression_ancestors(backend, scope_a):
    store_a = backend.open(scope_a)
    ids = dialogue(backend, scope_a)
    store_a.sessions.end(S, END_REASON_COMPRESSION)
    store_a.sessions.create(NewSession(session_id=CHILD, source="cli", parent_session_id=S))
    tip = store_a.messages.append(CHILD, msg("child ask", 10))
    alone = store_a.messages.conversation_revision(CHILD)
    merged = store_a.messages.conversation_revision(CHILD, include_lineage=True)
    assert (alone.lineage_ids, alone.active_count, alone.max_active_id) == ((CHILD,), 1, tip)
    assert (merged.lineage_ids, merged.active_count, merged.max_active_id) == ((S, CHILD), 7, max(tip, ids[-1]))


def test_read_session_timeline_pages_the_human_prompts(backend, scope_a):
    ids = dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    repo.set_user_message_content(S, ids[4], "  u3   spread\nout  ")
    first = repo.read_session_timeline(S, limit=2)
    assert [(e.row_id, e.preview, e.timestamp) for e in first.entries] == [(ids[0], "u1", ts(0)), (ids[2], "u2", ts(2))]
    assert (first.limit, first.after_row_id, first.returned, first.total, first.has_more, first.next_cursor) == (2, 0, 2, 3, True, ids[2])
    rest = repo.read_session_timeline(S, limit=2, after_row_id=first.next_cursor)
    assert [(e.row_id, e.preview) for e in rest.entries] == [(ids[4], "u3 spread out")]
    assert (rest.returned, rest.total, rest.has_more, rest.next_cursor) == (1, 3, False, None)
    empty = repo.read_session_timeline(SessionId("nobody"))
    assert (empty.entries, empty.total, empty.has_more) == ((), 0, False)


def test_read_prompt_page_starts_at_a_displayed_prompt(backend, scope_a):
    ids = dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    page = repo.read_prompt_page(S, ids[2])
    assert [r.id for r in page.messages] == ids[2:]
    assert (page.row_id, page.limit, page.returned, page.offset, page.total, page.has_older, page.has_newer) == (ids[2], 120, 4, 2, 6, True, False)
    narrow = repo.read_prompt_page(S, ids[0], limit=2)
    assert [r.id for r in narrow.messages] == ids[:2]
    assert (narrow.offset, narrow.has_older, narrow.has_newer) == (0, False, True)
    assert repo.read_prompt_page(S, ids[1]) is None  # an assistant row is no prompt
    rewind_from(repo, S, ids[4])
    assert repo.read_prompt_page(S, ids[4]) is None  # a rewound prompt is not displayed
    assert repo.read_prompt_page(OTHER, ids[0]) is None


def test_resolve_active_row_id_follows_a_compaction_clone(backend, scope_a):
    ids = dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    assert repo.resolve_active_row_id(S, ids[0]) == ids[0]
    # Nothing seen: every active row is cloned after the summary under a fresh id.
    repo.compact(S, [msg("SUMMARY", 9, "assistant")], watermark=Watermark(None))
    live = {r.content: r.id for r in repo.read_transcript(S)}
    assert repo.resolve_active_row_id(S, ids[2]) == live["u2"] != ids[2]
    rewind_from(repo, S, live["u3"])
    assert repo.resolve_active_row_id(S, ids[4]) is None  # its clone left the active set
    assert repo.resolve_active_row_id(S, live["u3"]) is None
    assert repo.resolve_active_row_id(OTHER, ids[0]) is None
    assert repo.resolve_active_row_id(S, MessageId(999_999)) is None


def test_rewritten_held_row_ids_is_the_compare_compact_runs_under_cas(backend, scope_a):
    ids = dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    held = (HeldMessage(role="user", content="u1", row_id=ids[0], persisted=True),
            HeldMessage(role="assistant", content="a1", row_id=ids[1], persisted=True),
            HeldMessage(role="user", content="unsaved", row_id=ids[2]),
            HeldMessage(role="user", content="gone", row_id=MessageId(999_999), persisted=True))
    assert repo.rewritten_held_row_ids(S, held) == ()
    assert repo.set_user_message_content(S, ids[0], "u1 edited")
    assert repo.rewritten_held_row_ids(S, held) == (ids[0],)
    before = visibility(backend, scope_a)
    with pytest.raises(HeldRowRewritten):
        repo.compact(S, [msg("SUMMARY", 9, "assistant")], held_for_cas=held)
    assert visibility(backend, scope_a) == before
    assert repo.compact(S, [msg("SUMMARY", 9, "assistant")], held_for_cas=held[1:]).active_count == 1
    assert repo.rewritten_held_row_ids(S, held) == ()  # inactive rows are not reported


def test_plan_user_turn_rewind_reads_and_checks_without_writing(backend, scope_a):
    ids = dialogue(backend, scope_a)
    repo = backend.open(scope_a).messages
    plan = repo.plan_user_turn_rewind(S, -2)
    assert isinstance(plan, RewindPlan)
    assert (plan.target_row_id, plan.expected_active_ids, plan.expected_target_content) == (ids[2], tuple(ids), "u2")
    assert (plan.user_ordinal, plan.durable_user, plan.scaffold, plan.live_text) == (1, (0, 2, 4), None, None)
    assert [m["content"] for m in plan.durable_prefix] == ["u1", "a1"] and plan.prefix == plan.durable_prefix
    assert plan.live_view["content"] == "u2"
    assert repo.plan_user_turn_rewind(S, -1, require_retryable=True).live_text == "u3"
    with pytest.raises(RewindTargetUnavailable):
        repo.plan_user_turn_rewind(S, 3)
    assert repo.active_message_ids(S) == tuple(ids)
    assert totals(backend, scope_a, S).rewind_count == 0


def _tool_turn(backend, scope) -> tuple[MessageId, MessageId]:
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=S, source="cli"))
    repo = backend.open(scope).messages
    repo.append(S, msg("run it", 0))
    call = repo.append(S, msg("", 1, "assistant", tool_calls=(CALL,)))
    result = repo.append(S, msg("long output", 2, "tool", tool_call_id="call-1", tool_name="terminal"))
    return call, result


def test_rewrite_pruned_rows_rewrites_in_place_and_keeps_the_old_body_archived(backend, scope_a):
    store_a = backend.open(scope_a)
    call, result = _tool_turn(backend, scope_a)
    repo = backend.open(scope_a).messages
    original = HeldMessage(role="tool", content="long output", tool_call_id="call-1", row_id=result, persisted=True)
    stub = msg("[pruned]", 2, "tool", tool_call_id="call-1", tool_name="terminal")
    written = msg("long output", 2, "tool", tool_call_id="call-1", tool_name="terminal")
    call_row = msg("", 1, "assistant", tool_calls=(CALL,))
    unchanged = PrunedRowChange(HeldMessage(role="assistant", content="", tool_calls=(CALL,), row_id=call, persisted=True), call_row, call_row)
    before = repo.conversation_revision(S)
    done = repo.rewrite_pruned_rows(S, [PrunedRowChange(original, stub, written), unchanged], model_config_patch={"_prune_anchor": 3})
    assert done.rewritten == 1 and done.snapshots[0] and done.snapshots[1] is None
    live = repo.read_transcript(S)
    assert [(r.id, r.content) for r in live] == [(live[0].id, "run it"), (call, ""), (result, "[pruned]")]
    archived = [r for r in repo.read_transcript(S, TranscriptRead(view=TranscriptView.ALL)) if r.visibility is RowVisibility.COMPACTION_ARCHIVED]
    assert [r.content for r in archived] == ["long output"]
    # The display read shows the stub once, in the message's place.
    assert [r.content for r in repo.read_transcript(S, TranscriptRead(view=TranscriptView.DISPLAY))] == ["run it", "", "[pruned]"]
    assert store_a.sessions.get(S).model_config["_prune_anchor"] == 3
    assert repo.conversation_revision(S) != before


def test_rewrite_pruned_rows_refuses_before_any_write(backend, scope_a):
    _call, result = _tool_turn(backend, scope_a)
    repo = backend.open(scope_a).messages
    before = visibility(backend, scope_a), [r.content for r in repo.read_transcript(S)]
    stub = msg("[pruned]", 2, "tool", tool_call_id="call-1", tool_name="terminal")
    fine = PrunedRowChange(HeldMessage(role="tool", content="long output", tool_call_id="call-1", row_id=result, persisted=True), stub,
                           msg("long output", 2, "tool", tool_call_id="call-1", tool_name="terminal"))
    user = PrunedRowChange(HeldMessage(role="user", content="run it", persisted=True), msg("[pruned]", 0), msg("run it", 0))
    stale = PrunedRowChange(HeldMessage(role="tool", content="other output", tool_call_id="call-1", row_id=result, persisted=True), stub,
                            msg("other output", 2, "tool", tool_call_id="call-1", tool_name="terminal"))
    with pytest.raises(PruneRowUnresolved) as unresolved:
        repo.rewrite_pruned_rows(S, [fine, user])
    assert not isinstance(unresolved.value, PruneRowStale)
    with pytest.raises(PruneRowStale):
        repo.rewrite_pruned_rows(S, [fine, stale])
    assert (visibility(backend, scope_a), [r.content for r in repo.read_transcript(S)]) == before
    with pytest.raises(ValueError, match="Session not found"):
        repo.rewrite_pruned_rows(OTHER, [], model_config_patch={"k": 1})


def test_chunked_batch_pairs_a_tool_result_with_its_call_across_chunks(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(session_id=S, source="cli"))
    batch = [msg("ask", 0), msg("", 1, "assistant", tool_calls=(CALL,)), msg("out", 2, "tool", tool_call_id="call-1")]
    result = store_a.messages.append_batch(S, batch, chunk_rows=2)
    _ask, assistant, tool = result.outcomes
    assert result.inserted == 3 and all(o.message_uid for o in result.outcomes)
    assert set(assistant.tool_call_uids) == {"call-1"}
    assert tool.tool_call_uid == assistant.tool_call_uids["call-1"]
    stored = store_a.messages.read_transcript(S)
    assert [r.message_uid for r in stored] == [o.message_uid for o in result.outcomes]
    assert json.loads(stored[1].tool_call_uids) == dict(assistant.tool_call_uids) and stored[2].tool_call_uid == tool.tool_call_uid


def test_terminal_tool_call_payloads_streams_assistant_calls_since_a_time(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.sessions.create(NewSession(session_id=S, source="cli"))
    store_a.sessions.create(NewSession(session_id=OTHER, source="cli"))
    repo = backend.open(scope_a).messages
    repo.append(S, msg("", 0, "assistant", tool_calls=({**CALL, "id": "old"},)))
    repo.append(S, msg("", 5, "assistant", tool_calls=(CALL,)))
    repo.append(OTHER, msg("", 6, "assistant", tool_calls=({**CALL, "id": "other"},)))
    repo.append(S, msg("out", 7, "tool", tool_call_id="call-1", tool_name="terminal"))
    repo.append(S, msg("plain", 8, "assistant"))
    payloads = list(repo.terminal_tool_call_payloads(since_ts=ts(5)))
    assert [[call["id"] for call in json.loads(p)] for p in payloads] == [["call-1"], ["other"]]
    assert list(store_b.messages.terminal_tool_call_payloads(since_ts=0.0)) == []


def test_blocked_tool_result_payloads_stream_marked_tool_rows_since_a_time(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.sessions.create(NewSession(session_id=S, source="cli"))
    store_a.sessions.create(NewSession(session_id=OTHER, source="cli"))
    repo = store_a.messages
    repo.append(S, msg("BLOCKED: User denied", 0, "tool", tool_call_id="old"))
    repo.append(S, msg("BLOCKED: User denied", 5, "tool", tool_call_id="denied"))
    repo.append(OTHER, msg("approval_required", 6, "tool", tool_call_id="pending"))
    repo.append(S, msg("blocked, in lower case", 7, "tool", tool_call_id="lower"))
    repo.append(S, msg("ok: done", 8, "tool", tool_call_id="ran"))
    repo.append(S, msg("BLOCKED without a call id", 9, "tool"))
    repo.append(S, msg("BLOCKED in an assistant reply", 10, "assistant"))
    repo.append(S, msg([{"type": "text", "text": "BLOCKED: structured"}], 11, "tool", tool_call_id="structured"))
    # CONTRACT: the rows come in no defined order, so they compare as a set.
    assert set(repo.blocked_tool_result_payloads(since_ts=ts(5))) == {
        ("denied", "BLOCKED: User denied"), ("pending", "approval_required"), ("lower", "blocked, in lower case")}
    assert list(store_b.messages.blocked_tool_result_payloads(since_ts=0.0)) == []


def test_has_any_message_and_spillover_reference_rows_read_the_whole_scope(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    repo = backend.open(scope_a).messages
    assert repo.has_any_message() is False
    store_a.sessions.create(NewSession(session_id=S, source="cli"))
    store_a.sessions.create(NewSession(session_id=OTHER, source="cli"))
    repo.append(S, msg("see /tmp/spillover/abc.txt", 0, "tool", tool_call_id="a"))
    repo.append(S, msg("nothing here", 1))
    repo.append(OTHER, msg("short", 2, api_content="full text in spillover/def.txt"))
    repo.append(OTHER, msg("spillover again", 3))
    assert repo.has_any_message() is True and store_b.messages.has_any_message() is False
    assert repo.spillover_reference_rows() == (
        ("see /tmp/spillover/abc.txt", None), ("short", "full text in spillover/def.txt"), ("spillover again", None))
    assert store_b.messages.spillover_reference_rows() == ()


def test_spillover_reference_rows_stop_at_the_row_cap(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(session_id=S, source="cli"))
    store_a.messages.append_batch(S, [msg(f"spillover/{i}.txt") for i in range(20002)])
    rows = store_a.messages.spillover_reference_rows()
    # CONTRACT: at most 20001 rows in no defined order, so only the count and membership are pinned.
    assert len(rows) == 20001 and len(set(rows)) == 20001
    assert set(rows) <= {(f"spillover/{i}.txt", None) for i in range(20002)}
