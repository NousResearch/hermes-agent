"""Contract cases for the stored-session reads behind ``session.transcript`` and the ``session.list`` picker fields: one read generation, the display-index pages and their durable keys, the compaction handoff lookups, the poller's cursor state, the lineages a transcript reads, the picker facts and the delegated-child lookup.

Every case runs each registered backend through public operations only.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import ConversationRequest, NewMessage, TranscriptView, Watermark
from persistence.conversation.sessions import NewSession, SessionLineageKind, SessionOrder, SessionQuery, SessionWindow
from persistence.core.identity import SessionId
from session_rules.turns import SUMMARY_PREFIX

S = SessionId("stored-reads-s1")


def ts(i: int) -> float:
    return datetime(2026, 11, 6, 10, 0, i, tzinfo=UTC).timestamp()


def _store(backend, scope, *texts, session_id=S, **fields):
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=session_id, source="cli", **fields))
    ids = [store.messages.append(session_id, NewMessage(role=("user", "assistant")[i % 2], content=text, timestamp=ts(i)))
           for i, text in enumerate(texts)]
    return store, ids


def _texts(page):
    return [paged.message.content for paged in page.messages]


def _display(store, session_id=S, lineage=False):
    request = ConversationRequest(view=TranscriptView.DISPLAY, include_lineage=lineage, include_row_ids=True)
    return [(m.role, m.content, m.row_id) for m in store.messages.read_conversation(session_id, request)]


# ── one read generation ──

def test_reads_inside_a_snapshot_see_one_generation(backend, scope_a):
    store, _ = _store(backend, scope_a, "q1", "a1")
    with store.messages.read_snapshot() as consistent:
        before = store.messages.display_count(S)
        store.messages.append(S, NewMessage(role="user", content="q2", timestamp=ts(5)))
        after = store.messages.display_count(S)
        if consistent.pinned:
            assert (after, consistent()) == (before, True)
        else:
            assert consistent() is False
        with store.messages.read_snapshot() as nested:
            assert nested.pinned is consistent.pinned
    assert store.messages.display_count(S) == before + 1


def test_a_snapshot_without_a_write_is_consistent(backend, scope_a):
    store, _ = _store(backend, scope_a, "q1", "a1")
    with store.messages.read_snapshot() as consistent:
        assert store.messages.display_count(S) == 2
        assert consistent() is True


# ── the display index ──

def test_a_fresh_segment_has_its_whole_display_index(backend, scope_a):
    store, _ = _store(backend, scope_a, "q1", "a1", "q2")
    assert store.messages.ensure_display_index(S) is True
    assert store.messages.display_index_complete(S) is True


def test_segment_pages_walk_the_display_index_both_ways(backend, scope_a):
    store, ids = _store(backend, scope_a, "m0", "m1", "m2", "m3", "m4")
    first = store.messages.transcript_segment_page(S, latest=False, after=None, count=2)
    assert (_texts(first), first.full) == (["m0", "m1"], True)
    assert [paged.durable_id for paged in first.messages] == ids[:2]
    assert [paged.message.row_id for paged in first.messages] == ids[:2]
    assert first.edge == (first.messages[-1].position, ids[1]) and first.start == first.messages[0].position
    second = store.messages.transcript_segment_page(S, latest=False, after=first.edge, count=2)
    assert _texts(second) == ["m2", "m3"]
    last = store.messages.transcript_segment_page(S, latest=False, after=second.edge, count=2)
    assert (_texts(last), last.full) == (["m4"], False)
    again = store.messages.transcript_segment_page(S, latest=False, after=second.edge, count=2, inclusive=True)
    assert _texts(again) == ["m3", "m4"]
    newest = store.messages.transcript_segment_page(S, latest=True, after=None, count=2)
    assert _texts(newest) == ["m3", "m4"] and newest.edge == (newest.messages[0].position, ids[3])
    older = store.messages.transcript_segment_page(S, latest=True, after=newest.edge, count=2)
    assert _texts(older) == ["m1", "m2"]
    empty = store.messages.transcript_segment_page(S, latest=False, after=last.edge, count=2)
    assert (empty.messages, empty.full, empty.edge, empty.start) == ((), False, None, None)


def test_a_carried_copy_keeps_its_original_durable_id_and_slot(backend, scope_a):
    store, ids = _store(backend, scope_a, "q1", "a1", "q2", "a2")
    first = store.messages.transcript_segment_page(S, latest=False, after=None, count=3)
    summary = NewMessage(role="user", content=f"{SUMMARY_PREFIX}\nthe first turn", timestamp=ts(10))
    carried = [NewMessage(role="user", content="q2", timestamp=ts(2)), NewMessage(role="assistant", content="a2", timestamp=ts(3))]
    store.messages.compact(S, [summary, *carried], watermark=Watermark(last_active=ids[-1]), carried_tail_count=2)
    page = store.messages.transcript_segment_page(S, latest=False, after=first.edge, count=5)
    assert _texts(page)[-1] == "a2" and "q2" not in _texts(page)
    assert page.messages[-1].durable_id == ids[3]
    whole = store.messages.transcript_segment_page(S, latest=False, after=None, count=10)
    assert [(m.content, p.durable_id) for p, m in ((p, p.message) for p in whole.messages) if m.content in ("q2", "a2")] == [
        ("q2", ids[2]), ("a2", ids[3])]


def test_display_generations_read_the_lineage_as_a_resume_shows_it(backend, scope_a):
    store, ids = _store(backend, scope_a, "q1", "a1", "q2", "a2")
    child = SessionId("stored-reads-child")
    store.lineage.publish_compression_child(
        S, CompressionChild(session_id=child, source="cli"),
        [NewMessage(role="user", content=f"{SUMMARY_PREFIX}\nearlier", timestamp=ts(20))],
        require_compression_lease=False, watermark=ids[1], watermark_ceiling=ids[3])
    lineage = store.lineage.resume_lineage_ids(child)
    assert lineage == (S, child)
    generations = store.messages.display_generations(child, lineage)
    assert [(m.message.role, m.message.content, m.message.row_id) for m in generations] == _display(store, child, lineage=True)
    durable = {m.message.content: m.durable_id for m in generations}
    assert (durable["q2"], durable["a2"]) == (ids[2], ids[3])
    assert all(m.position == m.durable_id for m in generations)


# ── compaction handoffs ──

def test_handoff_rows_name_the_rows_that_can_hold_a_handoff(backend, scope_a):
    store, ids = _store(backend, scope_a, "q1", f"{SUMMARY_PREFIX}\nwhat happened", "q2")
    found = store.messages.transcript_handoff_rows([*ids, ids[1]], [S])
    assert list(found) == [ids[1]]
    assert [(row.id, row.content) for row in found[ids[1]]] == [(ids[1], f"{SUMMARY_PREFIX}\nwhat happened")]
    assert store.messages.transcript_handoff_rows([], [S]) == {}


def test_a_handoff_shown_before_a_slot_is_found_by_its_uid(backend, scope_a):
    store, ids = _store(backend, scope_a, "q1", f"{SUMMARY_PREFIX}\nwhat happened", "q2", "a2")
    page = store.messages.transcript_segment_page(S, latest=False, after=None, count=4)
    slots = {paged.message.content: paged.position for paged in page.messages}
    carrier = store.messages.transcript_handoff_rows([ids[1]], [S])[ids[1]][0]
    assert carrier.message_uid
    handoff = (carrier.message_uid, carrier.content)
    shown = store.messages.transcript_handoffs_shown_before(S, slots["q2"], [handoff])
    assert [row.id for row in shown] == [ids[1]]
    assert store.messages.transcript_handoffs_shown_before(S, slots[carrier.content], [handoff]) == ()
    # A uid-less handoff matches only uid-less stored copies by content.
    assert store.messages.transcript_handoffs_shown_before(S, slots["q2"], [(None, carrier.content)]) == ()
    assert store.messages.transcript_handoffs_shown_before(S, slots["a2"], []) == ()


def test_tool_calls_before_a_row_are_read_nearest_first(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=S, source="cli"))
    calls = [{"id": f"c{i}", "type": "function", "function": {"name": "shell", "arguments": "{}"}} for i in range(3)]
    ids = [store.messages.append(S, NewMessage(role="assistant", content="", tool_calls=(call,), timestamp=ts(i)))
           for i, call in enumerate(calls)]
    result = store.messages.append(S, NewMessage(role="tool", content="ran", tool_call_id="c2", timestamp=ts(5)))
    found = store.messages.transcript_tool_calls_before([S], result, limit=2)
    assert [json.loads(raw_calls)[0]["id"] for raw_calls, _uids in found] == ["c2", "c1"]
    assert store.messages.transcript_tool_calls_before([S], ids[0]) == ()


# ── the poller's cursor state ──

def test_the_cursor_state_tells_an_append_from_every_other_change(backend, scope_a):
    store, ids = _store(backend, scope_a, "q1", "a1", "q2")
    state = store.messages.transcript_cursor_state(S, ids[-1], [S])
    store.messages.append(S, NewMessage(role="assistant", content="a2", timestamp=ts(5)))
    assert store.messages.transcript_cursor_state(S, ids[-1], [S]) == state
    store.messages.set_reaction(S, ids[1], "👍", author="user")
    reacted = store.messages.transcript_cursor_state(S, ids[-1], [S])
    assert reacted[:2] == state[:2] and reacted[2] != state[2]
    store.messages.deactivate_message(S, ids[0])
    rewound = store.messages.transcript_cursor_state(S, ids[-1], [S])
    assert (rewound[0], rewound[1]) == (state[0] - 1, state[1] + 1)


# ── lineages ──

def test_a_delegated_child_reads_its_own_lineage_only_once_marked(backend, scope_a):
    store, _ = _store(backend, scope_a, "q1")
    queued = SessionId("stored-reads-queued")
    store.sessions.create(NewSession(session_id=queued, source="subagent", parent_session_id=S))
    assert store.lineage.delegated_child_lineage_ids(queued) == (queued,)
    marked = SessionId("stored-reads-marked")
    store.sessions.create(NewSession(session_id=marked, source="subagent", parent_session_id=S,
                                     lineage_kind=SessionLineageKind.DELEGATE))
    assert store.lineage.delegated_child_lineage_ids(marked) == store.lineage.resume_lineage_ids(marked) == (marked,)
    assert store.lineage.resume_lineage_ids(SessionId("stored-reads-unknown")) == (SessionId("stored-reads-unknown"),)


def test_current_segments_follow_a_compression_to_its_tip(backend, scope_a):
    store, _ = _store(backend, scope_a, "q1", "a1")
    _store(backend, scope_a, "p1", session_id=SessionId("stored-reads-plain"))
    child = SessionId("stored-reads-tip")
    store.lineage.publish_compression_child(
        S, CompressionChild(session_id=child, source="cli"),
        [NewMessage(role="user", content=f"{SUMMARY_PREFIX}\nearlier", timestamp=ts(20))], require_compression_lease=False)
    assert store.lineage.current_segments([S, SessionId("stored-reads-plain"), SessionId("stored-reads-none")]) == {
        S: child, SessionId("stored-reads-plain"): SessionId("stored-reads-plain")}


# ── the picker ──

def test_the_workspace_filter_keeps_the_folder_and_its_repo(backend, scope_a):
    store = backend.open(scope_a)
    rows = {"repo": dict(cwd="/w/elsewhere", git_repo_root="/w/proj"), "under": dict(cwd="/w/proj/sub"),
            "at": dict(cwd="/w/proj"), "sibling": dict(cwd="/w/project"), "other": dict(cwd="/w/other")}
    for name, fields in rows.items():
        store.sessions.create(NewSession(session_id=SessionId(name), source="cli", **fields))
        store.messages.append(SessionId(name), NewMessage(role="user", content=name, timestamp=ts(1)))
    listed = store.sessions.list_sessions(SessionQuery(order=SessionOrder.LAST_ACTIVE, workspace_key="/w/proj"),
                                          SessionWindow(limit=20))
    assert {entry.session.session_id for entry in listed} == {"repo", "under", "at"}
    assert store.sessions.sessions_in_workspace([SessionId(n) for n in rows], "/w/proj") == {"repo", "under", "at"}


def test_picker_facts_read_roots_segments_and_the_last_shown_reply(backend, scope_a):
    store, _ = _store(backend, scope_a, "the first prompt", "the reply", "again", f"{SUMMARY_PREFIX}\nonly a handoff",
                      cwd="/w/proj", git_repo_root="/w/proj")
    store.sessions.set_pinned(S, True)
    _store(backend, scope_a, "just asked", session_id=SessionId("stored-reads-quiet"))
    facts = store.sessions.picker_session_facts([S, SessionId("stored-reads-none")], [S],
                                                [S, SessionId("stored-reads-quiet")])
    assert (facts.roots[S].cwd, facts.roots[S].git_repo_root) == ("/w/proj", "/w/proj")
    assert set(facts.roots) == {S}
    segment = facts.segments[S]
    assert (segment.preview, segment.pinned) == ("the first prompt", True) and segment.last_active is not None
    assert facts.last_replies == {S: "the reply", SessionId("stored-reads-quiet"): None}


def test_subagent_sessions_are_found_by_their_marker_newest_first(backend, scope_a):
    store = backend.open(scope_a)
    for i, (sid, source, marker) in enumerate((("sa-old", "subagent", "sa-1"), ("sa-new", "subagent", "sa-1"),
                                               ("sa-cli", "cli", "sa-1"), ("sa-other", "subagent", "sa-2"))):
        store.sessions.create(NewSession(session_id=SessionId(sid), source=source, model_config={"_subagent_id": marker}))
    found = store.sessions.subagent_session_ids("sa-1", "_subagent_id")
    assert set(found) == {"sa-old", "sa-new"}
    assert list(found) == sorted(found, key=lambda sid: (store.sessions.get(sid).started_at, sid), reverse=True)
    assert store.sessions.subagent_session_ids("sa-9", "_subagent_id") == ()
