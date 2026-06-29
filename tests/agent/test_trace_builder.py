"""Tests for the derive-on-read trace builder and exporters.

Builds real sessions/messages in a temp SQLite store and asserts the
reconstructed span tree, then checks the OTLP/JSON and Chrome export shapes.
"""

from __future__ import annotations

import json

import pytest

from agent.trace_builder import (
    KIND_AGENT,
    KIND_LLM,
    KIND_TOOL,
    STATUS_ERROR,
    STATUS_OK,
    build_session_turns,
    build_trace,
)
from agent.trace_export import to_chrome_trace, to_otlp_json
from hermes_state import SessionDB

BASE = 1_700_000_000.0


def _tool_call(call_id: str, name: str, args: dict):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


@pytest.fixture
def db(tmp_path):
    store = SessionDB(db_path=tmp_path / "sessions.db")
    yield store
    store.close()


def _set_times(db: SessionDB, session_id: str, started: float, ended: float):
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
        (started, ended, session_id),
    )
    db._conn.commit()


def _build_parent_with_subagent(db: SessionDB):
    """Parent session that reads a file, then delegates to a subagent."""
    db.create_session("parent", "cli", model="test-model")
    db.append_message("parent", "user", "do the thing", timestamp=BASE)
    db.append_message(
        "parent",
        "assistant",
        "",
        tool_calls=[_tool_call("call_read", "read_file", {"path": "a.py"})],
        token_count=10,
        timestamp=BASE + 1,
    )
    db.append_message(
        "parent",
        "tool",
        '{"success": true}',
        tool_name="read_file",
        tool_call_id="call_read",
        timestamp=BASE + 2,
    )
    db.append_message(
        "parent",
        "assistant",
        "",
        tool_calls=[_tool_call("call_deleg", "delegate_task", {"goal": "sub"})],
        timestamp=BASE + 3,
    )
    # Subagent child session.
    db.create_session("child", "tool", parent_session_id="parent", model="test-model")
    db.append_message("child", "user", "sub goal", timestamp=BASE + 3.5)
    db.append_message(
        "child",
        "assistant",
        "working",
        tool_calls=[_tool_call("call_search", "search_files", {"q": "x"})],
        timestamp=BASE + 4,
    )
    db.append_message(
        "child",
        "tool",
        "results",
        tool_name="search_files",
        tool_call_id="call_search",
        timestamp=BASE + 4.5,
    )
    db.append_message("child", "assistant", "subagent done", timestamp=BASE + 5)
    _set_times(db, "child", BASE + 3.4, BASE + 5)
    # Delegate result lands back in the parent, then the parent wraps up.
    db.append_message(
        "parent",
        "tool",
        "subagent done",
        tool_name="delegate_task",
        tool_call_id="call_deleg",
        timestamp=BASE + 6,
    )
    db.append_message("parent", "assistant", "all done", timestamp=BASE + 7)
    _set_times(db, "parent", BASE, BASE + 7)


def test_build_trace_basic_shape(db):
    _build_parent_with_subagent(db)
    trace = build_trace(db, "parent")

    assert trace is not None
    kinds = [s.kind for s in trace.spans]
    assert kinds.count(KIND_AGENT) == 2  # parent + child
    assert kinds.count(KIND_TOOL) == 3  # read_file, delegate_task, search_files
    assert kinds.count(KIND_LLM) == 5  # 3 parent assistants + 2 child assistants

    root = next(s for s in trace.spans if s.span_id == trace.root_span_id)
    assert root.kind == KIND_AGENT
    assert root.parent_id is None
    assert root.session_id == "parent"


def test_tool_span_pairs_and_times(db):
    _build_parent_with_subagent(db)
    trace = build_trace(db, "parent")

    read = next(s for s in trace.spans if s.attributes.get("tool.name") == "read_file")
    assert read.start == pytest.approx(BASE + 1)
    assert read.end == pytest.approx(BASE + 2)
    assert read.status == STATUS_OK
    assert read.attributes["tool.call_id"] == "call_read"


def test_subagent_nested_under_delegate_span(db):
    _build_parent_with_subagent(db)
    trace = build_trace(db, "parent")

    delegate = next(
        s for s in trace.spans if s.attributes.get("tool.name") == "delegate_task"
    )
    child_root = next(
        s for s in trace.spans if s.kind == KIND_AGENT and s.session_id == "child"
    )
    assert child_root.parent_id == delegate.span_id
    # The delegate tool span should envelop the child's work.
    assert delegate.start <= child_root.start
    assert delegate.end >= BASE + 6 - 0.001


def test_no_subagents_flag_excludes_children(db):
    _build_parent_with_subagent(db)
    trace = build_trace(db, "parent", include_subagents=False)
    assert all(s.session_id == "parent" for s in trace.spans)


def test_error_status_detected(db):
    db.create_session("err", "cli", model="m")
    db.append_message("err", "user", "go", timestamp=BASE)
    db.append_message(
        "err",
        "assistant",
        "",
        tool_calls=[_tool_call("c1", "terminal", {"cmd": "boom"})],
        timestamp=BASE + 1,
    )
    db.append_message(
        "err",
        "tool",
        '{"error": "command failed", "success": false}',
        tool_name="terminal",
        tool_call_id="c1",
        timestamp=BASE + 2,
    )
    _set_times(db, "err", BASE, BASE + 2)

    trace = build_trace(db, "err")
    tool = next(s for s in trace.spans if s.kind == KIND_TOOL)
    assert tool.status == STATUS_ERROR


def test_missing_session_returns_none(db):
    assert build_trace(db, "nope") is None


def test_otlp_export_shape(db):
    _build_parent_with_subagent(db)
    trace = build_trace(db, "parent")
    doc = to_otlp_json(trace)

    spans = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == len(trace.spans)
    one = spans[0]
    assert len(one["traceId"]) == 32  # 16 bytes hex
    assert len(one["spanId"]) == 16  # 8 bytes hex
    keys = {a["key"] for a in one["attributes"]}
    assert "openinference.span.kind" in keys
    # All non-root spans carry a parentSpanId.
    assert any("parentSpanId" in s for s in spans)


def test_ended_at_orphan_reap_does_not_balloon_root(db):
    # ended_at sits hours after the last message (cleanup reaper); the AGENT
    # span must clamp to real activity, not the bogus ended_at.
    db.create_session("orphan", "tui", model="m")
    db.append_message("orphan", "user", "go", timestamp=BASE)
    db.append_message("orphan", "assistant", "done", timestamp=BASE + 10)
    db._conn.execute(
        "UPDATE sessions SET started_at=?, ended_at=?, end_reason=? WHERE id=?",
        (BASE, BASE + 18_000, "ws_orphan_reap", "orphan"),
    )
    db._conn.commit()

    trace = build_trace(db, "orphan")
    root = next(s for s in trace.spans if s.kind == KIND_AGENT)
    assert root.duration < 60  # ~10s of real work, not 5 hours


def test_build_session_turns_splits_and_tightens(db):
    # Two turns separated by a long idle gap; each turn-trace must be tight.
    db.create_session("multi", "tui", model="m")
    db.append_message("multi", "user", "first task", timestamp=BASE)
    db.append_message("multi", "assistant", "done one", timestamp=BASE + 5)
    # User walks away for 800s, then a second turn.
    db.append_message("multi", "user", "second task", timestamp=BASE + 805)
    db.append_message("multi", "assistant", "done two", timestamp=BASE + 810)
    db._conn.execute(
        "UPDATE sessions SET started_at=?, ended_at=? WHERE id=?",
        (BASE, BASE + 810, "multi"),
    )
    db._conn.commit()

    turns = build_session_turns(db, "multi")
    assert len(turns) == 2
    # Neither turn contains the 800s idle gap.
    assert all(t.duration < 60 for t in turns)
    assert turns[0].metadata["turn"] == 0
    assert turns[1].metadata["turn"] == 1


def test_async_delegation_completion_merges_into_dispatch_turn(db):
    # A background delegation dispatched in turn 0 re-enters as a synthetic
    # `[ASYNC DELEGATION …]` user message. It must NOT open its own turn — it
    # merges into the turn that spawned it, so the completion processing lands in
    # the same group as the delegate_task call.
    db.create_session("async", "tui", model="m")
    db.append_message("async", "user", "kick off background work", timestamp=BASE)
    db.append_message(
        "async",
        "assistant",
        "",
        tool_calls=[_tool_call("call_bg", "delegate_task", {"goal": "bg", "background": True})],
        timestamp=BASE + 1,
    )
    db.append_message(
        "async",
        "tool",
        '{"delegation_id": "d1"}',
        tool_name="delegate_task",
        tool_call_id="call_bg",
        timestamp=BASE + 2,
    )
    db.append_message("async", "assistant", "dispatched, carrying on", timestamp=BASE + 3)
    # Later, the background result re-enters as a synthetic continuation.
    db.append_message(
        "async",
        "user",
        "[ASYNC DELEGATION COMPLETE — d1]\nA background subagent finished.",
        timestamp=BASE + 50,
    )
    db.append_message("async", "assistant", "acting on the result", timestamp=BASE + 52)
    db._conn.execute(
        "UPDATE sessions SET started_at=?, ended_at=? WHERE id=?",
        (BASE, BASE + 52, "async"),
    )
    db._conn.commit()

    turns = build_session_turns(db, "async")
    assert len(turns) == 1  # not split by the re-injection
    # Label is the real prompt, not the async marker.
    root = next(s for s in turns[0].spans if s.span_id == turns[0].root_span_id)
    assert "kick off background work" in root.name
    assert "ASYNC DELEGATION" not in root.name


def test_important_notification_merges_into_turn(db):
    # Background-process notifications (`[IMPORTANT: …]`) are continuations too.
    db.create_session("notif", "tui", model="m")
    db.append_message("notif", "user", "run the build", timestamp=BASE)
    db.append_message("notif", "assistant", "started", timestamp=BASE + 1)
    db.append_message(
        "notif",
        "user",
        "[IMPORTANT: Background process p1 exited with code 0]",
        timestamp=BASE + 30,
    )
    db.append_message("notif", "assistant", "build finished", timestamp=BASE + 31)
    db._conn.execute(
        "UPDATE sessions SET started_at=?, ended_at=? WHERE id=?",
        (BASE, BASE + 31, "notif"),
    )
    db._conn.commit()

    turns = build_session_turns(db, "notif")
    assert len(turns) == 1


def test_to_dict_round_trips_shape(db):
    _build_parent_with_subagent(db)
    trace = build_trace(db, "parent")
    d = trace.to_dict()
    assert d["root_session_id"] == "parent"
    assert d["root_span_id"] == trace.root_span_id
    assert len(d["spans"]) == len(trace.spans)
    span = d["spans"][0]
    assert {"span_id", "parent_id", "kind", "start", "end", "duration", "status"} <= set(span)


def test_chrome_export_one_track_per_session(db):
    _build_parent_with_subagent(db)
    trace = build_trace(db, "parent")
    doc = to_chrome_trace(trace)

    complete = [e for e in doc["traceEvents"] if e["ph"] == "X"]
    assert len(complete) == len(trace.spans)
    tids = {e["tid"] for e in complete}
    assert len(tids) == 2  # parent + child lanes
    assert all(e["ts"] >= 0 for e in complete)
