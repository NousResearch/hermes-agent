"""Shared todo hydration and explicit snapshot-fold contracts."""

import json
from functools import partial
from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import repair_message_sequence
from agent.conversation_compression_todo import compaction_todo_snapshot, fold_todo_snapshot
from agent.todo_hydration import decode_todo_result, latest_todo_response
from tools.todo_tool import MAX_TODO_RESULT_CHARS, TODO_INJECTION_HEADER, TodoStore


@pytest.mark.parametrize("content,expected", [
    (None, None), ('{"todos":', None), ('{"other": []}', None),
    ('{"todos": null}', None), ('{"todos": []}', ([], 1)),
    ('{"todos": [], "revision": "7"}', ([], "7")),
    ('{"todos": []}' + ' ' * MAX_TODO_RESULT_CHARS, None),
])
def test_decode_todo_result(content, expected):
    assert decode_todo_result(content) == expected


def _pair(name="todo", call_id="t", revision=2):
    return [
        {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
         "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps({"todos": [], "revision": revision})},
    ]


@pytest.mark.parametrize("boundary", ["user", "system", "assistant"])
def test_history_reader_rejects_unpaired_result_without_losing_earlier_clear(boundary):
    history = _pair() + _pair(revision=3)
    history.insert(-1, {"role": boundary, "content": "boundary"})
    assert latest_todo_response(history) == ([], 2)
    assert latest_todo_response([history[-1]]) is None


def test_history_reader_skips_bad_newer_result_and_accepts_bridge_alias():
    history = _pair()
    newer = _pair(name="tool_call", revision=4)
    newer[0]["tool_calls"][0]["function"]["arguments"] = json.dumps({"name": "todo_list", "arguments": {}})
    assert latest_todo_response(newer) == ([], 4)
    newer[-1]["content"] = '{"todos":'
    assert latest_todo_response(history + newer) == ([], 2)


@pytest.mark.parametrize("authoritative", [False, True])
def test_explicit_clear_controls_stale_snapshot_removal_and_seam_repair(authoritative):
    rows = [
        {"role": "assistant", "content": "before", "reasoning": "retained"},
        {"role": "user", "content": TODO_INJECTION_HEADER + "\nstale", "_todo_snapshot_synthetic": True},
        {"role": "assistant", "content": "after"},
    ]
    fold_todo_snapshot(rows, None, authoritative=authoritative,
                       repair_sequence=partial(repair_message_sequence, None))
    if authoritative:
        assert len(rows) == 1
        assert "before" in rows[0]["content"] and "after" in rows[0]["content"]
        assert rows[0]["reasoning"] == "retained"
    else:
        assert len(rows) == 3


def test_snapshot_state_preserves_opaque_fallback_and_structured_clear():
    store = TodoStore()
    assert compaction_todo_snapshot(store) == (None, False)
    assert compaction_todo_snapshot(store, authoritative=True) == (None, True)
    opaque = TODO_INJECTION_HEADER + "\nlegacy rendered task"
    assert compaction_todo_snapshot(store, fallback=opaque) == (opaque, True)


def test_gateway_decodes_verified_result_without_agent_methods(monkeypatch):
    from gateway.compaction_todo_hydration import fold_todos_for_compaction, hydrate_todos_for_compaction
    from run_agent import AIAgent

    def forbidden(*args, **kwargs):
        raise AssertionError("gateway hydration must not borrow private agent methods")

    monkeypatch.setattr(AIAgent, "_latest_todo_response", forbidden)
    monkeypatch.setattr(AIAgent, "_tool_response_matches_todo_call", forbidden)
    monkeypatch.setattr(AIAgent, "_repair_message_sequence", forbidden)
    payload = json.dumps({"todos": [{"id": "task", "content": "Keep exact imperative", "status": "pending"}], "revision": 8})
    db = SimpleNamespace(get_latest_todo_result=lambda session_id: payload)
    rows = [{"role": "user", "content": "human request"}]
    folded = fold_todos_for_compaction([], rows, db, "session")
    assert folded is rows
    assert "Keep exact imperative" in rows[0]["content"]
    assert not rows[0].get("_todo_snapshot_synthetic")
    detached = SimpleNamespace(_todo_store=TodoStore(), _session_db=db, session_id="session")
    assert hydrate_todos_for_compaction(detached, []) is detached
    assert detached._todo_store.snapshot()["revision"] == 8


def test_agent_fold_does_not_resolve_repair_until_a_seam_is_exposed():
    from agent.conversation_compression import _fold_todo_snapshot

    class State:
        _todo_store = TodoStore()

        @property
        def _repair_message_sequence(self):
            raise AssertionError("repair is unnecessary when no synthetic seam is removed")

    rows = [{"role": "user", "content": "human request"}]
    _fold_todo_snapshot(State(), rows)
    assert rows == [{"role": "user", "content": "human request"}]
