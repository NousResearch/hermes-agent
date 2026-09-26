"""Ownership of ACP tool rows across progress completion and the step fallback.

The model tool-call id is the identity. A name-only FIFO can close every bubble
and still attach one call's result to another call.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from acp_adapter import events
from agent.turn_iteration_prep import _previous_tool_round


def test_imports_the_worktree_under_test():
    root = Path(__file__).resolve().parents[2]
    assert Path(events.__file__).resolve().is_relative_to(root)
    assert Path(_previous_tool_round.__code__.co_filename).resolve().is_relative_to(root)


@pytest.fixture
def bridge(monkeypatch):
    ids, meta, updates = {}, {}, []
    recorded = []
    real_complete = events.build_tool_complete

    def complete(*args, **kwargs):
        recorded.append((args, kwargs))
        return real_complete(*args, **kwargs)

    monkeypatch.setattr(events, "build_tool_complete", complete)
    monkeypatch.setattr(events, "_send_update", lambda *_args: updates.append(_args[-1]))
    progress = events.make_tool_progress_cb(None, "session", None, ids, meta)
    step = events.make_step_cb(None, "session", None, ids, meta)

    def start(name, args, call_id):
        progress("tool.started", name, None, args, tool_call_id=call_id)
        return ids[name][-1]

    return SimpleNamespace(
        ids=ids, meta=meta, updates=updates, recorded=recorded,
        progress=progress, step=step, start=start,
    )


def _round(rows):
    """rows: (call_id, name, arguments dict, result)."""
    calls, tools = [], []
    for call_id, name, args, result in rows:
        calls.append({
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
        tools.append({"role": "tool", "tool_call_id": call_id, "content": result})
    return _previous_tool_round([{"role": "assistant", "tool_calls": calls}, *tools])


def _visible(update) -> str:
    parts = []
    for block in getattr(update, "content", None) or []:
        inner = getattr(block, "content", None)
        text = getattr(inner, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _terminals(bridge):
    return {
        update.tool_call_id: update
        for update in bridge.updates
        if getattr(update, "status", None) in {"completed", "failed"}
    }


def test_replayed_batch_closes_each_same_name_call_with_its_own_result(bridge):
    ui_a = bridge.start("terminal", {"command": "echo A"}, "call-A")
    ui_b = bridge.start("terminal", {"command": "echo B"}, "call-B")
    bridge.progress("tool.completed", "terminal", None, None, result="OUTPUT_A", tool_call_id="call-A")
    bridge.step(2, _round([
        ("call-A", "terminal", {"command": "echo A"}, "OUTPUT_A"),
        ("call-B", "terminal", {"command": "echo B"}, "OUTPUT_B"),
    ]))

    closed = _terminals(bridge)
    assert _visible(closed[ui_a]) == "OUTPUT_A"
    assert _visible(closed[ui_b]) == "OUTPUT_B"
    assert closed[ui_a].status == closed[ui_b].status == "completed"
    assert [kwargs["function_args"] for _args, kwargs in bridge.recorded] == [
        {"command": "echo A"},
        {"command": "echo B"},
    ]
    assert bridge.ids == {} and bridge.meta == {}
    assert [args[0] for args, _kwargs in bridge.recorded].count(ui_a) == 1


def test_sibling_left_open_by_a_replay_closes_from_its_own_completion(bridge):
    ui_a = bridge.start("terminal", {"command": "echo A"}, "call-A")
    ui_b = bridge.start("terminal", {"command": "echo B"}, "call-B")
    bridge.progress("tool.completed", "terminal", None, None, result="OUTPUT_A", tool_call_id="call-A")
    bridge.step(2, _round([("call-A", "terminal", {"command": "echo A"}, "OUTPUT_A")]))
    assert list(_terminals(bridge)) == [ui_a]
    assert _visible(_terminals(bridge)[ui_a]) == "OUTPUT_A"
    assert list(bridge.ids["terminal"]) == [ui_b]

    bridge.progress("tool.completed", "terminal", None, None, result="OUTPUT_B", tool_call_id="call-B")
    closed = _terminals(bridge)
    assert (_visible(closed[ui_a]), _visible(closed[ui_b])) == ("OUTPUT_A", "OUTPUT_B")
    assert [kwargs["function_args"]["command"] for _args, kwargs in bridge.recorded] == ["echo A", "echo B"]


def test_out_of_order_error_stays_on_the_call_that_failed(bridge):
    ui_a = bridge.start("terminal", {"command": "echo A"}, "call-A")
    ui_b = bridge.start("terminal", {"command": "echo B"}, "call-B")
    bridge.progress(
        "tool.completed", "terminal", None, None,
        result="cancelled-B", is_error=True, tool_call_id="call-B",
    )
    bridge.progress(
        "tool.completed", "terminal", None, None,
        result="OUTPUT_A", is_error=False, tool_call_id="call-A",
    )

    closed = _terminals(bridge)
    assert (closed[ui_a].status, _visible(closed[ui_a])) == ("completed", "OUTPUT_A")
    assert (closed[ui_b].status, _visible(closed[ui_b])) == ("failed", "cancelled-B")
    assert [kwargs["function_args"]["command"] for _args, kwargs in bridge.recorded] == ["echo B", "echo A"]


def test_progress_closed_todo_still_emits_its_plan(bridge):
    result = json.dumps({"todos": [{"id": "one", "content": "Verify the fix", "status": "completed"}]})
    ui = bridge.start("todo", {}, "call-todo")
    bridge.progress("tool.completed", "todo", None, None, result=result, tool_call_id="call-todo")
    bridge.step(2, _round([("call-todo", "todo", {}, result)]))

    plans = [update for update in bridge.updates if getattr(update, "session_update", None) == "plan"]
    assert list(_terminals(bridge)) == [ui]
    assert len(plans) == 1
    assert plans[0].entries[0].content == "Verify the fix"
    assert plans[0].entries[0].status == "completed"


def test_unreported_same_name_sibling_is_abandoned(bridge):
    bridge.start("terminal", {"command": "echo A"}, "call-A")
    ui_b = bridge.start("terminal", {"command": "echo B"}, "call-B")
    bridge.progress("tool.completed", "terminal", None, None, result="OUTPUT_A", tool_call_id="call-A")
    bridge.step(2, _round([("call-A", "terminal", {"command": "echo A"}, "OUTPUT_A")]))

    assert events.flush_open_tool_calls(None, "session", None, bridge.ids, bridge.meta) == 1
    abandoned = [update for update in bridge.updates if getattr(update, "status", None) == "failed"]
    assert [update.tool_call_id for update in abandoned] == [ui_b]
    assert bridge.ids == {} and bridge.meta == {}


def test_fallback_without_progress_keeps_each_result(bridge):
    ui_a = bridge.start("terminal", {"command": "echo A"}, "call-A")
    ui_b = bridge.start("terminal", {"command": "echo B"}, "call-B")
    bridge.step(2, _round([
        ("call-A", "terminal", {"command": "echo A"}, "OUTPUT_A"),
        ("call-B", "terminal", {"command": "echo B"}, "OUTPUT_B"),
    ]))
    closed = _terminals(bridge)
    assert (_visible(closed[ui_a]), _visible(closed[ui_b])) == ("OUTPUT_A", "OUTPUT_B")


def test_anonymous_completion_does_not_take_an_identified_sibling(bridge):
    bridge.start("terminal", {"command": "echo A"}, "call-A")
    bridge.start("terminal", {"command": "echo B"}, "call-B")
    bridge.progress("tool.completed", "terminal", None, None, result="OUTPUT_A")
    assert _terminals(bridge) == {}
    assert len(bridge.ids["terminal"]) == 2
