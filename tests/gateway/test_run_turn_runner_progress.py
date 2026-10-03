from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from gateway.run_turn_runner import TurnRunner


def test_subagent_tool_uses_existing_tool_progress_lane(monkeypatch):
    emitted = []
    ctx = SimpleNamespace(
        log_queue=None,
        progress_queue=queue.Queue(),
        _run_still_current=lambda: True,
        long_tool_hint_fired=[True],
        _thinking_enabled=False,
        _native_slack_task_cards=False,
        tool_progress_enabled=True,
        progress_mode="all",
        last_tool=[None],
        repeat_count=[0],
        last_progress_msg=[None],
        agent_holder=[None],
    )
    runner = TurnRunner(None, ctx)
    monkeypatch.setattr(runner, "_progress_live_status", lambda *args: None)
    monkeypatch.setattr(runner, "_progress_build_message", lambda *args: "child tool")
    monkeypatch.setattr(runner, "_progress_emit", emitted.append)

    runner.progress_callback("subagent.tool", "terminal", "pwd", {"command": "pwd"})

    assert emitted == ["child tool"]


def _new_mode_runner(monkeypatch):
    emitted = []
    ctx = SimpleNamespace(
        log_queue=None,
        progress_queue=queue.Queue(),
        _run_still_current=lambda: True,
        long_tool_hint_fired=[True],
        _thinking_enabled=False,
        _native_slack_task_cards=False,
        tool_progress_enabled=True,
        progress_mode="new",
        last_tool=[None],
        repeat_count=[0],
        last_progress_msg=[None],
        agent_holder=[None],
    )
    runner = TurnRunner(None, ctx)
    monkeypatch.setattr(runner, "_progress_live_status", lambda *args: None)
    monkeypatch.setattr(runner, "_progress_build_message", lambda event, *args: event)
    monkeypatch.setattr(runner, "_progress_emit", emitted.append)
    return runner, emitted


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (
            [("subagent.tool", "terminal"), ("tool.started", "terminal")],
            ["terminal", "terminal"],
        ),
        (
            [("tool.started", "terminal"), ("subagent.tool", "terminal"), ("tool.started", "terminal")],
            ["terminal", "terminal"],
        ),
    ],
    ids=["child-does-not-suppress-parent", "child-does-not-unlock-parent-repeat"],
)
def test_new_mode_child_events_do_not_share_parent_dedup_slot(monkeypatch, events, expected):
    runner, emitted = _new_mode_runner(monkeypatch)

    for event_type, tool_name in events:
        runner.progress_callback(event_type, tool_name)

    assert emitted == expected
