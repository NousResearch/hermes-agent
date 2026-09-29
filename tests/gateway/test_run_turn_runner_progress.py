from __future__ import annotations

import queue
from types import SimpleNamespace

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
