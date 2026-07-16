from collections import defaultdict, deque
from typing import Any, cast

from acp_adapter import events


def test_codex_progress_uses_stable_id_and_completes_started_card(monkeypatch):
    updates = []
    monkeypatch.setattr(
        events,
        "_send_update",
        lambda conn, session_id, loop, update: updates.append(update),
    )
    tool_call_ids = defaultdict(deque)
    tool_call_meta = {}
    callback = events.make_tool_progress_cb(
        cast(Any, object()),
        "session-1",
        cast(Any, object()),
        tool_call_ids,
        tool_call_meta,
    )

    callback(
        "tool.started",
        "terminal",
        "pwd",
        {"command": "pwd"},
        tool_call_id="codex_exec_cmd-1",
    )
    callback(
        "tool.completed",
        "terminal",
        None,
        {"command": "pwd"},
        result="ok",
        tool_call_id="codex_exec_cmd-1",
    )
    # Replayed and orphan completions must not create extra ACP cards.
    callback(
        "tool.completed",
        "terminal",
        None,
        {"command": "pwd"},
        result="duplicate",
        tool_call_id="codex_exec_cmd-1",
    )
    callback(
        "tool.completed",
        "terminal",
        None,
        {"command": "whoami"},
        result="orphan",
        tool_call_id="codex_exec_missing",
    )

    assert [update.tool_call_id for update in updates] == [
        "codex_exec_cmd-1",
        "codex_exec_cmd-1",
    ]
    assert updates[0].session_update == "tool_call"
    assert updates[1].session_update == "tool_call_update"
    assert updates[1].status == "completed"
    assert not tool_call_ids
    assert not tool_call_meta


def test_codex_progress_conflicting_completion_preserves_valid_start(monkeypatch):
    updates = []
    monkeypatch.setattr(
        events,
        "_send_update",
        lambda conn, session_id, loop, update: updates.append(update),
    )
    tool_call_ids = defaultdict(deque)
    tool_call_meta = {}
    callback = events.make_tool_progress_cb(
        cast(Any, object()),
        "session-1",
        cast(Any, object()),
        tool_call_ids,
        tool_call_meta,
    )

    callback(
        "tool.started",
        "terminal",
        "pwd",
        {"command": "pwd"},
        tool_call_id="stable-1",
    )
    callback(
        "tool.completed",
        "memory",
        None,
        {"action": "add"},
        result="wrong tool",
        tool_call_id="stable-1",
    )

    assert [update.session_update for update in updates] == ["tool_call"]
    assert list(tool_call_ids["terminal"]) == ["stable-1"]
    assert "stable-1" in tool_call_meta

    callback(
        "tool.completed",
        "terminal",
        None,
        {"command": "pwd"},
        result="ok",
        tool_call_id="stable-1",
    )

    assert [update.session_update for update in updates] == [
        "tool_call",
        "tool_call_update",
    ]
    assert updates[-1].tool_call_id == "stable-1"
    assert not tool_call_ids
    assert not tool_call_meta
