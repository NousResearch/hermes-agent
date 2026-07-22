"""Regression tests for the Codex app-server Kanban terminal fence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import codex_runtime
from agent.codex_runtime import (
    make_codex_app_server_event_bridge,
    run_codex_app_server_turn,
)
from agent.transports.codex_app_server_session import TurnResult
from hermes_cli import kanban_db as kb


def _terminal_item(tool: str) -> dict:
    return {
        "type": "mcpToolCall",
        "id": f"{tool}-1",
        "server": "hermes-tools",
        "tool": tool,
        "status": "completed",
        "arguments": {},
        "result": {"content": [{"type": "text", "text": '{"ok": true}'}]},
        "error": None,
    }


def _completed_notification(tool: str) -> dict:
    return {"method": "item/completed", "params": {"item": _terminal_item(tool)}}


def _create_claimed_task(conn, title: str) -> tuple[str, int]:
    task_id = kb.create_task(
        conn,
        title=title,
        assignee="test-worker",
        initial_status="running",
    )
    claimed = kb.claim_task(conn, task_id, claimer="test-worker")
    assert claimed is not None
    assert claimed.current_run_id is not None
    return task_id, int(claimed.current_run_id)


def _make_terminal_board(
    tmp_path: Path,
    terminal_tool: str,
) -> tuple[Path, str, int]:
    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path)
    try:
        task_id, run_id = _create_claimed_task(conn, terminal_tool)
        if terminal_tool == "kanban_complete":
            assert kb.complete_task(
                conn,
                task_id,
                summary="finished",
                expected_run_id=run_id,
            )
        else:
            assert kb.block_task(
                conn,
                task_id,
                reason="needs input",
                kind="needs_input",
                expected_run_id=run_id,
            )
    finally:
        conn.close()
    return db_path, task_id, run_id


def _set_worker_identity(
    monkeypatch: pytest.MonkeyPatch,
    db_path: Path,
    task_id: str,
    run_id: int,
) -> None:
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))


@pytest.mark.parametrize("terminal_tool", ["kanban_complete", "kanban_block"])
def test_exact_terminal_run_interrupts_session_read_only(
    tmp_path,
    monkeypatch,
    terminal_tool,
):
    db_path, task_id, run_id = _make_terminal_board(tmp_path, terminal_tool)
    _set_worker_identity(monkeypatch, db_path, task_id, run_id)
    sqlite_connect = codex_runtime.sqlite3.connect
    opened: list[tuple[object, dict]] = []

    def recording_connect(database, **kwargs):
        opened.append((database, kwargs))
        return sqlite_connect(database, **kwargs)

    monkeypatch.setattr(codex_runtime.sqlite3, "connect", recording_connect)

    session = SimpleNamespace(request_interrupt=MagicMock())
    agent = SimpleNamespace(_codex_session=session)
    bridge = make_codex_app_server_event_bridge(agent)
    bridge(_completed_notification(terminal_tool))

    session.request_interrupt.assert_called_once_with()
    assert agent._codex_kanban_terminal_fenced is True
    assert opened
    assert str(opened[0][0]).endswith("?mode=ro")
    assert opened[0][1]["uri"] is True


def test_terminal_event_for_another_run_does_not_interrupt(
    tmp_path,
    monkeypatch,
):
    db_path, _, _ = _make_terminal_board(tmp_path, "kanban_complete")
    conn = kb.connect(db_path)
    try:
        active_task_id, active_run_id = _create_claimed_task(conn, "still active")
    finally:
        conn.close()
    _set_worker_identity(monkeypatch, db_path, active_task_id, active_run_id)

    session = SimpleNamespace(request_interrupt=MagicMock())
    agent = SimpleNamespace(_codex_session=session)
    make_codex_app_server_event_bridge(agent)(
        _completed_notification("kanban_complete")
    )

    session.request_interrupt.assert_not_called()
    assert getattr(agent, "_codex_kanban_terminal_fenced", False) is False


@pytest.mark.parametrize(
    "env_name,env_value",
    [
        ("HERMES_KANBAN_DB", ""),
        ("HERMES_KANBAN_TASK", ""),
        ("HERMES_KANBAN_RUN_ID", "not-an-integer"),
    ],
)
def test_unprovable_worker_identity_fails_closed(
    tmp_path,
    monkeypatch,
    env_name,
    env_value,
):
    db_path, task_id, run_id = _make_terminal_board(tmp_path, "kanban_complete")
    _set_worker_identity(monkeypatch, db_path, task_id, run_id)
    monkeypatch.setenv(env_name, env_value)

    session = SimpleNamespace(request_interrupt=MagicMock())
    agent = SimpleNamespace(_codex_session=session)
    make_codex_app_server_event_bridge(agent)(
        _completed_notification("kanban_complete")
    )

    session.request_interrupt.assert_not_called()


def test_failed_terminal_tool_callback_does_not_interrupt(tmp_path, monkeypatch):
    db_path, task_id, run_id = _make_terminal_board(tmp_path, "kanban_complete")
    _set_worker_identity(monkeypatch, db_path, task_id, run_id)
    note = _completed_notification("kanban_complete")
    note["params"]["item"]["error"] = {"message": "MCP delivery failed"}

    session = SimpleNamespace(request_interrupt=MagicMock())
    agent = SimpleNamespace(_codex_session=session)
    make_codex_app_server_event_bridge(agent)(note)

    session.request_interrupt.assert_not_called()


def test_terminal_fence_closes_session_and_suppresses_post_turn_work(
    tmp_path,
    monkeypatch,
):
    db_path, task_id, run_id = _make_terminal_board(tmp_path, "kanban_complete")
    _set_worker_identity(monkeypatch, db_path, task_id, run_id)

    class FakeSession:
        def __init__(self) -> None:
            self.bridge = None
            self.interrupt_requested = False
            self.closed = False

        def request_interrupt(self) -> None:
            self.interrupt_requested = True

        def run_turn(self, user_input):
            assert self.bridge is not None
            self.bridge(_completed_notification("kanban_complete"))
            # Simulate a provider race that reports a normal-looking result
            # even though the lifecycle callback requested an interrupt.
            return TurnResult(
                final_text="late provider text",
                projected_messages=[],
                tool_iterations=1,
                interrupted=False,
                error=None,
                turn_id="turn-1",
                thread_id="thread-1",
            )

        def close(self) -> None:
            self.closed = True

    session = FakeSession()
    agent = SimpleNamespace(
        _codex_session=session,
        _codex_kanban_terminal_fenced=False,
        _session_db=None,
        _iters_since_skill=0,
        _skill_nudge_interval=1,
        valid_tool_names={"skill_manage"},
        session_api_calls=0,
        context_compressor=SimpleNamespace(
            awaiting_real_usage_after_compression=False
        ),
        _sync_external_memory_for_turn=MagicMock(),
        _spawn_background_review=MagicMock(),
    )
    session.bridge = make_codex_app_server_event_bridge(agent)

    result = run_codex_app_server_turn(
        agent,
        user_message="finish the task",
        original_user_message="finish the task",
        messages=[],
        effective_task_id=task_id,
        should_review_memory=True,
    )

    assert session.interrupt_requested is True
    assert session.closed is True
    assert agent._codex_session is None
    agent._sync_external_memory_for_turn.assert_not_called()
    agent._spawn_background_review.assert_not_called()
    assert result["completed"] is True
    assert result["partial"] is False
    assert result["error"] is None
