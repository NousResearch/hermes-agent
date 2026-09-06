"""Regression tests for cooperative Kanban shutdown draining."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_shutdown as ks


@pytest.fixture
def conn(tmp_path: Path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kbc._INITIALIZED_PATHS.clear()
    connection = kbc.connect()
    try:
        yield connection
    finally:
        connection.close()


def test_shutdown_drain_marker_is_unique_and_atomic(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DRAIN_MARKER", raising=False)

    marker = ks.prepare_shutdown_drain_marker()
    payload = ks.request_shutdown_drain(reason="gateway shutdown")

    assert marker == Path(payload["marker"])
    assert marker.parent == home / "kanban"
    assert marker.exists()
    assert ks.shutdown_drain_requested() is True
    assert json.loads(marker.read_text(encoding="utf-8"))["action"] == "pause-at-turn-boundary"


def test_pause_current_run_requeues_to_original_lane(conn, monkeypatch):
    task_id = kb.create_task(conn, title="pause me", assignee="builder")
    claimed = kb.claim_task(conn, task_id, claimer="builder:owner")
    assert claimed is not None
    assert claimed.current_run_id is not None

    assert ks.pause_task(
        conn,
        task_id,
        expected_run_id=claimed.current_run_id,
        claimer="builder:owner",
        reason="gateway shutdown",
    ) is True

    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "ready"
    assert task.current_run_id is None
    run = kb.latest_run(conn, task_id)
    assert run is not None
    assert run.outcome == "paused"
    event = [e for e in kb.list_events(conn, task_id) if e.kind == "paused"][-1]
    assert event.payload == {
        "reason": "gateway shutdown",
        "run_id": claimed.current_run_id,
        "retry_status": "ready",
    }


def test_pause_current_run_rejects_stale_owner(conn):
    task_id = kb.create_task(conn, title="owner check", assignee="builder")
    claimed = kb.claim_task(conn, task_id, claimer="builder:owner")
    assert claimed is not None

    assert ks.pause_task(
        conn,
        task_id,
        expected_run_id=claimed.current_run_id,
        claimer="builder:other",
        reason="gateway shutdown",
    ) is False
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "running"


def test_worker_pause_helper_uses_dispatcher_identity(conn, monkeypatch):
    task_id = kb.create_task(conn, title="worker pause", assignee="builder")
    claimed = kb.claim_task(conn, task_id, claimer="builder:owner")
    assert claimed is not None
    marker = ks.prepare_shutdown_drain_marker()
    marker.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "builder:owner")

    from agent.kanban_stop import (
        kanban_shutdown_drain_requested,
        pause_current_kanban_run,
    )

    assert kanban_shutdown_drain_requested() is True
    from types import SimpleNamespace
    from unittest.mock import Mock
    from agent.turn_final_response import finish_text_response

    agent = Mock(valid_tool_names=[], quiet_mode=True)
    agent._has_content_after_think_block.return_value = True
    agent._strip_think_blocks.side_effect = lambda text: text
    agent._build_assistant_message.return_value = {"role": "assistant", "content": "Checkpoint saved."}
    messages = [{"role": "user", "content": "Continue the task."}]
    verdict = finish_text_response(
        agent, assistant_message=SimpleNamespace(content="Checkpoint saved.", tool_calls=[]),
        response=None, finish_reason="stop", messages=messages, api_messages=[],
        conversation_history=[], api_call_count=1, user_message="Continue the task.",
        active_system_prompt="unchanged", final_response=None, _turn_exit_reason=None,
        _preflight_compression_blocked=False, codex_ack_continuations=0,
        truncated_response_parts=[], length_continue_retries=0,
        _pending_verification_response=None, _pending_verification_response_previewed=False,
    )
    assert verdict.action == "break"
    assert verdict._turn_exit_reason == "kanban_shutdown_paused"
    assert [m["role"] for m in messages] == ["user", "assistant"]
    agent._flush_messages_to_session_db.assert_called_once_with(messages, [])
    assert kb.goal_run_status(conn, task_id, claimed.current_run_id) == "paused"
    assert kb.get_task(conn, task_id).status == "ready"
