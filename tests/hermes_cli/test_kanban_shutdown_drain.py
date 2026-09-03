"""Regression tests for cooperative Kanban shutdown draining."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path: Path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    connection = kb.connect()
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

    marker = kb.prepare_shutdown_drain_marker()
    payload = kb.request_shutdown_drain(reason="gateway shutdown")

    assert marker == Path(payload["marker"])
    assert marker.parent == home / "kanban"
    assert marker.exists()
    assert kb.shutdown_drain_requested() is True
    assert json.loads(marker.read_text(encoding="utf-8"))["action"] == "pause-at-turn-boundary"


def test_pause_current_run_requeues_to_original_lane(conn, monkeypatch):
    task_id = kb.create_task(conn, title="pause me", assignee="builder")
    claimed = kb.claim_task(conn, task_id, claimer="builder:owner")
    assert claimed is not None
    assert claimed.current_run_id is not None

    assert kb.pause_task(
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

    assert kb.pause_task(
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
    marker = kb.prepare_shutdown_drain_marker()
    marker.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "builder:owner")

    from agent.kanban_stop import (
        kanban_shutdown_drain_requested,
        pause_current_kanban_run,
    )

    assert kanban_shutdown_drain_requested() is True
    assert pause_current_kanban_run(reason="gateway shutdown") is True
    assert kb.get_task(conn, task_id).status == "ready"
