from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


@pytest.fixture()
def kanban_agent():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("kanban_heartbeat"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent


@pytest.fixture()
def kanban_worker_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="worker-test", assignee="test-worker")
        claimed = kb.claim_task(conn, task_id, claimer="worker-lock", ttl_seconds=30)
        assert claimed is not None
        run_id = claimed.current_run_id
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "worker-lock")
    return {"task_id": task_id, "run_id": run_id}


def test_touch_activity_updates_kanban_heartbeat(kanban_agent, kanban_worker_env):
    from hermes_cli import kanban_db as kb

    task_id = kanban_worker_env["task_id"]
    run_id = kanban_worker_env["run_id"]

    conn = kb.connect()
    try:
        conn.execute(
            "UPDATE tasks SET claim_expires = 1, last_heartbeat_at = NULL WHERE id = ?",
            (task_id,),
        )
        conn.execute(
            "UPDATE task_runs SET claim_expires = 1, last_heartbeat_at = NULL WHERE id = ?",
            (run_id,),
        )
        conn.commit()
    finally:
        conn.close()

    kanban_agent._touch_activity("still running")

    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        run = conn.execute(
            "SELECT claim_expires, last_heartbeat_at FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        events = kb.list_events(conn, task_id)
    finally:
        conn.close()

    heartbeat_events = [event for event in events if event.kind == "heartbeat"]
    assert task.last_heartbeat_at is not None
    assert task.claim_expires is not None and task.claim_expires > 1
    assert run["last_heartbeat_at"] == task.last_heartbeat_at
    assert run["claim_expires"] == task.claim_expires
    assert heartbeat_events
    assert heartbeat_events[-1].payload is None


def test_touch_activity_outside_worker_does_not_mutate_kanban(
    kanban_agent, kanban_worker_env, monkeypatch
):
    from hermes_cli import kanban_db as kb

    task_id = kanban_worker_env["task_id"]
    run_id = kanban_worker_env["run_id"]
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)

    conn = kb.connect()
    try:
        conn.execute(
            "UPDATE tasks SET claim_expires = 1, last_heartbeat_at = NULL WHERE id = ?",
            (task_id,),
        )
        conn.execute(
            "UPDATE task_runs SET claim_expires = 1, last_heartbeat_at = NULL WHERE id = ?",
            (run_id,),
        )
        conn.commit()
    finally:
        conn.close()

    kanban_agent._touch_activity("local-only activity")

    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        run = conn.execute(
            "SELECT claim_expires, last_heartbeat_at FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    finally:
        conn.close()

    assert task.last_heartbeat_at is None
    assert task.claim_expires == 1
    assert run["last_heartbeat_at"] is None
    assert run["claim_expires"] == 1


def test_touch_activity_rate_limits_auto_heartbeat(kanban_agent, monkeypatch):
    from tools import kanban_tools as kt

    calls: list[bool] = []
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_rate")
    times = iter([1000.0, 1030.0, 1061.0])
    monkeypatch.setattr(run_agent.time, "time", lambda: next(times))
    monkeypatch.setattr(
        kt,
        "heartbeat_current_worker_from_env",
        lambda: calls.append(True) or True,
    )

    kanban_agent._touch_activity("first")
    kanban_agent._touch_activity("second")
    kanban_agent._touch_activity("third")

    assert len(calls) == 2


def test_touch_activity_swallows_auto_heartbeat_failures(kanban_agent, monkeypatch):
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_failure")

    def _boom():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(kt, "heartbeat_current_worker_from_env", _boom)

    kanban_agent._touch_activity("still active")

    assert kanban_agent._last_activity_desc == "still active"
    assert kanban_agent._last_activity_ts > 0


def test_recent_auto_heartbeat_prevents_stale_reclaim(
    kanban_agent, kanban_worker_env
):
    from hermes_cli import kanban_db as kb

    task_id = kanban_worker_env["task_id"]
    run_id = kanban_worker_env["run_id"]
    old_started_at = 1

    conn = kb.connect()
    try:
        conn.execute(
            "UPDATE tasks SET started_at = ?, last_heartbeat_at = NULL WHERE id = ?",
            (old_started_at, task_id),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ?, last_heartbeat_at = NULL WHERE id = ?",
            (old_started_at, run_id),
        )
        conn.commit()
    finally:
        conn.close()

    kanban_agent._touch_activity("making progress")

    conn = kb.connect()
    try:
        reclaimed = kb.detect_stale_running(
            conn,
            stale_timeout_seconds=60,
            signal_fn=lambda *args, **kwargs: None,
        )
        task = kb.get_task(conn, task_id)
    finally:
        conn.close()

    assert reclaimed == []
    assert task.status == "running"
    assert task.last_heartbeat_at is not None
