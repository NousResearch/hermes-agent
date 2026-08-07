from __future__ import annotations

import os
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.kanban_worker_heartbeat import (
    KanbanWorkerHeartbeat,
    worker_heartbeat_keepalive,
)
from hermes_cli import kanban_db as kb


def _wait_until(predicate, *, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_keepalive_heartbeats_without_foreground_activity():
    calls = 0

    def heartbeat() -> bool:
        nonlocal calls
        calls += 1
        return True

    keepalive = KanbanWorkerHeartbeat(heartbeat, interval_seconds=0.01)
    keepalive.start()
    try:
        assert _wait_until(lambda: calls >= 3)
    finally:
        keepalive.stop()

    stopped_at = calls
    time.sleep(0.05)
    assert calls == stopped_at
    assert not keepalive.is_alive


def test_keepalive_stops_when_task_is_no_longer_running():
    results = iter((True, False))
    calls = 0

    def heartbeat() -> bool:
        nonlocal calls
        calls += 1
        return next(results)

    keepalive = KanbanWorkerHeartbeat(heartbeat, interval_seconds=0.01)
    keepalive.start()

    assert _wait_until(lambda: not keepalive.is_alive)
    assert calls == 2


def test_keepalive_stops_when_worker_scope_errors():
    keepalive = None

    with pytest.raises(RuntimeError, match="model failed"):
        with worker_heartbeat_keepalive(
            interval_seconds=0.01,
            heartbeat_fn=lambda: True,
        ) as active_keepalive:
            keepalive = active_keepalive
            assert keepalive is not None
            assert _wait_until(lambda: keepalive.is_alive)
            raise RuntimeError("model failed")

    assert keepalive is not None
    assert not keepalive.is_alive


def test_chat_process_wraps_agent_run_in_worker_keepalive(monkeypatch):
    from agent import kanban_worker_heartbeat as heartbeat_module
    from hermes_cli import main

    events = []

    @contextmanager
    def fake_keepalive():
        events.append("heartbeat-started")
        try:
            yield None
        finally:
            events.append("heartbeat-stopped")

    def fake_cli_main(**_kwargs):
        events.append("agent-run")

    monkeypatch.setattr(heartbeat_module, "worker_heartbeat_keepalive", fake_keepalive)
    monkeypatch.setattr(main, "_resolve_use_tui", lambda _args: False)
    monkeypatch.setattr(main, "_apply_safe_mode", lambda _args: None)
    monkeypatch.setattr(main, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main, "_termux_should_prefetch_update_check", lambda: False)
    monkeypatch.setattr(main, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(main, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setitem(sys.modules, "cli", types.SimpleNamespace(main=fake_cli_main))

    args = SimpleNamespace(model=None, toolsets=None, query="work kanban task t_test")
    main.cmd_chat(args)

    assert events == ["heartbeat-started", "agent-run", "heartbeat-stopped"]


@pytest.fixture
def running_worker(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db(board="default")

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="long model turn", assignee="worker")
        assert kb.claim_task(conn, task_id, claimer=kb._claimer_id())
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb._set_worker_pid(conn, task_id, os.getpid())

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", str(task.claim_lock))
    return task_id


def test_real_keepalive_refreshes_claim_and_prevents_stale_reclaim(
    running_worker, monkeypatch
):
    task_id = running_worker
    old = int(time.time()) - (kb.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS + 60)
    with kb.connect() as conn:
        conn.execute(
            "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? WHERE id = ?",
            (int(time.time()) - 1, old, task_id),
        )
        conn.commit()

    with worker_heartbeat_keepalive(interval_seconds=0.01):
        def refreshed() -> bool:
            with kb.connect() as conn:
                task = kb.get_task(conn, task_id)
                return bool(task and task.last_heartbeat_at and task.last_heartbeat_at > old)

        assert _wait_until(refreshed)
        with kb.connect() as conn:
            assert kb.release_stale_claims(conn) == 0
            task = kb.get_task(conn, task_id)
            assert task is not None
            assert task.status == "running"


@pytest.mark.parametrize("final_status", ["blocked", "done"])
def test_real_keepalive_self_stops_after_task_finalization(
    running_worker, final_status
):
    task_id = running_worker
    with worker_heartbeat_keepalive(interval_seconds=0.01) as keepalive:
        assert keepalive is not None
        assert _wait_until(lambda: keepalive.is_alive)
        with kb.connect() as conn:
            conn.execute(
                "UPDATE tasks SET status = ? WHERE id = ?", (final_status, task_id)
            )
            conn.commit()
        assert _wait_until(lambda: not keepalive.is_alive)
