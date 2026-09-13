"""Regression coverage for gateway restart handoff of Kanban workers (#77184)."""

import asyncio
import signal
from pathlib import Path

import pytest

from gateway.run_shutdown import GatewayShutdownMixin
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path: Path):
    connection = kbc.connect(db_path=tmp_path / "kanban.db")
    try:
        yield connection
    finally:
        connection.close()


def _running_task(conn, monkeypatch, *, heartbeat: int | None = 1, pid: int = 4242) -> str:
    task_id = kb.create_task(conn, title="restart handoff", assignee="worker")
    host = kb._claimer_id().split(":", 1)[0]
    assert kb.claim_task(conn, task_id, claimer=f"{host}:dispatcher") is not None
    kbd._set_worker_pid(conn, task_id, pid)
    conn.execute(
        "UPDATE tasks SET started_at = ?, last_heartbeat_at = ? WHERE id = ?",
        (1, heartbeat, task_id),
    )
    conn.execute(
        "UPDATE task_runs SET started_at = ?, last_heartbeat_at = ? WHERE task_id = ?",
        (1, heartbeat, task_id),
    )
    conn.commit()
    monkeypatch.setattr(kbd, "_pid_alive", lambda _pid: True)
    return task_id


def test_running_worker_scan_excludes_wedged_from_graceful_wait(conn, monkeypatch):
    fresh = _running_task(conn, monkeypatch, heartbeat=4990, pid=4242)
    wedged = _running_task(conn, monkeypatch, heartbeat=1, pid=4243)

    graceful = kbd.list_running_workers(conn, now=5000)
    assert [worker["task_id"] for worker in graceful] == [fresh]

    force = kbd.list_running_workers(conn, include_wedged=True, now=5000)
    assert {worker["task_id"] for worker in force} == {fresh, wedged}
    assert next(worker for worker in force if worker["task_id"] == wedged)["wedged"] is True


def test_restart_force_path_blocks_before_sigterm_without_retry_failure(conn, monkeypatch):
    task_id = _running_task(conn, monkeypatch, heartbeat=1, pid=4242)
    sent: list[tuple[int, int]] = []

    prepared = kbd.prepare_workers_for_gateway_restart(
        conn,
        signal_fn=lambda pid, sig: sent.append((pid, sig)),
    )

    assert len(prepared) == 1
    assert prepared[0]["task_id"] == task_id
    assert prepared[0]["reason"] == "gateway restart"
    assert prepared[0]["signaled"] is True
    row = conn.execute(
        "SELECT status, worker_pid, consecutive_failures FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert dict(row) == {
        "status": "blocked",
        "worker_pid": None,
        "consecutive_failures": 0,
    }
    assert sent == [(4242, signal.SIGTERM)]
    event = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    assert event["kind"] == "gateway_restart_worker"
    assert "gateway restart" in event["payload"]


def test_restart_wait_excludes_wedged_kanban_workers_and_interrupts_them():
    interrupted: list[str] = []

    class Runner(GatewayShutdownMixin):
        _restart_after_turn_timeout = 1800

        def _running_agent_count(self):
            return 0

        def _active_cron_job_count(self):
            return 0

        def _active_api_run_count(self):
            return 0

        def _active_deferred_agent_worker_count(self):
            return 0

        def _active_kanban_worker_count(self):
            return 1

        def _wedged_agent_count(self):
            return 0

        def _wedged_kanban_worker_count(self):
            return 1

        def _interrupt_kanban_workers_for_restart(self):
            interrupted.append("gateway restart")

        def _scale_to_zero_status(self, *args, **kwargs):
            pass

    result = asyncio.run(Runner()._await_active_work_before_restart())
    assert result is False
    assert interrupted == ["gateway restart"]


def test_active_work_count_includes_kanban_workers():
    class Runner(GatewayShutdownMixin):
        def _running_agent_count(self):
            return 1

        def _active_cron_job_count(self):
            return 2

        def _active_api_run_count(self):
            return 3

        def _active_deferred_agent_worker_count(self):
            return 4

        def _active_kanban_worker_count(self):
            return 5

    assert Runner()._active_work_count() == 15
