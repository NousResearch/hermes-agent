"""Restart adoption recovers orphan leases without taking over live stop intent."""
from __future__ import annotations

import sqlite3
import time
from types import SimpleNamespace

import psutil
import pytest

from hermes_cli import kanban_claims as claims
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as connect
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_worker_identity as identity
from hermes_cli import kanban_worker_recovery as recovery


@pytest.fixture
def survivor(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    db_path = tmp_path / "board.db"
    with connect.connect_closing(db_path) as conn:
        task_id = kb.create_task(conn, title="Survive dispatcher restart", assignee="default")
        owner_pid = 987654321
        old_lock = f"{kb._host_prefix()}{owner_pid}"
        task = claims.claim_task(conn, task_id, claimer=old_lock)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET worker_pid = 123, worker_pid_started_at = 1, "
            "worker_registered_at = ?, last_heartbeat_at = ?, worker_scope = ?, "
            "claim_expires = ?, reclaim_reserved_at = ? WHERE id = ?",
            (now, now, f"hermes-kanban-{task_id}-r{task.current_run_id}.scope", now - 1, now - 61, task_id),
        )
        original_process = psutil.Process

        def missing_owner(pid):
            if pid == owner_pid:
                raise psutil.NoSuchProcess(pid)
            return original_process(pid)

        monkeypatch.setattr(psutil, "Process", missing_owner)
        monkeypatch.setattr(identity, "_run_worker_alive", lambda row: (True, "scope_active"))
        yield conn, task, db_path


@pytest.mark.parametrize("intent", [None, "active_reservation", "own_worker_handoff", "scope_stopping", "stop_pending", "owner_live", "owner_unknown"])
def test_adoption_recovers_expired_orphan_reservation_but_preserves_pending_stop(survivor, monkeypatch, intent):
    conn, task, _ = survivor
    if intent in ("owner_live", "owner_unknown"):
        old_process = psutil.Process
        owner_pid = int(task.claim_lock.rsplit(":", 1)[1])

        def owner_state(pid):
            if pid != owner_pid:
                return old_process(pid)
            if intent == "owner_unknown":
                raise psutil.AccessDenied(pid)
            return SimpleNamespace(status=lambda: psutil.STATUS_RUNNING)

        monkeypatch.setattr(psutil, "Process", owner_state)
    elif intent == "active_reservation":
        conn.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) + 60, task.id))
    elif intent == "stop_pending":
        conn.execute("UPDATE task_runs SET stop_pending = 1 WHERE id = ?", (task.current_run_id,))
    elif intent is not None:
        kb._append_event(conn, task.id, intent, {}, run_id=task.current_run_id)
    before = tuple(conn.execute("SELECT * FROM tasks WHERE id = ?", (task.id,)).fetchone())
    adopted = recovery.adopt_surviving_running_workers(conn)
    if intent is not None:
        assert adopted == []
        assert tuple(conn.execute("SELECT * FROM tasks WHERE id = ?", (task.id,)).fetchone()) == before
        return
    assert adopted == [task.id]
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task.id,)).fetchone()
    assert row["current_run_id"] == task.current_run_id
    assert row["reclaim_reserved_at"] is None
    assert row["claim_lock"] == kb._claimer_id()
    assert dispatch.heartbeat_worker(conn, task.id, expected_run_id=task.current_run_id)
    assert claims.heartbeat_claim(conn, task.id, expected_run_id=task.current_run_id, claimer=kb._claimer_id())
    assert recovery.adopt_surviving_running_workers(conn) == []


@pytest.mark.parametrize("changed_field", ["current_run_id", "worker_pid_started_at", "worker_scope"])
def test_adoption_cannot_consume_liveness_evidence_for_replaced_attempt(survivor, monkeypatch, changed_field):
    conn, task, db_path = survivor
    new_value = "hermes-kanban-successor-r2.scope" if changed_field == "worker_scope" else 2
    events_before = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]

    def concurrent_takeover(row):
        with sqlite3.connect(db_path) as other:
            other.execute(f"UPDATE tasks SET {changed_field} = ? WHERE id = ?", (new_value, task.id))
        return True, "scope_active"

    monkeypatch.setattr(identity, "_run_worker_alive", concurrent_takeover)
    assert recovery.adopt_surviving_running_workers(conn) == []
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task.id,)).fetchone()
    assert row[changed_field] == new_value
    assert row["claim_lock"] == task.claim_lock
    assert row["reclaim_reserved_at"] is not None
    assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == events_before
