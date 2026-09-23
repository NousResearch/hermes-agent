"""Tests for kanban worker/runs read endpoints.

Covers:
  GET /workers/active
  GET /runs/{run_id}
  GET /runs/{run_id}/inspect
  POST /runs/{run_id}/terminate
"""

from __future__ import annotations

import importlib.util
import secrets
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _load_plugin_router():
    """Dynamically load plugins/kanban/dashboard/plugin_api.py and return its router."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    mod_name = "hermes_dashboard_plugin_kanban_worker_runs_test"
    # Re-use a cached module if already loaded to avoid duplicate-router issues.
    if mod_name in sys.modules:
        return sys.modules[mod_name].router

    spec = importlib.util.spec_from_file_location(mod_name, plugin_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _insert_run(conn, task_id, *, worker_pid=None, ended_at=None):
    """Insert a task_runs row directly (bypassing claim machinery) and return run_id."""
    lock = secrets.token_hex(8)
    future = int(time.time()) + 3600
    cur = conn.execute(
        "INSERT INTO task_runs "
        "(task_id, status, claim_lock, claim_expires, worker_pid, started_at, ended_at) "
        "VALUES (?, 'running', ?, ?, ?, ?, ?)",
        (task_id, lock, future, worker_pid, int(time.time()), ended_at),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# GET /workers/active
# ---------------------------------------------------------------------------

def test_workers_active_empty_board(client):
    """Board with no running tasks returns an empty workers list."""
    r = client.get("/api/plugins/kanban/workers/active")
    assert r.status_code == 200
    body = r.json()
    assert body["workers"] == []
    assert body["count"] == 0
    assert "checked_at" in body


# ---------------------------------------------------------------------------
# GET /runs/{run_id}
# ---------------------------------------------------------------------------

def test_get_run_404_unknown_id(client):
    """Non-existent run_id returns 404."""
    r = client.get("/api/plugins/kanban/runs/999999")
    assert r.status_code == 404
    assert "999999" in r.json()["detail"]


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/inspect
# ---------------------------------------------------------------------------

def test_inspect_run_404(client):
    """Non-existent run_id returns 404."""
    r = client.get("/api/plugins/kanban/runs/888888/inspect")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /runs/{run_id}/terminate
# ---------------------------------------------------------------------------

def _setup_running_task_with_run(conn, *, title, assignee, worker_pid):
    """Create a task in 'running' state with a matching open task_runs row.

    Mirrors what dispatcher_claim does: stamps tasks.status='running',
    tasks.claim_lock, tasks.worker_pid; inserts task_runs row with the
    same claim_lock so reclaim_task's preconditions are satisfied.
    """
    task_id = kb.create_task(conn, title=title, assignee=assignee)
    lock = secrets.token_hex(8)
    future = int(time.time()) + 3600
    conn.execute(
        "UPDATE tasks SET status='running', claim_lock=?, "
        "claim_expires=?, worker_pid=? WHERE id=?",
        (lock, future, worker_pid, task_id),
    )
    cur = conn.execute(
        "INSERT INTO task_runs "
        "(task_id, status, claim_lock, claim_expires, worker_pid, started_at) "
        "VALUES (?, 'running', ?, ?, ?, ?)",
        (task_id, lock, future, worker_pid, int(time.time())),
    )
    conn.commit()
    return task_id, cur.lastrowid


def test_terminate_run_404_unknown_id(client):
    """POST to unknown run_id returns 404."""
    r = client.post(
        "/api/plugins/kanban/runs/777777/terminate",
        json={"reason": "test"},
    )
    assert r.status_code == 404
    assert "777777" in r.json()["detail"]


def _install_successor_run(conn, task_id, *, worker_pid, claim_lock):
    """Stamp a live successor claim without closing the prior task_runs row."""
    future = int(time.time()) + 3600
    cur = conn.execute(
        "INSERT INTO task_runs "
        "(task_id, status, claim_lock, claim_expires, worker_pid, started_at) "
        "VALUES (?, 'running', ?, ?, ?, ?)",
        (task_id, claim_lock, future, worker_pid, int(time.time())),
    )
    run_b = cur.lastrowid
    conn.execute(
        "UPDATE tasks SET claim_lock=?, claim_expires=?, worker_pid=?, "
        "current_run_id=? WHERE id=?",
        (claim_lock, future, worker_pid, run_b, task_id),
    )
    conn.commit()
    return run_b


def test_terminate_stale_run_does_not_reclaim_successor(client):
    """POST /runs/{A}/terminate must not kill or release successor B.

    Simulates a stale in-memory read of still-open run A after the task has
    already been re-claimed by B (new lock/pid/current_run_id, A.ended_at NULL).
    """
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        conn = kbc.connect()
        try:
            task_id, run_a = _setup_running_task_with_run(
                conn, title="successor-isolation", assignee="w", worker_pid=991001,
            )
            # Helper does not stamp current_run_id; CAS is meaningless without it.
            conn.execute(
                "UPDATE tasks SET current_run_id=? WHERE id=?",
                (run_a, task_id),
            )
            conn.commit()
            lock_b = f"{kb._host_prefix()}{secrets.token_hex(8)}"
            run_b = _install_successor_run(
                conn, task_id, worker_pid=sleeper.pid, claim_lock=lock_b,
            )
        finally:
            conn.close()

        r = client.post(
            f"/api/plugins/kanban/runs/{run_a}/terminate",
            json={"reason": "stale terminate A"},
        )
        assert r.status_code == 409, r.text

        conn = kbc.connect()
        try:
            row = conn.execute(
                "SELECT status, claim_lock, worker_pid, current_run_id "
                "FROM tasks WHERE id=?",
                (task_id,),
            ).fetchone()
            assert row["status"] == "running"
            assert row["claim_lock"] == lock_b
            assert row["worker_pid"] == sleeper.pid
            assert int(row["current_run_id"]) == int(run_b)
            a_row = conn.execute(
                "SELECT ended_at FROM task_runs WHERE id=?", (run_a,),
            ).fetchone()
            assert a_row["ended_at"] is None
        finally:
            conn.close()

        assert sleeper.poll() is None, "terminate(A) must not signal successor B"
    finally:
        if sleeper.poll() is None:
            sleeper.terminate()
        sleeper.wait(timeout=5)


