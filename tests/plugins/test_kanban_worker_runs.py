"""Tests for kanban worker/runs read endpoints.

Covers:
  GET /workers/active
  GET /runs/{run_id}
  GET /runs/{run_id}/inspect
"""

from __future__ import annotations

import importlib.util
import secrets
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


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


def test_workers_active_with_running_task(client):
    """A running task with an open run row and worker_pid appears in the list."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="active-worker", assignee="alice")
        conn.execute(
            "UPDATE tasks SET status='running' WHERE id=?", (task_id,),
        )
        _insert_run(conn, task_id, worker_pid=12345)
    finally:
        conn.close()

    r = client.get("/api/plugins/kanban/workers/active")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    w = body["workers"][0]
    assert w["task_id"] == task_id
    assert w["worker_pid"] == 12345
    assert w["task_status"] == "running"
    assert w["task_title"] == "active-worker"
    assert w["task_assignee"] == "alice"


def test_workers_active_excludes_ended_runs(client):
    """Runs with ended_at set are excluded even if task is running."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="ended-run", assignee="bob")
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (task_id,))
        _insert_run(conn, task_id, worker_pid=99999, ended_at=int(time.time()) - 60)
    finally:
        conn.close()

    r = client.get("/api/plugins/kanban/workers/active")
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_workers_active_excludes_runs_without_pid(client):
    """Runs with no worker_pid are not considered active workers."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="no-pid", assignee="carol")
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (task_id,))
        _insert_run(conn, task_id, worker_pid=None)
    finally:
        conn.close()

    r = client.get("/api/plugins/kanban/workers/active")
    assert r.status_code == 200
    assert r.json()["count"] == 0


# ---------------------------------------------------------------------------
# GET /runs/{run_id}
# ---------------------------------------------------------------------------

def test_get_run_404_unknown_id(client):
    """Non-existent run_id returns 404."""
    r = client.get("/api/plugins/kanban/runs/999999")
    assert r.status_code == 404
    assert "999999" in r.json()["detail"]


def test_get_run_ok(client):
    """Existing run row returns 200 with expected shape."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="run-lookup", assignee="dave")
        run_id = _insert_run(conn, task_id, worker_pid=55555)
    finally:
        conn.close()

    r = client.get(f"/api/plugins/kanban/runs/{run_id}")
    assert r.status_code == 200
    body = r.json()
    assert "run" in body
    run = body["run"]
    assert run["id"] == run_id
    assert run["task_id"] == task_id
    assert run["worker_pid"] == 55555
    assert run["ended_at"] is None


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/inspect
# ---------------------------------------------------------------------------

def test_inspect_run_404(client):
    """Non-existent run_id returns 404."""
    r = client.get("/api/plugins/kanban/runs/888888/inspect")
    assert r.status_code == 404


def test_inspect_run_already_ended(client):
    """Run with ended_at set returns alive=false with reason."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="ended", assignee="eve")
        run_id = _insert_run(conn, task_id, worker_pid=11111, ended_at=int(time.time()) - 10)
    finally:
        conn.close()

    r = client.get(f"/api/plugins/kanban/runs/{run_id}/inspect")
    assert r.status_code == 200
    body = r.json()
    assert body["alive"] is False
    assert "ended" in body["reason"]


def test_inspect_run_no_pid(client):
    """Run with no worker_pid returns alive=false with reason."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="no-pid-inspect", assignee="frank")
        run_id = _insert_run(conn, task_id, worker_pid=None)
    finally:
        conn.close()

    r = client.get(f"/api/plugins/kanban/runs/{run_id}/inspect")
    assert r.status_code == 200
    body = r.json()
    assert body["alive"] is False
    assert "worker_pid" in body["reason"]


def test_inspect_run_dead_pid(client, monkeypatch):
    """Run with a non-existent PID returns alive=false via psutil.NoSuchProcess."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="dead-pid", assignee="grace")
        run_id = _insert_run(conn, task_id, worker_pid=999999)
    finally:
        conn.close()

    # Mock psutil to raise NoSuchProcess for any PID.
    mock_psutil = MagicMock()
    mock_psutil.NoSuchProcess = Exception
    mock_psutil.AccessDenied = PermissionError

    def _raise_no_such(*args, **kwargs):
        raise mock_psutil.NoSuchProcess("no such process")

    mock_psutil.Process = _raise_no_such

    # Patch the module-level _psutil in the loaded plugin module.
    plugin_mod_name = "hermes_dashboard_plugin_kanban_worker_runs_test"
    plugin_mod = sys.modules.get(plugin_mod_name)
    if plugin_mod is not None:
        monkeypatch.setattr(plugin_mod, "_psutil", mock_psutil)
    else:
        pytest.skip("plugin module not yet loaded")

    r = client.get(f"/api/plugins/kanban/runs/{run_id}/inspect")
    assert r.status_code == 200
    body = r.json()
    assert body["alive"] is False
    assert body["pid"] == 999999
    assert "not found" in body["reason"]


def test_inspect_run_live_pid(client, monkeypatch):
    """Run with a live PID returns alive=true with psutil fields."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="live-pid", assignee="heidi")
        run_id = _insert_run(conn, task_id, worker_pid=12345)
    finally:
        conn.close()

    # Build a realistic mock psutil.
    mock_psutil = MagicMock()
    mock_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    mock_psutil.AccessDenied = type("AccessDenied", (Exception,), {})

    fake_mem = MagicMock()
    fake_mem.rss = 1024 * 1024 * 50  # 50 MB
    fake_mem.vms = 1024 * 1024 * 200

    fake_proc = MagicMock()
    fake_proc.as_dict.return_value = {
        "cpu_percent": 3.5,
        "memory_info": fake_mem,
        "num_threads": 4,
        "status": "sleeping",
        "create_time": time.time() - 300,
        "cmdline": ["python", "-m", "hermes"],
    }
    fake_proc.num_fds.return_value = 12
    mock_psutil.Process.return_value = fake_proc

    plugin_mod_name = "hermes_dashboard_plugin_kanban_worker_runs_test"
    plugin_mod = sys.modules.get(plugin_mod_name)
    if plugin_mod is not None:
        monkeypatch.setattr(plugin_mod, "_psutil", mock_psutil)
    else:
        pytest.skip("plugin module not yet loaded")

    r = client.get(f"/api/plugins/kanban/runs/{run_id}/inspect")
    assert r.status_code == 200
    body = r.json()
    assert body["alive"] is True
    assert body["pid"] == 12345
    assert body["cpu_percent"] == 3.5
    assert body["memory_rss_bytes"] == fake_mem.rss
    assert body["num_threads"] == 4
    assert body["status"] == "sleeping"
