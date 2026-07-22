"""Tests for the Profile Tasks dashboard plugin backend."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


def _load_plugin_module():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "profile-tasks" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_profile_tasks_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def profile_tasks_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "profiles" / "worker").mkdir(parents=True)
    (home / "profiles" / "reviewer").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def plugin_mod(profile_tasks_home):
    return _load_plugin_module()


@pytest.fixture
def client(plugin_mod):
    app = FastAPI()
    app.include_router(plugin_mod.router, prefix="/api/plugins/profile-tasks")
    return TestClient(app)


def test_profiles_and_boards_contract(client):
    r = client.get("/api/plugins/profile-tasks/profiles")
    assert r.status_code == 200, r.text
    data = r.json()
    names = {p["name"] for p in data["profiles"]}
    assert {"default", "worker", "reviewer"}.issubset(names)
    # Safe metadata only: do not leak local profile paths or env contents.
    assert all("path" not in p for p in data["profiles"])
    assert all("env" not in p for p in data["profiles"])

    r = client.get("/api/plugins/profile-tasks/boards")
    assert r.status_code == 200, r.text
    boards = r.json()["boards"]
    assert boards[0]["slug"] == "default"
    assert "db_path" not in boards[0]
    assert boards[0]["db_exists"] is True


def test_tasks_filter_by_profile_status_and_recent_done(client):
    conn = kb.connect()
    try:
        running = kb.create_task(conn, title="worker running", assignee="worker")
        blocked = kb.create_task(conn, title="worker blocked", assignee="worker")
        review = kb.create_task(conn, title="worker review", assignee="worker")
        done = kb.create_task(conn, title="worker done", assignee="worker")
        ready = kb.create_task(conn, title="worker ready", assignee="worker")
        other = kb.create_task(conn, title="reviewer running", assignee="reviewer")
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET status='running', started_at=?, claim_expires=?, last_heartbeat_at=? WHERE id=?",
            (now - 3600, now - 30, now - 2000, running),
        )
        conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (blocked,))
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (review,))
        conn.execute("UPDATE tasks SET status='done', completed_at=? WHERE id=?", (now - 5, done))
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (ready,))
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (other,))
        conn.commit()
    finally:
        conn.close()

    r = client.get("/api/plugins/profile-tasks/tasks", params={"profile": "worker"})
    assert r.status_code == 200, r.text
    cols = r.json()["columns"]
    assert {t["id"] for t in cols["running"]} == {running}
    assert {t["id"] for t in cols["blocked"]} == {blocked}
    assert {t["id"] for t in cols["review"]} == {review}
    assert {t["id"] for t in cols["recent_done"]} == {done}
    assert "ready" not in cols
    assert other not in {t["id"] for tasks in cols.values() for t in tasks}
    assert {w["kind"] for w in cols["running"][0]["warnings"]} >= {"stale_claim", "stale_heartbeat"}

    r = client.get(
        "/api/plugins/profile-tasks/tasks",
        params={"profile": "worker", "include_ready": "true"},
    )
    assert r.status_code == 200, r.text
    assert {t["id"] for t in r.json()["columns"]["ready"]} == {ready}


def test_tasks_response_excludes_sensitive_kanban_fields(client):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="secret task",
            body="FULL PRIVATE BODY",
            assignee="worker",
        )
        conn.execute(
            "UPDATE tasks SET status='done', result=?, completed_at=? WHERE id=?",
            ("FULL PRIVATE RESULT " * 40, int(time.time()), tid),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, summary, metadata, error) "
            "VALUES (?, 'worker', 'done', ?, ?, 'completed', ?, ?, ?)",
            (tid, int(time.time()) - 10, int(time.time()), "SAFE SUMMARY", '{"secret":"metadata"}', "RAW ERROR"),
        )
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, 'user', 'PRIVATE COMMENT', ?)",
            (tid, int(time.time())),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'x', ?, ?)",
            (tid, '{"private":"payload"}', int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()

    r = client.get("/api/plugins/profile-tasks/tasks", params={"profile": "worker"})
    assert r.status_code == 200, r.text
    task = r.json()["columns"]["recent_done"][0]
    forbidden = {"body", "result", "last_failure_error", "workspace_path", "claim_lock", "worker_pid", "model_override"}
    assert forbidden.isdisjoint(task.keys())
    assert "comments" not in task
    assert "events" not in task
    assert task["summary_preview"] == "SAFE SUMMARY"
    assert set(task["latest_run"]) == {
        "id", "task_id", "profile", "step_key", "status", "started_at", "ended_at", "outcome", "summary_preview",
    }
    assert "metadata" not in task["latest_run"]
    assert "error" not in task["latest_run"]


def test_invalid_profile_and_missing_board_db(client, profile_tasks_home):
    assert client.get("/api/plugins/profile-tasks/tasks", params={"profile": "missing"}).status_code == 404

    db_path = profile_tasks_home / "kanban.db"
    db_path.unlink()
    r = client.get("/api/plugins/profile-tasks/tasks", params={"profile": "worker"})
    assert r.status_code == 200, r.text
    assert r.json()["warnings"][0]["kind"] == "missing_board_db"


def test_readonly_connection_rejects_writes(plugin_mod, profile_tasks_home):
    with plugin_mod._open_board_readonly("default") as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO tasks (id, title, status, created_at) VALUES ('t_ro', 'nope', 'ready', ?)",
                (int(time.time()),),
            )

    # Confirm the failed write did not leave a row behind via a normal read.
    conn = kb.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM tasks WHERE id='t_ro'").fetchone()[0] == 0
    finally:
        conn.close()


def test_dashboard_bundle_registers_plugin_and_polls():
    repo_root = Path(__file__).resolve().parents[2]
    bundle = (repo_root / "plugins" / "profile-tasks" / "dashboard" / "dist" / "index.js").read_text()
    manifest = (repo_root / "plugins" / "profile-tasks" / "dashboard" / "manifest.json").read_text()

    assert 'REG.register("profile-tasks", ProfileTasks)' in bundle
    assert 'window.setInterval(loadTasks, 15000)' in bundle
    assert 'selectChangeHandler(setProfile)' in bundle
    assert '"path": "/profile-tasks"' in manifest
    assert '"api": "plugin_api.py"' in manifest
