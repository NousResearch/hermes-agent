"""Dashboard mutation-boundary coverage for ``on_kanban_task_updated``.

The dashboard plugin API's priority/title/body editors write task rows with
direct SQL, bypassing every ``kanban_db`` mutator — the exact gap the RFC
#58548 mutation-boundary review called out. These tests verify each
direct-SQL write path (single-task PATCH and bulk POST) reports through
``kanban_db.notify_task_updated`` with the right ``changed_fields``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.plugins import get_plugin_manager


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_task_updated_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
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


@pytest.fixture
def captured_updates():
    mgr = get_plugin_manager()
    events: list[dict] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("on_kanban_task_updated", []).append(
        lambda **kw: events.append(kw)
    )
    try:
        yield events
    finally:
        mgr._hooks = saved


def _make_task(title="t"):
    conn = kbc.connect()
    try:
        return kb.create_task(conn, title=title, assignee="alice")
    finally:
        conn.close()


def test_patch_priority_fires_task_updated(client, captured_updates):
    tid = _make_task()
    captured_updates.clear()
    r = client.patch(f"/api/plugins/kanban/tasks/{tid}", json={"priority": 5})
    assert r.status_code == 200
    assert len(captured_updates) == 1
    kw = captured_updates[0]
    assert kw["task_id"] == tid
    assert kw["changed_fields"] == ["priority"]
    assert kw["board"]


def test_patch_goal_configuration_reports_fields_and_locks_after_claim(client, captured_updates):
    tid = _make_task()
    assert client.get(f"/api/plugins/kanban/tasks/{tid}").json()["goal_configuration_locked"] is False
    captured_updates.clear()
    response = client.patch(
        f"/api/plugins/kanban/tasks/{tid}", json={"goal_mode": True, "goal_max_turns": 40},
    )
    assert response.status_code == 200, response.text
    assert captured_updates[-1]["changed_fields"] == ["goal_mode", "goal_max_turns"]

    disabled = client.patch(f"/api/plugins/kanban/tasks/{tid}", json={"goal_mode": False})
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["task"]["goal_mode"] is False
    assert disabled.json()["task"]["goal_max_turns"] == 40
    assert captured_updates[-1]["changed_fields"] == ["goal_mode"]

    cleared = client.patch(
        f"/api/plugins/kanban/tasks/{tid}",
        json={"goal_mode": False, "goal_max_turns": None},
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["task"]["goal_max_turns"] is None
    assert captured_updates[-1]["changed_fields"] == ["goal_mode", "goal_max_turns"]

    with kbc.connect() as conn:
        assert kb.claim_task(conn, tid, claimer="worker") is not None
    assert client.get(f"/api/plugins/kanban/tasks/{tid}").json()["goal_configuration_locked"] is True
    locked = client.patch(f"/api/plugins/kanban/tasks/{tid}", json={"goal_mode": False})
    assert locked.status_code == 409
    assert "cannot be changed after execution has started" in locked.json()["detail"]

    # A lock refusal happens before subsequent fields in the same PATCH mutate.
    combined = client.patch(
        f"/api/plugins/kanban/tasks/{tid}", json={"goal_mode": False, "title": "must not persist"},
    )
    assert combined.status_code == 409
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).title == "t"


def test_goal_patch_distinguishes_omission_clear_and_invalid_values(client):
    tid = _make_task()
    stored = client.patch(f"/api/plugins/kanban/tasks/{tid}", json={"goal_max_turns": 31})
    assert stored.status_code == 200

    # Unrelated PATCH omits goal settings and preserves the stored budget.
    omitted = client.patch(f"/api/plugins/kanban/tasks/{tid}", json={"title": "renamed"})
    assert omitted.status_code == 200
    assert omitted.json()["task"]["goal_max_turns"] == 31

    for invalid in (0, -1, 1.5, True, "bad", [], {}):
        response = client.patch(f"/api/plugins/kanban/tasks/{tid}", json={"goal_max_turns": invalid})
        assert response.status_code == 400, (invalid, response.text)
    assert client.patch(f"/api/plugins/kanban/tasks/{tid}", json={"priority": "bad"}).status_code == 422
    assert client.patch(
        f"/api/plugins/kanban/tasks/{tid}", json={"goal_mode": False, "goal_max_turns": 7},
    ).status_code == 400

    cleared = client.patch(f"/api/plugins/kanban/tasks/{tid}", json={"goal_max_turns": None})
    assert cleared.status_code == 200
    assert cleared.json()["task"]["goal_max_turns"] is None


def test_goal_patch_precedes_dispatchable_assignment_and_status(client, monkeypatch):
    """Combined PATCHes persist the launch contract before a dispatcher can claim."""
    with kbc.connect() as conn:
        blocked_id = kb.create_task(conn, title="blocked", assignee="worker", initial_status="blocked")
        unassigned_id = kb.create_task(conn, title="unassigned")

    real_unblock = kb.unblock_task
    real_assign = kb.assign_task

    def unblock_then_claim(conn, task_id, *args, **kwargs):
        ok = real_unblock(conn, task_id, *args, **kwargs)
        if ok:
            assert kb.claim_task(conn, task_id, claimer="status-race") is not None
        return ok

    def assign_then_claim(conn, task_id, profile, *args, **kwargs):
        ok = real_assign(conn, task_id, profile, *args, **kwargs)
        if ok:
            assert kb.claim_task(conn, task_id, claimer="assignment-race") is not None
        return ok

    monkeypatch.setattr(kb, "unblock_task", unblock_then_claim)
    monkeypatch.setattr(kb, "assign_task", assign_then_claim)

    for task_id, patch in (
        (blocked_id, {"status": "ready", "goal_mode": True, "goal_max_turns": 7}),
        (unassigned_id, {"assignee": "worker", "goal_mode": True, "goal_max_turns": 9}),
    ):
        response = client.patch(f"/api/plugins/kanban/tasks/{task_id}", json=patch)
        assert response.status_code == 200, response.text
        with kbc.connect() as conn:
            task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.goal_mode is True
        assert task.goal_max_turns == patch["goal_max_turns"]


def test_bulk_priority_fires_task_updated_per_task(client, captured_updates):
    tid1 = _make_task("a")
    tid2 = _make_task("b")
    captured_updates.clear()
    r = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={"ids": [tid1, tid2], "priority": 3},
    )
    assert r.status_code == 200
    assert all(entry["ok"] for entry in r.json()["results"])
    fired = {kw["task_id"]: kw for kw in captured_updates}
    assert set(fired) == {tid1, tid2}
    assert all(kw["changed_fields"] == ["priority"] for kw in fired.values())
