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
