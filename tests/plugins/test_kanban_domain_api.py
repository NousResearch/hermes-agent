"""P2a plugin API parity: task_kind/parent_task_id surface + epic endpoints.

Attaches the dashboard plugin router to a bare FastAPI app and exercises the
new generic surface: create/patch serialization, board cards carrying the new
fields, and the public epic routes (list/show/create) with hierarchy.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


def _load_plugin_module():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_p2a_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_plugin_router():
    return _load_plugin_module().router


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


def test_create_task_surfaces_task_kind_and_parent(client):
    parent = client.post(
        "/api/plugins/kanban/tasks", json={"title": "parent"}
    ).json()["task"]
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "subtask", "task_kind": "subtask", "parent_task_id": parent["id"]},
    )
    assert r.status_code == 200, r.text
    task = r.json()["task"]
    assert task["task_kind"] == "subtask"
    assert task["parent_task_id"] == parent["id"]
    # Non-blocking containment: ready, not todo.
    assert task["status"] == "ready"


def test_create_task_default_task_kind(client):
    task = client.post("/api/plugins/kanban/tasks", json={"title": "plain"}).json()["task"]
    assert task["task_kind"] == "task"
    assert task["parent_task_id"] is None


def test_create_task_rejects_bad_kind(client):
    r = client.post("/api/plugins/kanban/tasks", json={"title": "x", "task_kind": "story"})
    assert r.status_code == 400


def test_create_task_accepts_epic_id(client):
    eid = client.post(
        "/api/plugins/kanban/epics", json={"title": "Epic"}
    ).json()["epic"]["id"]
    task = client.post(
        "/api/plugins/kanban/tasks", json={"title": "t", "epic_id": eid}
    ).json()["task"]
    assert task["epic_id"] == eid


def test_board_cards_carry_new_fields(client):
    parent = client.post("/api/plugins/kanban/tasks", json={"title": "p"}).json()["task"]
    child = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "c", "task_kind": "bug", "parent_task_id": parent["id"]},
    ).json()["task"]

    board = client.get("/api/plugins/kanban/board").json()
    cards = {t["id"]: t for col in board["columns"] for t in col["tasks"]}
    assert cards[parent["id"]]["task_kind"] == "task"
    assert cards[child["id"]]["task_kind"] == "bug"
    assert cards[child["id"]]["parent_task_id"] == parent["id"]


def test_epic_list_create_show_hierarchy(client):
    parent = client.post("/api/plugins/kanban/epics", json={"title": "Platform"}).json()["epic"]
    child = client.post(
        "/api/plugins/kanban/epics",
        json={"title": "Billing", "parent_epic_id": parent["id"]},
    ).json()["epic"]
    assert child["parent_epic_id"] == parent["id"]

    epics = client.get("/api/plugins/kanban/epics").json()["epics"]
    by_id = {e["id"]: e for e in epics}
    assert by_id[parent["id"]]["parent_epic_id"] == ""
    assert by_id[child["id"]]["parent_epic_id"] == parent["id"]

    shown = client.get(f"/api/plugins/kanban/epics/{child['id']}").json()
    assert shown["epic"]["id"] == child["id"]


def test_epic_create_rejects_unknown_parent(client):
    r = client.post(
        "/api/plugins/kanban/epics", json={"title": "orphan", "parent_epic_id": "epic_nope"}
    )
    assert r.status_code == 400


def test_epic_create_rejects_unknown_fields(client):
    r = client.post(
        "/api/plugins/kanban/epics",
        json={"title": "a", "id": "caller-controlled-id"},
    )
    assert r.status_code == 422


def test_epic_patch_updates_fields_and_clears_parent(client):
    parent = client.post(
        "/api/plugins/kanban/epics", json={"title": "Platform"}
    ).json()["epic"]
    child = client.post(
        "/api/plugins/kanban/epics",
        json={"title": "Billing", "parent_epic_id": parent["id"]},
    ).json()["epic"]

    r = client.patch(
        f"/api/plugins/kanban/epics/{child['id']}",
        json={
            "title": "Billing v2",
            "description": "Canonical hierarchy",
            "status": "done",
            "parent_epic_id": None,
        },
    )
    assert r.status_code == 200, r.text
    epic = r.json()["epic"]
    assert epic["title"] == "Billing v2"
    assert epic["description"] == "Canonical hierarchy"
    assert epic["status"] == "done"
    assert epic["parent_epic_id"] == ""


def test_epic_update_rejects_cycle(client):
    a = client.post("/api/plugins/kanban/epics", json={"title": "a"}).json()["epic"]
    b = client.post(
        "/api/plugins/kanban/epics", json={"title": "b", "parent_epic_id": a["id"]}
    ).json()["epic"]
    # b is under a; making a a child of b closes a hierarchy cycle.
    r = client.patch(
        f"/api/plugins/kanban/epics/{a['id']}",
        json={"parent_epic_id": b["id"]},
    )
    assert r.status_code == 400


def test_canonical_mutation_routes_enter_through_project_host():
    module = _load_plugin_module()
    direct_writes = (
        "kanban_db.create_task",
        "kanban_db.update_task",
        "kanban_db.assign_task",
        "kanban_db.transition_task",
        "kanban_db.add_comment",
        "kanban_db.link_tasks",
        "kanban_db.unlink_tasks",
        "kanban_db.create_epic",
        "kanban_db.update_epic",
    )
    for name in (
        "create_task",
        "update_task",
        "bulk_update",
        "add_comment",
        "add_link",
        "delete_link",
        "create_epic",
        "update_epic",
    ):
        source = inspect.getsource(getattr(module, name))
        assert "_host(" in source, name
        assert [token for token in direct_writes if token in source] == [], name


def test_canonical_mutation_models_reject_unknown_fields(client):
    task = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "strict fixture"},
    ).json()["task"]
    other = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "strict link fixture"},
    ).json()["task"]

    responses = (
        client.post(
            "/api/plugins/kanban/tasks",
            json={"title": "bad", "unknown": True},
        ),
        client.patch(
            f"/api/plugins/kanban/tasks/{task['id']}",
            json={"priority": 2, "unknown": True},
        ),
        client.post(
            "/api/plugins/kanban/tasks/bulk",
            json={"ids": [task["id"]], "unknown": True},
        ),
        client.post(
            f"/api/plugins/kanban/tasks/{task['id']}/comments",
            json={"body": "note", "unknown": True},
        ),
        client.post(
            "/api/plugins/kanban/links",
            json={
                "parent_id": task["id"],
                "child_id": other["id"],
                "unknown": True,
            },
        ),
    )
    assert [response.status_code for response in responses] == [422] * len(responses)
