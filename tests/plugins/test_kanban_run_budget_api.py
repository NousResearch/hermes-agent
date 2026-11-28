"""Supported REST creation/edit/readback against a temporary board."""
import importlib.util
from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    path = Path(__file__).resolve().parents[2] / "plugins/kanban/dashboard/plugin_api.py"
    spec = importlib.util.spec_from_file_location("kanban_budget_api_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router)
    with TestClient(app) as http:
        yield http


def test_create_edit_clear_and_readback(client):
    result = client.post("/tasks", json={"title": "budget", "max_worker_runs": 1})
    assert result.status_code == 200
    task = result.json()["task"]
    tid = task["id"]
    assert (task["max_worker_runs"], task["worker_runs_used"]) == (1, 0)
    with kbc.connect_closing() as conn:
        assert kb.claim_task(conn, tid)
    assert client.get(f"/tasks/{tid}").json()["task"]["worker_runs_used"] == 1
    assert client.patch(f"/tasks/{tid}", json={"max_worker_runs": 4}).status_code == 200
    assert client.get(f"/tasks/{tid}").json()["task"]["max_worker_runs"] == 4
    assert client.patch(f"/tasks/{tid}", json={"title": "renamed"}).status_code == 200
    assert client.get(f"/tasks/{tid}").json()["task"]["max_worker_runs"] == 4
    assert client.patch(f"/tasks/{tid}", json={"clear_max_worker_runs": True}).status_code == 200
    assert client.get(f"/tasks/{tid}").json()["task"]["max_worker_runs"] is None


@pytest.mark.parametrize("cap", [-1, True, 1.5, "2"])
def test_rejects_invalid_caps(client, cap):
    assert client.post("/tasks", json={"title": "bad", "max_worker_runs": cap}).status_code == 422
    tid = client.post("/tasks", json={"title": "valid"}).json()["task"]["id"]
    assert client.patch(f"/tasks/{tid}", json={"max_worker_runs": cap}).status_code == 422


def test_conflict_and_parked_status(client):
    tid = client.post("/tasks", json={"title": "parked", "max_worker_runs": 0}).json()["task"]["id"]
    assert client.patch(f"/tasks/{tid}", json={"max_worker_runs": 2, "clear_max_worker_runs": True}).status_code == 400
    assert client.patch(f"/tasks/{tid}", json={"max_worker_runs": 2}).json()["task"]["status"] == "blocked"
    assert client.patch(f"/tasks/{tid}", json={"clear_max_worker_runs": True}).json()["task"]["status"] == "blocked"


def test_dashboard_direct_requeue_parks_exhausted_card(client):
    tid = client.post("/tasks", json={"title": "finished", "max_worker_runs": 1}).json()["task"]["id"]
    with kbc.connect_closing() as conn:
        assert kb.claim_task(conn, tid)
        assert kb.complete_task(conn, tid, summary="Fixture work completed with isolated test evidence.", force=True)
    result = client.patch(f"/tasks/{tid}", json={"status": "ready"})
    assert result.status_code == 200
    assert result.json()["task"]["status"] == "blocked"
    detail = client.get(f"/tasks/{tid}").json()
    assert any(event["kind"] == "run_budget_exhausted" for event in detail["events"])
