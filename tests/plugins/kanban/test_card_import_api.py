from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


INTENT = "b" * 64


def _load_router():
    root = Path(__file__).resolve().parents[3]
    path = root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("card_import_api_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.router


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    app = FastAPI()
    app.include_router(_load_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _payload():
    return {
        "project": "kalibrio-aios",
        "card_id": "root-api-m1b",
        "intent_hash": INTENT,
        "steps": [
            {"step": "build", "title": "Build", "assignee": "worker"},
            {
                "step": "deliver",
                "title": "Deliver",
                "assignee": "operator",
                "parents": ["build"],
            },
        ],
    }


def test_api_generates_keys_and_retries_converge(client):
    first = client.post("/api/plugins/kanban/card-imports", json=_payload())
    assert first.status_code == 200, first.text
    second = client.post("/api/plugins/kanban/card-imports", json=_payload())
    assert second.status_code == 200, second.text
    assert second.json()["tasks"] == first.json()["tasks"]
    assert second.json()["created"] == []
    assert first.json()["task_rows"]["build"]["idempotency_key"] == (
        f"v1:kalibrio-aios:root-api-m1b:{INTENT}:build"
    )


def test_api_rejects_worker_chosen_idempotency_key(client):
    payload = _payload()
    payload["idempotency_key"] = "worker-picked"
    response = client.post("/api/plugins/kanban/card-imports", json=payload)
    assert response.status_code == 422

    response = client.post(
        "/api/plugins/kanban/tasks",
        json={
            "title": "manual bypass",
            "idempotency_key": f"v1:kalibrio-aios:root-api-m1b:{INTENT}:manual",
        },
    )
    assert response.status_code == 400

    payload = _payload()
    payload["steps"][0]["idempotency_key"] = "worker-picked"
    response = client.post("/api/plugins/kanban/card-imports", json=payload)
    assert response.status_code == 422


def test_bad_external_parent_leaves_no_partial_import(client):
    payload = _payload()
    payload["card_id"] = "root-api-rollback"
    payload["steps"][1]["external_parents"] = ["t_missing"]
    response = client.post("/api/plugins/kanban/card-imports", json=payload)
    assert response.status_code == 400

    board = client.get("/api/plugins/kanban/board").json()
    tasks = [task for column in board["columns"] for task in column["tasks"]]
    assert all(task.get("card_id") != "root-api-rollback" for task in tasks)


def test_observables_endpoint_uses_typed_pr_state(client):
    imported = client.post(
        "/api/plugins/kanban/card-imports", json=_payload()
    ).json()
    task_id = imported["tasks"]["build"]
    response = client.put(
        f"/api/plugins/kanban/tasks/{task_id}/observables",
        json={
            "generation": 2,
            "pr_head": "deadbeef",
            "pr_checks": {"quality": "success", "review": "failure"},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["observables"]["generation"] == 2
    assert body["observables"]["pr"] == {
        "head": "deadbeef",
        "checks": {"quality": "success", "review": "failure"},
    }
    assert len(body["fingerprint"]) == 64

    cleared = client.put(
        f"/api/plugins/kanban/tasks/{task_id}/observables",
        json={"pr_head": None},
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["observables"]["pr"]["head"] is None
