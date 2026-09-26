"""Public-route regressions for model/provider payload validation (107718 K16)."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def board_client(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from plugins.kanban.dashboard.plugin_api import router

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Override fixture", model_override="original-model", provider_override="original-provider")
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client, task_id, kb, kbc


def send(client, task_id, surface, payload):
    if surface == "patch":
        return client.patch(f"/tasks/{task_id}", json=payload)
    return client.post("/tasks/bulk", json={"ids": [task_id], **payload})


@pytest.mark.parametrize("surface", ["patch", "bulk"])
@pytest.mark.parametrize("extra", [{}, {"priority": 123}, {"model_override": " "}, {"clear_model_override": True}])
def test_provider_only_rejected_before_any_mutation(board_client, surface, extra):
    client, task_id, kb, kbc = board_client
    with kbc.connect() as conn:
        before_events = kb.list_events(conn, task_id)
    response = send(client, task_id, surface, {"provider_override": "fixture-provider", **extra})
    assert response.status_code == 400, response.text
    assert "provider_override requires a model_override" in response.json()["detail"]
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert (task.model_override, task.provider_override, task.priority) == ("original-model", "original-provider", 0)
        assert kb.list_events(conn, task_id) == before_events


@pytest.mark.parametrize("surface", ["patch", "bulk"])
@pytest.mark.parametrize("payload, expected", [
    ({}, ("original-model", "original-provider")),
    ({"provider_override": None}, ("original-model", "original-provider")),
    ({"provider_override": " "}, ("original-model", "original-provider")),
    ({"model_override": " next-model ", "provider_override": " next-provider "}, ("next-model", "next-provider")),
    ({"model_override": "next-model"}, ("next-model", None)),
    ({"model_override": ""}, (None, None)),
    ({"clear_model_override": True}, (None, None)),
])
def test_valid_override_and_fallback_semantics_preserved(board_client, surface, payload, expected):
    client, task_id, kb, kbc = board_client
    response = send(client, task_id, surface, payload)
    assert response.status_code == 200, response.text
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert (task.model_override, task.provider_override) == expected


def test_empty_bulk_still_rejected(board_client):
    client, _, _, _ = board_client
    response = client.post("/tasks/bulk", json={"ids": []})
    assert response.status_code == 400
    assert response.json()["detail"] == "ids is required"
