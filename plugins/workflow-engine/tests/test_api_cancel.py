"""
Tests for POST /runs/{run_id}/cancel.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine.wiring import create_engine

_SIMPLE_YAML = """\
id: cancel-test
name: Cancel Test
description: Workflow for cancel endpoint tests
nodes:
  - id: ask
    approval: Wait here so the run stays non-terminal
"""


@pytest.fixture()
def client():
    engine = create_engine(
        db_path=":memory:",
        seed_bundled=False,
        write_manifest=False,
        crash_recovery=False,
    )
    app = FastAPI()

    import plugins.workflow_engine.dashboard.plugin_api as api_mod
    original = api_mod._engine
    api_mod._engine = engine
    app.include_router(api_mod.router)

    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/definitions", json={
            "id": "cancel-test",
            "name": "Cancel Test",
            "yaml": _SIMPLE_YAML,
            "source": "user",
        })
        yield c

    api_mod._engine = original


def test_cancel_run_not_found(client):
    r = client.post("/runs/nonexistent/cancel")
    assert r.status_code == 404
    assert "not found" in r.json()["error"]


def test_cancel_run_success(client):
    r = client.post("/runs", json={
        "workflow_id": "cancel-test",
        "conversation_id": "conv-cancel-001",
        "user_message": "run it",
    })
    assert r.status_code == 201
    run_id = r.json()["run"]["id"]

    r2 = client.post(f"/runs/{run_id}/cancel")
    assert r2.status_code == 200
    assert r2.json() == {"ok": True}

    # Verify status transitioned to cancelled
    detail = client.get(f"/runs/{run_id}").json()
    assert detail["run"]["status"] == "cancelled"


def test_cancel_already_terminal(client):
    r = client.post("/runs", json={
        "workflow_id": "cancel-test",
        "conversation_id": "conv-cancel-002",
        "user_message": "run it",
    })
    assert r.status_code == 201
    run_id = r.json()["run"]["id"]

    # First cancel succeeds
    r2 = client.post(f"/runs/{run_id}/cancel")
    assert r2.status_code == 200

    # Second cancel returns 409
    r3 = client.post(f"/runs/{run_id}/cancel")
    assert r3.status_code == 409
    assert "already terminal" in r3.json()["error"]
