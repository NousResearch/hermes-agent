"""
Tests for Phase 3-ext endpoints:
  DELETE /definitions/{id}
  GET    /runs/by-conversation/{conv_id}
  GET    /runs/active
  POST   /runs/{run_id}/resume
  GET    /runs/{run_id}/nodes
  GET    /node-runs/{node_run_id}
  POST   /runs/{run_id}/events
  GET    /runs/{run_id}/events
  POST   /runs/{run_id}/phase-transitions
  GET    /runs/{run_id}/phase-transitions
  POST   /runs/{run_id}/approval-claim
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine.wiring import create_engine

_HELLO_YAML = """\
id: hello-world
name: Hello World
description: A minimal test workflow
nodes:
  - id: greet
    prompt: Say hello
"""


@pytest.fixture()
def client():
    engine = create_engine(db_path=":memory:", seed_bundled=False, write_manifest=False, crash_recovery=False)
    app = FastAPI()

    import plugins.workflow_engine.dashboard.plugin_api as api_mod
    original = api_mod._engine
    api_mod._engine = engine
    app.include_router(api_mod.router)

    with TestClient(app, raise_server_exceptions=True) as c:
        # Seed definition
        c.post("/definitions", json={
            "id": "hello-world",
            "name": "Hello World",
            "yaml": _HELLO_YAML,
            "source": "user",
        })
        yield c

    api_mod._engine = original


@pytest.fixture()
def run_id(client):
    """Create a run and return its id."""
    r = client.post("/runs", json={
        "workflow_id": "hello-world",
        "conversation_id": "conv-ext-001",
        "user_message": "go",
    })
    assert r.status_code == 201
    return r.json()["run"]["id"]


# ---------------------------------------------------------------------------
# DELETE /definitions/{id}
# ---------------------------------------------------------------------------

def test_delete_definition_success(client):
    r = client.delete("/definitions/hello-world")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    # Gone
    r2 = client.get("/definitions/hello-world")
    assert r2.status_code == 404


def test_delete_definition_not_found(client):
    r = client.delete("/definitions/no-such-def")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /runs/by-conversation/{conv_id}
# ---------------------------------------------------------------------------

def test_find_run_by_conversation_id_found(client, run_id):
    r = client.get("/runs/by-conversation/conv-ext-001")
    assert r.status_code == 200
    body = r.json()
    assert body["run"] is not None
    assert body["run"]["conversation_id"] == "conv-ext-001"


def test_find_run_by_conversation_id_not_found(client):
    r = client.get("/runs/by-conversation/no-such-conv")
    assert r.status_code == 200
    assert r.json()["run"] is None


# ---------------------------------------------------------------------------
# GET /runs/active
# ---------------------------------------------------------------------------

def test_get_active_run_no_match(client):
    r = client.get("/runs/active?scope_path=/no/such/path")
    assert r.status_code == 200
    assert r.json()["run"] is None


def test_get_active_run_empty_path(client):
    r = client.get("/runs/active?scope_path=")
    assert r.status_code == 200
    # Returns dict with run key regardless
    assert "run" in r.json()


# ---------------------------------------------------------------------------
# POST /runs/{run_id}/resume
# ---------------------------------------------------------------------------

def test_resume_run_not_found(client):
    r = client.post("/runs/no-such-run-id/resume")
    assert r.status_code == 404


def test_resume_run_success(client, run_id):
    r = client.post(f"/runs/{run_id}/resume")
    assert r.status_code == 200
    assert "run" in r.json()


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/nodes
# ---------------------------------------------------------------------------

def test_list_node_runs(client, run_id):
    r = client.get(f"/runs/{run_id}/nodes")
    assert r.status_code == 200
    body = r.json()
    assert "nodeRuns" in body
    assert isinstance(body["nodeRuns"], list)


def test_list_node_runs_not_found(client):
    r = client.get("/runs/no-such/nodes")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /node-runs/{node_run_id}
# ---------------------------------------------------------------------------

def test_find_node_run_by_id_not_found(client):
    r = client.get("/node-runs/no-such-node-run")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /runs/{run_id}/events
# ---------------------------------------------------------------------------

def test_append_event_success(client, run_id):
    r = client.post(f"/runs/{run_id}/events", json={
        "event_type": "test_event",
        "data": {"msg": "hello"},
    })
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_append_event_missing_type(client, run_id):
    r = client.post(f"/runs/{run_id}/events", json={"data": {}})
    assert r.status_code == 400


def test_append_event_run_not_found(client):
    r = client.post("/runs/no-such-run/events", json={"event_type": "x"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/events
# ---------------------------------------------------------------------------

def test_list_run_events_empty(client, run_id):
    r = client.get(f"/runs/{run_id}/events")
    assert r.status_code == 200
    body = r.json()
    assert "events" in body
    assert isinstance(body["events"], list)


def test_list_run_events_after_append(client, run_id):
    client.post(f"/runs/{run_id}/events", json={"event_type": "my_event"})
    r = client.get(f"/runs/{run_id}/events?limit=10")
    assert r.status_code == 200
    events = r.json()["events"]
    assert any(e.get("event_type") == "my_event" for e in events)


def test_list_run_events_not_found(client):
    r = client.get("/runs/no-such/events")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /runs/{run_id}/phase-transitions
# ---------------------------------------------------------------------------

def test_record_phase_transition_success(client, run_id):
    r = client.post(f"/runs/{run_id}/phase-transitions", json={
        "toPhase": "execute",
        "decidedBy": "user",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["from"] == "plan"
    assert body["to"] == "execute"


def test_record_phase_transition_missing_fields(client, run_id):
    r = client.post(f"/runs/{run_id}/phase-transitions", json={"toPhase": "execute"})
    assert r.status_code == 400


def test_record_phase_transition_run_not_found(client):
    r = client.post("/runs/no-such/phase-transitions", json={
        "toPhase": "execute",
        "decidedBy": "user",
    })
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/phase-transitions
# ---------------------------------------------------------------------------

def test_list_phase_transitions_empty(client, run_id):
    r = client.get(f"/runs/{run_id}/phase-transitions")
    assert r.status_code == 200
    body = r.json()
    assert "phaseTransitions" in body
    assert isinstance(body["phaseTransitions"], list)


def test_list_phase_transitions_after_record(client, run_id):
    client.post(f"/runs/{run_id}/phase-transitions", json={
        "toPhase": "execute",
        "decidedBy": "system",
    })
    r = client.get(f"/runs/{run_id}/phase-transitions")
    assert r.status_code == 200
    transitions = r.json()["phaseTransitions"]
    assert len(transitions) >= 1
    assert transitions[-1]["to_phase"] == "execute"


def test_list_phase_transitions_run_not_found(client):
    r = client.get("/runs/no-such/phase-transitions")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /runs/{run_id}/approval-claim
# ---------------------------------------------------------------------------

def test_approval_claim_missing_fields(client, run_id):
    r = client.post(f"/runs/{run_id}/approval-claim", json={"decision": "approved"})
    assert r.status_code == 400


def test_approval_claim_invalid_decision(client, run_id):
    r = client.post(f"/runs/{run_id}/approval-claim", json={
        "nodeRunId": "some-node-run-id",
        "decision": "maybe",
    })
    assert r.status_code == 400


def test_approval_claim_node_not_paused(client, run_id):
    # node_run_id doesn't exist — claimed=False
    r = client.post(f"/runs/{run_id}/approval-claim", json={
        "nodeRunId": "nonexistent-node-run",
        "decision": "approved",
        "approvalResponse": "lgtm",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["claimed"] is False
    assert body["terminalStatus"] == "completed"


def test_approval_claim_run_not_found(client):
    r = client.post("/runs/no-such/approval-claim", json={
        "nodeRunId": "x",
        "decision": "approved",
    })
    assert r.status_code == 404
