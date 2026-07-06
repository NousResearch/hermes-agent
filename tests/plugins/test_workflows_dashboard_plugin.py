"""Tests for the Workflows dashboard plugin backend."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import workflows_db as wfdb

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "workflows" / "dashboard"

PASS_SPEC = {
    "id": "dashboard_demo",
    "name": "Dashboard Demo",
    "version": 1,
    "triggers": [{"type": "manual", "id": "manual"}],
    "nodes": {"start": {"type": "pass", "output": {"ok": True}}},
}

WAIT_SPEC = {
    "id": "dashboard_wait",
    "name": "Dashboard Wait",
    "version": 1,
    "triggers": [{"type": "manual", "id": "manual"}],
    "nodes": {
        "start": {"type": "pass", "output": {"seen": "${ input.value }"}},
        "pause": {"type": "wait", "seconds": 60},
    },
    "edges": [{"from": "start", "to": "pause"}],
}


def _load_plugin_router():
    plugin_file = PLUGIN_DIR / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_workflows_test", plugin_file
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def workflows_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    wfdb.init_db()
    return home


@pytest.fixture
def client(workflows_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/workflows")
    return TestClient(app)


def _deploy(client: TestClient, spec: dict = PASS_SPEC) -> dict:
    r = client.post("/api/plugins/workflows/definitions/deploy", json={"spec": spec})
    assert r.status_code == 200, r.text
    return r.json()["definition"]


def _assert_pass_spec(spec: dict) -> None:
    assert spec["id"] == PASS_SPEC["id"]
    assert spec["name"] == PASS_SPEC["name"]
    assert spec["version"] == PASS_SPEC["version"]
    assert spec["nodes"]["start"]["type"] == "pass"


def test_manifest_points_to_plugin_api():
    manifest_file = PLUGIN_DIR / "manifest.json"
    assert manifest_file.exists(), f"manifest missing: {manifest_file}"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["name"] == "workflows"
    assert manifest["label"] == "Workflows"
    assert manifest["tab"] == {"path": "/workflows", "position": "after:kanban"}
    assert manifest["entry"] == "dist/index.js"
    assert manifest["api"] == "plugin_api.py"


def test_validate_deploy_list_show_roundtrip(client):
    r = client.post("/api/plugins/workflows/definitions/validate", json=PASS_SPEC)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["valid"] is True
    assert body["definition"]["workflow_id"] == "dashboard_demo"
    _assert_pass_spec(body["definition"]["spec"])

    deployed = _deploy(client)
    assert deployed["workflow_id"] == "dashboard_demo"
    assert deployed["created_by"] == "dashboard"
    _assert_pass_spec(deployed["spec"])

    listed = client.get("/api/plugins/workflows/definitions").json()["definitions"]
    assert [item["workflow_id"] for item in listed] == ["dashboard_demo"]

    shown = client.get("/api/plugins/workflows/definitions/dashboard_demo").json()["definition"]
    assert shown["workflow_id"] == "dashboard_demo"
    assert shown["version"] == 1
    _assert_pass_spec(shown["spec"])


def test_run_endpoint_creates_execution_and_list_show_return_it(client):
    _deploy(client)

    r = client.post(
        "/api/plugins/workflows/definitions/dashboard_demo/run",
        json={"input": {"value": 7}},
    )
    assert r.status_code == 200, r.text
    execution = r.json()["execution"]
    assert execution["workflow_id"] == "dashboard_demo"
    assert execution["input"] == {"value": 7}
    assert execution["status"] in {"queued", "running", "waiting", "succeeded"}

    with wfdb.connect() as conn:
        stored = wfdb.get_execution(conn, execution["execution_id"])
    assert stored.workflow_id == "dashboard_demo"
    assert stored.input == {"value": 7}

    listed = client.get(
        "/api/plugins/workflows/executions", params={"workflow_id": "dashboard_demo"}
    ).json()["executions"]
    assert [item["execution_id"] for item in listed] == [execution["execution_id"]]

    shown = client.get(
        f"/api/plugins/workflows/executions/{execution['execution_id']}"
    ).json()["execution"]
    assert shown["execution_id"] == execution["execution_id"]
    assert shown["input"] == {"value": 7}


def test_events_endpoint_returns_append_only_events(client):
    _deploy(client)
    execution_id = client.post(
        "/api/plugins/workflows/definitions/dashboard_demo/run", json={}
    ).json()["execution"]["execution_id"]

    with wfdb.connect() as conn:
        wfdb.append_event(conn, execution_id, "custom_one", {"n": 1})
        wfdb.append_event(conn, execution_id, "custom_two", {"n": 2})

    r = client.get(f"/api/plugins/workflows/executions/{execution_id}/events")
    assert r.status_code == 200, r.text
    custom = [e for e in r.json()["events"] if e["kind"].startswith("custom_")]
    assert [e["kind"] for e in custom] == ["custom_one", "custom_two"]
    assert [e["payload"] for e in custom] == [{"n": 1}, {"n": 2}]
    assert custom[0]["id"] < custom[1]["id"]


def test_cancel_endpoint_is_idempotent(client):
    _deploy(client, WAIT_SPEC)
    execution_id = client.post(
        "/api/plugins/workflows/definitions/dashboard_wait/run",
        json={"input_json": '{"value": 3}'},
    ).json()["execution"]["execution_id"]

    first = client.post(f"/api/plugins/workflows/executions/{execution_id}/cancel")
    assert first.status_code == 200, first.text
    assert first.json()["cancelled"] is True
    assert first.json()["execution"]["status"] == "cancelled"

    second = client.post(f"/api/plugins/workflows/executions/{execution_id}/cancel")
    assert second.status_code == 200, second.text
    assert second.json()["cancelled"] is False
    assert second.json()["execution"]["status"] == "cancelled"


def test_bad_spec_returns_400_with_validation_message(client):
    r = client.post(
        "/api/plugins/workflows/definitions/validate",
        json={"id": "bad_demo", "name": "Bad", "version": 1, "nodes": {}},
    )
    assert r.status_code == 400
    assert "workflow must define at least one node" in r.json()["detail"]


def test_bad_run_input_returns_400(client):
    _deploy(client)
    r = client.post(
        "/api/plugins/workflows/definitions/dashboard_demo/run",
        json={"input_json": []},
    )
    assert r.status_code == 400
    assert "input_json" in r.json()["detail"]
