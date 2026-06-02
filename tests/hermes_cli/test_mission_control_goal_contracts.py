from __future__ import annotations

import json
import subprocess
import urllib.request

import pytest


@pytest.fixture()
def dashboard_client(_isolate_hermes_home, monkeypatch):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli import web_server
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {"dashboard": {"goal_contracts_enabled": True}},
    )
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def test_goal_contract_routes_require_dashboard_token(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app

    client = TestClient(app)
    assert client.get("/api/mission-control/goal-contracts").status_code == 401
    assert client.post("/api/mission-control/goal-contracts", json={"title": "x"}).status_code == 401
    assert client.get("/api/mission-control/goal-contracts/contract_demo").status_code == 401
    assert client.put("/api/mission-control/goal-contracts/contract_demo", json={"title": "x"}).status_code == 401
    assert client.delete("/api/mission-control/goal-contracts/contract_demo").status_code == 401


def test_goal_contract_routes_are_not_public():
    from hermes_cli.web_server import _PUBLIC_API_PATHS

    assert "/api/mission-control/goal-contracts" not in _PUBLIC_API_PATHS


def test_goal_contracts_feature_flag_default_disabled(_isolate_hermes_home, monkeypatch):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli import web_server
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    assert DEFAULT_CONFIG["dashboard"]["goal_contracts_enabled"] is False
    monkeypatch.setattr(web_server, "load_config", lambda: {})
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    resp = client.get("/api/mission-control/goal-contracts")

    assert resp.status_code == 404
    assert "disabled" in resp.json()["detail"]


def test_create_list_get_update_archive_goal_contract(dashboard_client):
    from hermes_constants import get_hermes_home

    create = dashboard_client.post(
        "/api/mission-control/goal-contracts",
        json={
            "title": "Goal contract",
            "objective": "Keep an inert local record.",
            "success_criteria": ["record exists", "no execution semantics"],
            "constraints": ["G2 only"],
            "source_refs": ["discord://thread/123", "/home/jenny/demo path"],
            "linked_mission_brief_id": "brief_20260602T000000Z_abcdef123456",
        },
    )

    assert create.status_code == 200
    contract = create.json()["contract"]
    assert contract["title"] == "Goal contract"
    assert contract["objective"] == "Keep an inert local record."
    assert contract["status"] == "draft"
    assert contract["success_criteria"] == ["record exists", "no execution semantics"]
    assert contract["constraints"] == ["G2 only"]
    assert contract["source_refs"] == ["discord://thread/123", "/home/jenny/demo path"]
    assert contract["vocabulary_version"] == "g1"
    assert contract["author"] == "dashboard"
    assert contract["trusted_for_execution"] is False
    assert contract["inert_context_only"] is True

    list_resp = dashboard_client.get("/api/mission-control/goal-contracts")
    assert list_resp.status_code == 200
    assert [item["id"] for item in list_resp.json()["items"]] == [contract["id"]]

    get_resp = dashboard_client.get(f"/api/mission-control/goal-contracts/{contract['id']}")
    assert get_resp.status_code == 200
    assert get_resp.json()["contract"]["id"] == contract["id"]

    update = dashboard_client.put(
        f"/api/mission-control/goal-contracts/{contract['id']}",
        json={
            "title": "Updated contract",
            "status": "active",
            "source_refs": ["~/literal", "https://example.invalid/a?b=c"],
        },
    )
    assert update.status_code == 200
    updated = update.json()["contract"]
    assert updated["title"] == "Updated contract"
    assert updated["status"] == "active"
    assert updated["source_refs"] == ["~/literal", "https://example.invalid/a?b=c"]
    assert updated["trusted_for_execution"] is False
    assert updated["inert_context_only"] is True

    archive = dashboard_client.delete(f"/api/mission-control/goal-contracts/{contract['id']}")
    assert archive.status_code == 200
    assert archive.json()["contract"]["status"] == "archived"

    default_list = dashboard_client.get("/api/mission-control/goal-contracts")
    assert default_list.status_code == 200
    assert default_list.json()["items"] == []

    direct_read = dashboard_client.get(f"/api/mission-control/goal-contracts/{contract['id']}")
    assert direct_read.status_code == 200
    assert direct_read.json()["contract"]["status"] == "archived"

    state_dir = get_hermes_home() / "state" / "mission-control" / "goal-contracts"
    audit_path = get_hermes_home() / "state" / "mission-control" / "goal-contracts-audit.jsonl"
    assert state_dir.is_dir()
    assert (state_dir / f"{contract['id']}.json").is_file()
    assert "goal_contract_archived" in audit_path.read_text(encoding="utf-8")


def test_status_validation_allows_only_draft_active_archived(dashboard_client):
    valid = dashboard_client.post(
        "/api/mission-control/goal-contracts",
        json={"title": "x", "objective": "y", "status": "active"},
    )
    assert valid.status_code == 200
    assert valid.json()["contract"]["status"] == "active"

    invalid = dashboard_client.post(
        "/api/mission-control/goal-contracts",
        json={"title": "x", "objective": "y", "status": "approved"},
    )
    assert invalid.status_code == 400
    assert "status" in invalid.json()["detail"]

    invalid_update = dashboard_client.put(
        f"/api/mission-control/goal-contracts/{valid.json()['contract']['id']}",
        json={"status": "completed"},
    )
    assert invalid_update.status_code == 400
    assert "status" in invalid_update.json()["detail"]


def test_source_refs_are_opaque_and_do_not_trigger_inspection(dashboard_client, monkeypatch):
    from pathlib import Path

    def fail_network(*args, **kwargs):
        raise AssertionError("URL-like source refs must not be fetched")

    def fail_command(*args, **kwargs):
        raise AssertionError("Goal Contract source refs must not trigger commands")

    original_stat = Path.stat
    stat_blocklist = {
        "/tmp/does-not-need-to-exist",
        "../not-normalized/../value",
    }

    def guarded_stat(self, *args, **kwargs):
        if str(self) in stat_blocklist:
            raise AssertionError("Goal Contract source refs must not be statted")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", fail_network)
    monkeypatch.setattr(subprocess, "run", fail_command)
    monkeypatch.setattr(subprocess, "Popen", fail_command)
    monkeypatch.setattr(Path, "stat", guarded_stat)

    refs = [
        "~/not-expanded",
        "../not-normalized/../value",
        "/tmp/does-not-need-to-exist",
        "https://example.invalid/never-fetch?token=still-opaque",
    ]

    resp = dashboard_client.post(
        "/api/mission-control/goal-contracts",
        json={"title": "Opaque refs", "objective": "Store refs as entered.", "source_refs": refs},
    )

    assert resp.status_code == 200
    contract = resp.json()["contract"]
    assert contract["source_refs"] == refs

    detail = dashboard_client.get(f"/api/mission-control/goal-contracts/{contract['id']}")
    assert detail.status_code == 200
    assert detail.json()["contract"]["source_refs"] == refs


def test_goal_contracts_do_not_use_runtime_goal_or_execution_paths(dashboard_client, monkeypatch):
    import gateway.run as gateway_run
    import hermes_cli.goal_contract_spec as spec
    import hermes_cli.goals as runtime_goals
    import hermes_cli.mission_control as packets

    def fail(*args, **kwargs):
        raise AssertionError("Goal Contract records must remain inert")

    monkeypatch.setattr(runtime_goals, "load_goal", fail)
    monkeypatch.setattr(runtime_goals, "save_goal", fail)
    monkeypatch.setattr(runtime_goals, "GoalManager", fail)
    monkeypatch.setattr(gateway_run.GatewayRunner, "_handle_goal_command", fail)
    monkeypatch.setattr(gateway_run.GatewayRunner, "_handle_subgoal_command", fail)
    monkeypatch.setattr(gateway_run.GatewayRunner, "_persist_safe_goal_task_contract", fail)
    monkeypatch.setattr(packets, "parse_worker_result_metadata", fail)
    monkeypatch.setattr(spec, "ACTION_CATEGORIES", fail)
    monkeypatch.setattr(spec, "PRESET_NAMES", fail)

    resp = dashboard_client.post(
        "/api/mission-control/goal-contracts",
        json={"title": "Static G1", "objective": "Reference vocabulary version only."},
    )

    assert resp.status_code == 200
    contract = resp.json()["contract"]
    assert contract["vocabulary_version"] == "g1"
    assert "actions" not in contract
    assert "checkpoints" not in contract
    assert "approval" not in json.dumps(contract).lower()


def test_goal_contract_audit_redacts_secrets(dashboard_client):
    from hermes_constants import get_hermes_home

    resp = dashboard_client.post(
        "/api/mission-control/goal-contracts",
        json={
            "title": "Secret contract",
            "objective": "Authorization: Bearer CONTRACTSECRET",
            "source_refs": ["api_key=REFSECRET"],
        },
    )

    assert resp.status_code == 200
    rendered = json.dumps(resp.json())
    assert "CONTRACTSECRET" not in rendered
    assert "REFSECRET" not in rendered

    contract = resp.json()["contract"]
    stored = (
        get_hermes_home()
        / "state"
        / "mission-control"
        / "goal-contracts"
        / f"{contract['id']}.json"
    ).read_text(encoding="utf-8")
    audit = (
        get_hermes_home() / "state" / "mission-control" / "goal-contracts-audit.jsonl"
    ).read_text(encoding="utf-8")
    assert "CONTRACTSECRET" not in stored
    assert "REFSECRET" not in stored
    assert "CONTRACTSECRET" not in audit
    assert "REFSECRET" not in audit
