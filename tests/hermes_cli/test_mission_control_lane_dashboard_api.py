from __future__ import annotations

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
        lambda: {"dashboard": {"approval_slices_enabled": True, "evidence_cards_enabled": True}},
    )
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def _envelope_payload(**overrides):
    payload = {
        "title": "Hermes OS lane dashboard MVP",
        "mode": "implement-slice",
        "mode_label": "Code/test only",
        "allowed_actions": ["inspect_repo", "read_files", "search_files", "edit_files", "run_focused_tests", "run_build", "run_lint"],
        "forbidden_actions": ["deploy", "restart_service", "push", "open_pr", "external_network", "touch_secrets", "destructive_git"],
        "checkpoints": ["stop_after_validation_report", "stop_on_unrelated_dirty_files"],
        "checkpoint_requirements": ["stop_on_scope_expansion", "stop_on_restart_or_deploy_needed"],
        "repo_context": {
            "path": "/home/jenny/.hermes/hermes-context-routing-e1d-integration",
            "branch": "mission-control-os-stateful-foundation",
            "head": "7399c9785",
            "dirty_state": "only excluded generated website files",
            "source": "OR1C Start Gate",
        },
        "lane_lock": {"active_lane": "Hermes OS lane dashboard MVP"},
        "relationships": {"source_thread": "discord://thread/mission-control-os"},
        "source": "manual_command",
        "raw_user_approval": "Build read-only dashboard stub only.",
        "metadata": {
            "approval_tier": "code/test only",
            "start_gate_status": "pass",
            "next_recommended_action": "Review read-only MVP and commit after Travis approval.",
            "context_budget": {"input_estimate": 9000, "output_estimate": 2500, "remaining_context_window": "moderate"},
            "quarantine_warning": "No parent scans or quarantined path access allowed.",
        },
    }
    payload.update(overrides)
    return payload


def _slice_payload(**overrides):
    payload = {
        "title": "Read-only dashboard approval slice",
        "repo_path": "/home/jenny/.hermes/hermes-context-routing-e1d-integration",
        "allowed_paths": ["hermes_cli/web_server.py", "web/src/components/MissionControlLaneDashboard.tsx"],
        "forbidden_paths": ["gateway/run.py", "website/static/api/skills.json"],
        "expected_locality": "current repo only",
        "allowed_actions": ["inspect_repo", "read_files", "search_files", "edit_files", "run_focused_tests"],
        "forbidden_actions": ["deploy", "restart_service", "push", "open_pr"],
        "stop_condition": "stop_after_validation_report",
        "checkpoint": "stop_after_validation_report",
        "created_by": "Travis",
        "created_from": "manual_command",
        "raw_user_approval": "Code/test only; stop before commit.",
    }
    payload.update(overrides)
    return payload


def _card_payload(**overrides):
    payload = {
        "kind": "validation",
        "title": "Focused dashboard validation",
        "summary": "Focused backend and frontend validation planned for lane dashboard MVP.",
        "details": "Details are inert evidence data only.",
        "structured_payload": {"command": "pytest focused", "exit_code": 0},
        "limitations": ["No broad tests requested."],
        "redaction_notes": ["No raw transcript content included."],
        "source": "operator_supplied",
        "created_by": "dashboard-test",
        "created_from": "operator_supplied",
    }
    payload.update(overrides)
    return payload


def test_lane_dashboard_requires_dashboard_token(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app

    client = TestClient(app)
    assert client.get("/api/mission-control/lane-dashboard").status_code == 401


def test_lane_dashboard_is_not_public():
    from hermes_cli.web_server import _PUBLIC_API_PATHS

    assert "/api/mission-control/lane-dashboard" not in _PUBLIC_API_PATHS


def test_lane_dashboard_returns_empty_inert_read_model(dashboard_client):
    resp = dashboard_client.get("/api/mission-control/lane-dashboard")

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "local_read_only"
    assert body["active_lane"]["label"] == "No active lane"
    assert body["task_control_envelope"]["exists"] is False
    assert body["approval_tier"]["label"] == "No approval slice"
    assert body["start_gate"]["status"] == "unknown"
    assert body["evidence"]["summaries"] == []
    assert body["next_recommended_action"] == "Attach a Task Control Envelope before starting lane work."
    assert body["token_context_budget"]["show_token_estimates"] is True
    assert body["safety"]["quarantine_parent_scan_warning"]
    assert body["safety"]["transcript_loaded"] is False
    assert body["safety"]["execution_controls"] is False
    assert body["trusted_for_execution"] is False
    assert body["inert_context_only"] is True


def test_lane_dashboard_summarizes_envelope_slice_and_evidence_without_runtime_access(dashboard_client, monkeypatch):
    import subprocess
    from pathlib import Path

    import model_tools
    from hermes_cli.mission_control_approval_slices import create_approval_slice
    from hermes_cli.mission_control_evidence_cards import create_card
    from hermes_cli.mission_control_task_control_envelopes import create_task_control_envelope

    envelope = create_task_control_envelope(_envelope_payload())
    approval_slice = create_approval_slice(_slice_payload())
    create_card(_card_payload(title="Start Gate", summary="OR1C Start Gate passed with only excluded generated website files."))
    create_card(_card_payload(kind="secret_scan", title="Added-line secret scan", summary="No hardcoded secrets found in added lines."))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("lane dashboard must not execute commands or call tools")

    original_stat = Path.stat

    def guarded_stat(self, *args, **kwargs):
        if str(self) == "/tmp/lane-dashboard-must-not-stat":
            raise AssertionError("lane dashboard must not stat opaque paths")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_if_called)
    monkeypatch.setattr(subprocess, "Popen", fail_if_called)
    monkeypatch.setattr(model_tools, "handle_function_call", fail_if_called)
    monkeypatch.setattr(Path, "stat", guarded_stat)

    resp = dashboard_client.get("/api/mission-control/lane-dashboard")

    assert resp.status_code == 200
    body = resp.json()
    assert body["active_lane"] == {
        "label": "Hermes OS lane dashboard MVP",
        "mode": "implement-slice",
        "status": "active",
        "source": "persisted_task_control_envelope",
    }
    assert body["task_control_envelope"]["id"] == envelope["id"]
    assert body["task_control_envelope"]["summary"] == "Hermes OS lane dashboard MVP"
    assert body["approval_tier"]["label"] == "code/test only"
    assert body["approval_tier"]["active_slice_count"] == 1
    assert body["approval_tier"]["latest_slice_id"] == approval_slice["id"]
    assert body["start_gate"] == {
        "status": "pass",
        "source": "OR1C Start Gate",
        "repo_state": "only excluded generated website files",
    }
    assert body["allowed_actions"] == _envelope_payload()["allowed_actions"]
    assert body["forbidden_actions"] == _envelope_payload()["forbidden_actions"]
    assert body["evidence"]["count"] == 2
    assert [item["title"] for item in body["evidence"]["summaries"]] == ["Added-line secret scan", "Start Gate"]
    assert body["evidence"]["details_on_demand"] is True
    assert body["token_context_budget"]["estimated_input_tokens"] == 9000
    assert body["token_context_budget"]["estimated_output_tokens"] == 2500
    assert body["safety"]["parent_scan_performed"] is False
    assert body["safety"]["quarantined_path_accessed"] is False
    assert body["safety"]["transcript_loaded"] is False
    assert body["safety"]["execution_controls"] is False


def test_lane_dashboard_exposes_no_mutation_methods():
    from hermes_cli.web_server import app

    methods = {
        method
        for route in app.routes
        if getattr(route, "path", None) == "/api/mission-control/lane-dashboard"
        for method in getattr(route, "methods", set())
    }

    assert methods == {"GET"}
