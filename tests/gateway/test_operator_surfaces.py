"""Tests for Hermes-native operator mission/triage surfaces."""

from gateway import operator_surfaces


def test_build_operator_mission_surface_for_healthy_runtime(monkeypatch):
    runtime_status = {
        "gateway_state": "running",
        "exit_reason": None,
        "restart_requested": False,
        "active_agents": 2,
        "platforms": {
            "telegram": {
                "state": "connected",
                "updated_at": "2026-04-14T07:00:00Z",
                "error_code": None,
                "error_message": None,
            }
        },
        "updated_at": "2026-04-14T07:00:01Z",
    }
    validation = {
        "pid": {"exists": True, "valid": True, "errors": []},
        "runtime_status": {"exists": True, "valid": True, "errors": []},
        "evidence": {"exists": True, "line_count": 3, "last_event": "gateway_running"},
    }

    monkeypatch.setattr(operator_surfaces, "read_runtime_status", lambda: runtime_status)
    monkeypatch.setattr(operator_surfaces, "validate_runtime_artifacts", lambda: validation)

    mission = operator_surfaces.build_operator_mission_surface()

    assert mission["schema"] == "hermes.operator_mission_surface.v1"
    assert mission["gateway"]["state"] == "running"
    assert mission["gateway"]["active_agents"] == 2
    assert mission["headline"] == "Gateway healthy"
    assert mission["recommended_actions"] == []
    assert mission["validation"]["runtime_valid"] is True
    assert mission["evidence_refs"]["runtime_status"].endswith("gateway_state.json")
    assert mission["evidence_refs"]["runtime_events"].endswith("gateway_runtime_events.jsonl")


def test_build_operator_triage_surface_prioritizes_fatal_platforms_and_invalid_runtime(monkeypatch):
    runtime_status = {
        "gateway_state": "startup_failed",
        "exit_reason": "telegram conflict",
        "restart_requested": True,
        "active_agents": 0,
        "platforms": {
            "telegram": {
                "state": "fatal",
                "updated_at": "2026-04-14T07:00:00Z",
                "error_code": "telegram_polling_conflict",
                "error_message": "another poller is active",
            }
        },
        "updated_at": "2026-04-14T07:00:01Z",
    }
    validation = {
        "pid": {"exists": True, "valid": True, "errors": []},
        "runtime_status": {
            "exists": True,
            "valid": False,
            "errors": ["gateway_state must be one of ['running']"],
        },
        "evidence": {"exists": True, "line_count": 5, "last_event": "gateway_startup_failed"},
    }

    monkeypatch.setattr(operator_surfaces, "read_runtime_status", lambda: runtime_status)
    monkeypatch.setattr(operator_surfaces, "validate_runtime_artifacts", lambda: validation)

    triage = operator_surfaces.build_operator_triage_surface()

    assert triage["schema"] == "hermes.operator_triage_surface.v1"
    assert triage["severity"] == "critical"
    assert triage["issue_count"] == 3
    issue_kinds = [issue["kind"] for issue in triage["issues"]]
    assert issue_kinds == [
        "platform_failure",
        "runtime_artifact_invalid",
        "restart_requested",
    ]
    assert triage["issues"][0]["suggested_command"] == "hermes gateway operator-status --json"
    assert triage["issues"][1]["evidence_ref"].endswith("gateway_state.json")


def test_build_operator_triage_surface_returns_healthy_summary_when_no_issues(monkeypatch):
    runtime_status = {
        "gateway_state": "running",
        "exit_reason": None,
        "restart_requested": False,
        "active_agents": 1,
        "platforms": {"telegram": {"state": "connected", "updated_at": "2026-04-14T07:00:00Z"}},
        "updated_at": "2026-04-14T07:00:01Z",
    }
    validation = {
        "pid": {"exists": True, "valid": True, "errors": []},
        "runtime_status": {"exists": True, "valid": True, "errors": []},
        "evidence": {"exists": True, "line_count": 1, "last_event": "gateway_running"},
    }

    monkeypatch.setattr(operator_surfaces, "read_runtime_status", lambda: runtime_status)
    monkeypatch.setattr(operator_surfaces, "validate_runtime_artifacts", lambda: validation)

    triage = operator_surfaces.build_operator_triage_surface()

    assert triage["severity"] == "info"
    assert triage["issue_count"] == 0
    assert triage["issues"] == []
    assert triage["summary"] == "No operator action required"
