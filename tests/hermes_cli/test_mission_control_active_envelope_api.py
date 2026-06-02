from __future__ import annotations

import pytest


@pytest.fixture()
def dashboard_client(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def test_active_envelope_requires_dashboard_token(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app

    client = TestClient(app)

    assert client.get("/api/mission-control/active-envelope").status_code == 401


def test_active_envelope_is_not_public():
    from hermes_cli.web_server import _PUBLIC_API_PATHS

    assert "/api/mission-control/active-envelope" not in _PUBLIC_API_PATHS


def test_active_envelope_returns_exact_empty_state(dashboard_client):
    resp = dashboard_client.get("/api/mission-control/active-envelope")

    assert resp.status_code == 200
    assert resp.json() == {
        "exists": False,
        "active_lane": None,
        "active_mode": None,
        "execution_boundary": "no_active_authorization",
        "allowed_actions": [],
        "forbidden_actions": [],
        "checkpoint": None,
        "repo_state": {
            "status": "unknown",
            "source": "not_probed",
        },
        "evidence": {
            "count": 0,
            "links": [],
        },
        "data_source": "no_persisted_envelope",
        "trusted_for_execution": False,
        "inert_context_only": True,
    }


def test_active_envelope_does_not_call_probe_or_persistence_helpers(dashboard_client, monkeypatch):
    import hermes_cli.mission_control as mission_control
    from hermes_cli import web_server

    blocked_names = [
        "project_status",
        "open_tasks",
        "latest_worker_results",
        "repo_status",
        "approval_gates",
        "recent_audit_log",
        "list_packets",
        "get_packet",
        "save_next_codex_prompt",
        "import_worker_result",
        "set_block_flag",
        "create_rejection_audit",
    ]

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("active-envelope must not probe repo/state or persist data")

    for name in blocked_names:
        monkeypatch.setattr(mission_control, name, fail_if_called)
    monkeypatch.setattr(web_server, "load_config", fail_if_called)

    resp = dashboard_client.get("/api/mission-control/active-envelope")

    assert resp.status_code == 200


def test_active_envelope_exposes_no_mutation_methods():
    from hermes_cli.web_server import app

    methods = {
        method
        for route in app.routes
        if getattr(route, "path", None) == "/api/mission-control/active-envelope"
        for method in getattr(route, "methods", set())
    }

    assert methods == {"GET"}
