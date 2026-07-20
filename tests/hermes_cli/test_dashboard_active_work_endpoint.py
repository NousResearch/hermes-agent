from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers


def test_runtime_active_work_requires_dashboard_token():
    clear_providers()
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    try:
        client = TestClient(web_server.app, base_url="http://127.0.0.1:8080")

        response = client.get("/api/runtime/active-work")

        assert response.status_code == 401
    finally:
        web_server.app.state.auth_required = prev_required


def test_runtime_active_work_returns_backend_snapshot(monkeypatch):
    clear_providers()
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    monkeypatch.setattr(
        "tui_gateway.server.active_work_snapshot",
        lambda: {
            "active": True,
            "running_sessions": 1,
            "waiting_sessions": 0,
            "starting_sessions": 0,
            "active_subagents": 2,
        },
    )
    try:
        client = TestClient(web_server.app, base_url="http://127.0.0.1:8080")
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

        response = client.get("/api/runtime/active-work")

        assert response.status_code == 200
        assert response.json() == {
            "active": True,
            "running_sessions": 1,
            "waiting_sessions": 0,
            "starting_sessions": 0,
            "active_subagents": 2,
        }
    finally:
        web_server.app.state.auth_required = prev_required
