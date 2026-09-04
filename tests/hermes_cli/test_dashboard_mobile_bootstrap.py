from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import __version__, web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")


@pytest.fixture
def loopback_client():
    clear_providers()
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    yield client
    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


@pytest.fixture
def gated_client():
    clear_providers()
    register_provider(StubAuthProvider())
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "hermes.example.test"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://hermes.example.test")
    yield client
    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def test_mobile_bootstrap_is_public_in_loopback_mode(loopback_client):
    response = loopback_client.get("/api/mobile/bootstrap")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "server_version": __version__,
        "api_version": 1,
        "auth_required": False,
        "auth_providers": [],
        "features": {
            "dashboard_status": True,
            "desktop_gateway_ws": True,
            "pty_chat": True,
            "ws_ticket_auth": False,
            "device_pairing": False,
            "hosted_relay": False,
        },
    }


def test_mobile_bootstrap_reports_gated_auth_without_login(gated_client):
    response = gated_client.get("/api/mobile/bootstrap")

    assert response.status_code == 200
    body = response.json()
    assert body["server_version"] == __version__
    assert body["api_version"] == 1
    assert body["auth_required"] is True
    assert body["auth_providers"] == ["stub"]
    assert body["features"]["ws_ticket_auth"] is True
    assert body["features"]["device_pairing"] is False
    assert body["features"]["hosted_relay"] is False


def test_mobile_bootstrap_does_not_expose_host_or_session_detail(gated_client):
    response = gated_client.get("/api/mobile/bootstrap")

    assert response.status_code == 200
    body = response.json()
    forbidden = {
        "hermes_home",
        "config_path",
        "env_path",
        "gateway_pid",
        "gateway_health_url",
        "active_sessions",
        "session_id",
        "token",
        "user_id",
    }
    assert forbidden.isdisjoint(body)
