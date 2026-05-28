"""Phase 7 — /api/status exposes auth-gate state + AuthWidget integration.

The dashboard's status endpoint now reports ``auth_required`` and
``auth_providers`` so the AuthWidget + StatusPage can render the
correct "gated / loopback" badge without a separate round trip. This
test asserts both shapes (gated and loopback).

The AuthWidget itself is .tsx — no Python test here. The widget's
behaviour (renders nothing on 401, shows truncated user_id, etc.) is
documented in AuthWidget.tsx; covered manually via the Phase 4.2
smoke test against staging Portal.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

# These tests mutate ``web_server.app.state.auth_required`` so they share
# the same xdist group as the other dashboard-auth gated_app tests.
pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")


@pytest.fixture
def gated_client():
    clear_providers()
    register_provider(StubAuthProvider())
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


@pytest.fixture
def loopback_client():
    clear_providers()
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 8080
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app, base_url="http://127.0.0.1:8080")
    yield client
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def _login(client: TestClient) -> None:
    """Drive the stub OAuth round trip so the gated client is authed."""
    r1 = client.get("/auth/login?provider=stub", follow_redirects=False)
    assert r1.status_code == 302
    state = r1.headers["location"].split("state=")[1]
    r2 = client.get(
        f"/auth/callback?code=stub_code&state={state}", follow_redirects=False
    )
    assert r2.status_code == 302


def test_status_reports_auth_required_in_gated_mode(gated_client):
    _login(gated_client)
    r = gated_client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["auth_required"] is True
    assert body["auth_providers"] == ["stub"]


def test_status_reports_auth_disabled_in_loopback_mode(loopback_client):
    r = loopback_client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["auth_required"] is False
    # Loopback mode has no registered providers (the Nous plugin's env
    # vars aren't set in test).
    assert body["auth_providers"] == []


def test_status_preserves_existing_fields(loopback_client):
    """Defence-in-depth: adding auth_required/auth_providers must not
    have dropped any previous field (the dashboard's React StatusPage
    relies on the full payload shape)."""
    r = loopback_client.get("/api/status")
    body = r.json()
    expected_keys = {
        "version", "release_date", "hermes_home", "config_path", "env_path",
        "config_version", "latest_config_version", "gateway_running",
        "gateway_pid", "gateway_health_url", "gateway_state",
        "gateway_platforms", "gateway_exit_reason", "gateway_updated_at",
        "active_sessions", "auth_required", "auth_providers",
    }
    missing = expected_keys - set(body.keys())
    assert not missing, f"/api/status dropped fields: {missing}"
