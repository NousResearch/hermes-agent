"""Loopback JSON bootstrap for the dashboard session token (#117976).

``auth_required: false`` on ``/api/health`` previously implied no credential was
needed, but ``/api/ws`` still requires ``?token=``. The token was only injected
into HTML. These tests lock the JSON bootstrap contract.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture
def loopback_client():
    """Default loopback bind: OAuth gate off, legacy session token in effect."""
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    yield client
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
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def test_session_token_path_is_public_allowlisted():
    assert "/api/session-token" in PUBLIC_API_PATHS


def test_health_reports_session_token_required_on_loopback(loopback_client):
    r = loopback_client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["auth_required"] is False
    assert body["session_token_required"] is True
    # Health stays secret-free — token lives on /api/session-token only.
    assert "token" not in body
    assert "session_token" not in body


def test_health_reports_session_token_not_required_when_gated(gated_client):
    r = gated_client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["auth_required"] is True
    assert body["session_token_required"] is False


def test_session_token_bootstrap_returns_token_on_loopback(loopback_client):
    r = loopback_client.get("/api/session-token")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token"] == web_server._SESSION_TOKEN
    assert body["header"] == "X-Hermes-Session-Token"
    assert body["query_param"] == "token"


def test_session_token_bootstrap_refuses_gated_bind(gated_client):
    r = gated_client.get("/api/session-token")
    assert r.status_code == 401
    detail = r.json()["detail"]
    assert "ws-ticket" in detail
    assert "token" not in r.json()


def test_session_token_bootstrap_refuses_non_loopback_peer(loopback_client):
    """Even with the OAuth gate off, a non-loopback TCP peer must not get the token.

    ``--insecure`` binds can accept remote WS peers that already know the token;
    this endpoint must not hand the secret to arbitrary remotes.
    """
    from hermes_cli.web_routers import status as status_mod

    real = status_mod._request_peer_is_loopback
    status_mod._request_peer_is_loopback = lambda _request: False
    try:
        r = loopback_client.get("/api/session-token")
        assert r.status_code == 401
        assert "loopback" in r.json()["detail"].lower()
        assert "token" not in r.json()
    finally:
        status_mod._request_peer_is_loopback = real


def test_session_token_works_as_bearer_for_gated_api(loopback_client):
    """The bootstrapped token authenticates a normally-gated /api route."""
    token = loopback_client.get("/api/session-token").json()["token"]
    denied = loopback_client.get("/api/sessions")
    assert denied.status_code == 401
    allowed = loopback_client.get(
        "/api/sessions",
        headers={"X-Hermes-Session-Token": token},
    )
    assert allowed.status_code == 200
