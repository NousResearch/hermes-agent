"""Regression and behavioural tests for ?token= query parameter authentication

on /api/files/download in gated OAuth mode (Issue #80577).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.base import ProviderError
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider, _sign


class UnreachableAuthProvider(StubAuthProvider):
    name = "unreachable_stub"

    def verify_session(self, *, access_token: str):
        raise ProviderError("unreachable_idp")


@pytest.fixture
def gated_client():
    """Configure web_server.app for gated mode with StubAuthProvider."""
    clear_providers()
    provider = StubAuthProvider()
    register_provider(provider)

    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)

    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True

    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    import time
    valid_token = _sign({
        "sub": "stub-user-1",
        "email": "stub@example.test",
        "name": "Stub User",
        "org_id": "stub-org-1",
        "exp": int(time.time()) + 3600,
    })

    yield client, valid_token

    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


@pytest.fixture
def loopback_client():
    """Configure web_server.app for loopback mode (auth_required=False)."""
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


def test_gated_download_valid_query_token(gated_client, tmp_path, monkeypatch):
    """GET /api/files/download?path=...&token=<valid_token> succeeds under auth_required=True."""
    client, valid_token = gated_client

    test_file = tmp_path / "sample.mp4"
    test_file.write_bytes(b"fake mp4 video content")

    # Point read_desktop_file or file resolver if needed
    r = client.get(
        f"/api/files/download?path={test_file}&token={valid_token}"
    )
    # The request must pass auth (not 401/403/302). It should return 200 with content or file response.
    assert r.status_code == 200
    assert r.content == b"fake mp4 video content"


def test_gated_download_invalid_query_token(gated_client, tmp_path):
    """Invalid query token on /api/files/download returns 401 with reason='invalid_or_expired_session'."""
    client, _ = gated_client
    test_file = tmp_path / "sample.mp4"
    test_file.write_bytes(b"content")

    r = client.get(
        f"/api/files/download?path={test_file}&token=invalid_bogus_token"
    )
    assert r.status_code == 401
    body = r.json()
    assert body.get("error") == "session_expired"
    assert body.get("reason") == "invalid_or_expired_session"


def test_gated_non_download_route_ignores_query_token(gated_client):
    """Non-download /api/* route with ?token=<valid token> ignores query token and returns 401 no_cookie."""
    client, valid_token = gated_client

    r = client.get(f"/api/sessions?token={valid_token}")
    assert r.status_code == 401
    body = r.json()
    assert body.get("error") == "unauthenticated"
    assert body.get("reason") == "no_cookie"


def test_loopback_download_query_token_regression(loopback_client, tmp_path, monkeypatch):
    """Loopback mode (auth_required=False) still validates _SESSION_TOKEN query token on /api/files/download."""
    client = loopback_client
    test_file = tmp_path / "sample.mp4"
    test_file.write_bytes(b"loopback sample content")

    session_token = getattr(web_server, "_SESSION_TOKEN", "test_session_token")
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", session_token)

    r = client.get(
        f"/api/files/download?path={test_file}&token={session_token}"
    )
    assert r.status_code == 200
    assert r.content == b"loopback sample content"


def test_bearer_takes_precedence_over_query_token(gated_client, tmp_path):
    """When Bearer header and query token are both present, Bearer header is evaluated first."""
    client, valid_token = gated_client
    test_file = tmp_path / "sample.mp4"
    test_file.write_bytes(b"content")

    # Case 1: Valid Bearer + invalid query token -> succeeds via Bearer (200)
    r1 = client.get(
        f"/api/files/download?path={test_file}&token=bogus_invalid",
        headers={"Authorization": f"Bearer {valid_token}"},
    )
    assert r1.status_code == 200

    # Case 2: Invalid Bearer + valid query token -> fails via Bearer (401 invalid_or_expired_session)
    r2 = client.get(
        f"/api/files/download?path={test_file}&token={valid_token}",
        headers={"Authorization": "Bearer bogus_invalid"},
    )
    assert r2.status_code == 401
    assert r2.json().get("reason") == "invalid_or_expired_session"


def test_provider_unreachable_returns_503(tmp_path):
    """When auth provider raises ProviderError during query token verification, returns 503."""
    clear_providers()
    register_provider(UnreachableAuthProvider())

    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)

    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True

    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    test_file = tmp_path / "sample.mp4"
    test_file.write_bytes(b"content")

    try:
        r = client.get(f"/api/files/download?path={test_file}&token=any_token")
        assert r.status_code == 503
        assert "unreachable" in r.json().get("detail", "")
    finally:
        clear_providers()
        web_server.app.state.bound_host = prev_host
        web_server.app.state.bound_port = prev_port
        web_server.app.state.auth_required = prev_required
