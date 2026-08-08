"""Tests for POST /api/auth/revoke (issue #76706).

The revoke endpoint is the auth-required JSON counterpart of
``/auth/logout``: it drives the providers' server-side revocation (for the
basic provider, a persisted revocation set keyed by token ``jti``) so a
captured refresh token can be killed without waiting out its sliding TTL.

``revoke_all_sessions`` is a provider opt-in capability; providers without
it fall back to revoking the caller's own refresh token (the same
best-effort loop ``/auth/logout`` already runs).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture
def gated_app():
    """Configure web_server.app for gated mode + register the stub provider."""
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


def _complete_stub_login(client) -> None:
    """Walk the stub OAuth round trip so ``client`` carries a valid session."""
    r1 = client.get("/auth/login?provider=stub", follow_redirects=False)
    assert r1.status_code == 302
    state = r1.headers["location"].split("state=")[1]
    r2 = client.get(
        f"/auth/callback?code=stub_code&state={state}",
        follow_redirects=False,
    )
    assert r2.status_code == 302


class _RevokeTrackingProvider(StubAuthProvider):
    """Stub that records revoke calls so tests can assert what ran."""

    def __init__(self):
        super().__init__()
        self.revoked_tokens: list[str] = []
        self.revoke_all_calls = 0

    def revoke_session(self, *, refresh_token: str) -> None:
        self.revoked_tokens.append(refresh_token)

    def revoke_all_sessions(self) -> None:
        self.revoke_all_calls += 1


def test_revoke_requires_auth(gated_app):
    """No session cookie → the gate rejects the request with 401."""
    r = gated_app.post("/api/auth/revoke", json={"all": True})
    assert r.status_code == 401


def test_revoke_all_success_and_clears_cookies(gated_app):
    """{all: true} succeeds and drops the caller's own session cookies.

    The plain stub has no ``revoke_all_sessions``, so the endpoint falls
    back to revoking the caller's own refresh token — the same best-effort
    loop /auth/logout runs. Either way the caller's session is gone and the
    cookies must be cleared so the SPA lands on /login cleanly.
    """
    _complete_stub_login(gated_app)
    r = gated_app.post("/api/auth/revoke", json={"all": True})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["revoked_all"] is True
    assert body["providers"] == {"stub": "ok"}

    set_cookie = r.headers.get("set-cookie", "")
    assert "hermes_session_rt=" in set_cookie, (
        "revoke-all must clear the caller's refresh-token cookie; "
        f"Set-Cookie was: {set_cookie!r}"
    )


def test_revoke_all_calls_provider_revoke_all(gated_app):
    """Providers that opt into ``revoke_all_sessions`` get called with it."""
    clear_providers()
    tracking = _RevokeTrackingProvider()
    register_provider(tracking)
    _complete_stub_login(gated_app)

    r = gated_app.post("/api/auth/revoke", json={"all": True})
    assert r.status_code == 200
    assert tracking.revoke_all_calls == 1
    assert tracking.revoked_tokens == [], (
        "revoke-all must use revoke_all_sessions, not per-token revoke"
    )


def test_revoke_specific_refresh_token(gated_app):
    """{refresh_token: ...} revokes just that lineage, cookies untouched."""
    clear_providers()
    tracking = _RevokeTrackingProvider()
    register_provider(tracking)
    _complete_stub_login(gated_app)

    r = gated_app.post(
        "/api/auth/revoke", json={"refresh_token": "captured-token-abc"}
    )
    assert r.status_code == 200
    assert tracking.revoked_tokens == ["captured-token-abc"]
    body = r.json()
    assert body["revoked_all"] is False
    assert "hermes_session_rt=" not in r.headers.get("set-cookie", ""), (
        "revoking a different session must not clear the caller's cookies"
    )


def test_revoke_without_body_revokes_own_cookie(gated_app):
    """No body → the caller's own refresh-token cookie is revoked."""
    _complete_stub_login(gated_app)
    r = gated_app.post("/api/auth/revoke")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    # Own session revoked → cookies cleared (same as logout).
    assert "hermes_session_rt=" in r.headers.get("set-cookie", "")
