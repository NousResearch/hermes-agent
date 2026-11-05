"""The gated dashboard accepts the deployment session token in ``X-Hermes-Session-Token``.

Regression for #147387: Hermes Desktop's "Session token" connection sends its token in that
header on REST, but the gate read only a bearer token from the Authorization header, so every
REST call got 401 ``no_cookie`` even with the right token. Only the session-token provider may
verify the header; anything else there leaves the request to the cookie path as before.
"""
from __future__ import annotations

import secrets

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server, webapp_sign_in
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests
import plugins.dashboard_auth.session_token as session_token_plugin

_HEADER = "X-Hermes-Session-Token"


class _OtherLoginProvider(session_token_plugin.SessionTokenProvider):
    """Another session provider (stands in for a password/OAuth login's access tokens)."""

    name = "other-login"


@pytest.fixture
def token():
    return secrets.token_urlsafe(32)


@pytest.fixture
def other_token():
    return secrets.token_urlsafe(32)


@pytest.fixture
def gated(token, other_token):
    """``web_server.app`` in gated mode; the test chooses which providers to register."""
    _reset_for_tests()
    clear_providers()
    prev = {k: getattr(web_server.app.state, k, None)
            for k in ("bound_host", "bound_port", "auth_required")}
    web_server.app.state.bound_host = "dashboard.example"
    web_server.app.state.bound_port = 9119
    web_server.app.state.auth_required = True

    def _client(*, session_token_provider: bool = True):
        register_provider(_OtherLoginProvider(token=other_token))
        if session_token_provider:
            register_provider(session_token_plugin.SessionTokenProvider(token=token))
        return TestClient(web_server.app, base_url="https://dashboard.example")

    yield _client
    clear_providers()
    _reset_for_tests()
    for k, v in prev.items():
        setattr(web_server.app.state, k, v)


def test_deployment_token_in_session_header_is_authorized(gated, token):
    r = gated().get("/api/auth/me", headers={_HEADER: token})
    assert r.status_code == 200, (r.status_code, r.text)
    assert r.json().get("provider") == "session-token"


def test_wrong_session_header_token_is_401_as_without_it(gated):
    r = gated().get("/api/auth/me", headers={_HEADER: secrets.token_urlsafe(32)})
    assert r.status_code == 401
    assert r.json()["reason"] == "no_cookie"


def test_other_providers_token_in_session_header_is_401(gated, other_token):
    """The header is the session-token provider's path only: another provider's valid token,
    which works as a bearer, opens nothing there."""
    client = gated()
    assert client.get("/api/auth/me", headers={
        "Authorization": f"Bearer {other_token}"}).status_code == 200
    r = client.get("/api/auth/me", headers={_HEADER: other_token})
    assert r.status_code == 401
    assert r.json()["reason"] == "no_cookie"


def test_session_header_ignored_without_session_token_provider(gated, token):
    """Opted out (``dashboard.session_token_auth.enabled: false``): the provider is not
    registered and the header opens nothing, as before."""
    r = gated(session_token_provider=False).get("/api/auth/me", headers={_HEADER: token})
    assert r.status_code == 401
    assert r.json()["reason"] == "no_cookie"


def test_bearer_wins_over_session_header(gated, token, other_token):
    client = gated()
    r = client.get("/api/auth/me", headers={
        "Authorization": f"Bearer {secrets.token_urlsafe(32)}", _HEADER: token})
    assert r.status_code == 401
    assert r.json()["reason"] == "invalid_or_expired_session"
    r = client.get("/api/auth/me", headers={
        "Authorization": f"Bearer {other_token}", _HEADER: secrets.token_urlsafe(32)})
    assert r.status_code == 200
    assert r.json().get("provider") == "other-login"


def test_session_header_follows_the_webapps_bearer_rule(gated, token, monkeypatch):
    """Where a bearer token may not work alone (the gateway's Webapp), neither does the header."""
    client = gated()
    monkeypatch.setattr(webapp_sign_in, "token_works_alone", lambda request, tok: False)
    r = client.get("/api/auth/me", headers={_HEADER: token})
    assert r.status_code == 401
    assert r.json()["reason"] == "no_cookie"


def test_wrong_session_header_leaves_a_cookie_session_working(gated, other_token):
    client = gated()
    client.cookies.set("hermes_session_at", other_token)
    r = client.get("/api/auth/me", headers={_HEADER: secrets.token_urlsafe(32)})
    assert r.status_code == 200, (r.status_code, r.text)
    assert r.json().get("provider") == "other-login"


def test_valid_session_header_leaves_a_cookie_session_to_its_provider(gated, token, other_token):
    """A cookie session is served by the cookie's provider, as on main, even when the
    deployment token is also in the header."""
    client = gated()
    client.cookies.set("hermes_session_at", other_token)
    r = client.get("/api/auth/me", headers={_HEADER: token})
    assert r.status_code == 200, (r.status_code, r.text)
    assert r.json().get("provider") == "other-login"


def test_a_header_session_write_skips_the_cookie_write_checks(gated, token):
    """The header carries its credential itself, so like a bearer it is not held to the
    cookie-write checks, even when the browser says another page sent it."""
    r = gated().post("/api/auth/ws-ticket", headers={_HEADER: token, "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 200, (r.status_code, r.text)
    assert r.json()["ticket"]
    assert "set-cookie" not in r.headers
