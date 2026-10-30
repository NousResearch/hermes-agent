"""End-to-end gate tests for the session-token auth provider.

The bellona-shaped scenario: a non-loopback dashboard whose ONLY credential is the
provisioned ``HERMES_DASHBOARD_SESSION_TOKEN`` (the nix/agenix deployment story).
Before the session-token provider, that deployment hit the fail-closed gate, got
talked into basic auth by the interactive prompt, and its own token was rejected
on every gated leg (REST 401, ``/api/ws`` 403) — the desktop's "HTTP passes but the
WebSocket fails" incident. These tests pin the fixed behaviour:

  * the gate is satisfied with the token provider as the only registered provider,
    and still fails closed with none,
  * the REST bearer leg verifies the provisioned token,
  * the ``/api/ws`` upgrade accepts ``?token=<provisioned>`` and rejects anything else.
"""
from __future__ import annotations

import secrets
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
import hermes_cli.web_server_chat as _web_server_chat
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests
import plugins.dashboard_auth.session_token as session_token_plugin


@pytest.fixture
def token():
    return secrets.token_urlsafe(32)


@pytest.fixture
def gated_token_app(token):
    """web_server.app in gated mode with ONLY the session-token provider registered."""
    _reset_for_tests()
    clear_providers()
    register_provider(session_token_plugin.SessionTokenProvider(token=token))
    prev = {k: getattr(web_server.app.state, k, None)
            for k in ("bound_host", "bound_port", "auth_required")}
    web_server.app.state.bound_host = "bellona.ts.net"
    web_server.app.state.bound_port = 9119
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://bellona.ts.net")
    yield client
    clear_providers()
    _reset_for_tests()
    for k, v in prev.items():
        setattr(web_server.app.state, k, v)


def _fake_ws(*, query: dict, client_host: str = "203.0.113.7", path: str = "/api/ws"):
    class _QP:
        def __init__(self, q):
            self._q = q

        def get(self, k, default=""):
            return self._q.get(k, default)

    return SimpleNamespace(
        query_params=_QP(query),
        headers={},
        client=SimpleNamespace(host=client_host),
        url=SimpleNamespace(path=path),
    )


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


class TestGateEngagement:
    def test_token_provider_satisfies_the_fail_closed_gate(self, token):
        """A non-loopback bind with only the session-token provider starts."""
        clear_providers()
        register_provider(session_token_plugin.SessionTokenProvider(token=token))
        web_server._configure_auth_gate(
            "0.0.0.0", allow_public=False, ssh_session_token=None, ssh_owner_nonce=None)
        assert web_server.app.state.auth_required is True

    def test_no_providers_still_fails_closed(self, token):
        clear_providers()
        with pytest.raises(SystemExit):
            web_server._configure_auth_gate(
                "0.0.0.0", allow_public=False, ssh_session_token=None, ssh_owner_nonce=None)


# ---------------------------------------------------------------------------
# The two legs the desktop exercises
# ---------------------------------------------------------------------------


class TestRestBearerLeg:
    # /api/auth/me is the real identity route: an exact 200 + provider proves the bearer was
    # verified BY this provider (a mere "not 401" would also pass on an unrouted 404).
    def test_provisioned_token_passes_the_gate(self, gated_token_app, token):
        r = gated_token_app.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200, (r.status_code, r.text)
        assert r.json().get("provider") == "session-token"

    def test_foreign_token_is_401(self, gated_token_app, token):
        r = gated_token_app.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {secrets.token_urlsafe(32)}"})
        assert r.status_code == 401

    def test_the_desktops_stale_copy_is_401(self, gated_token_app, token):
        """The incident's trap: a token from BEFORE a secret rotation must not verify."""
        stale = secrets.token_urlsafe(32)
        assert stale != token
        r = gated_token_app.get("/api/auth/me", headers={"Authorization": f"Bearer {stale}"})
        assert r.status_code == 401


class TestWsUpgradeLeg:
    def test_provisioned_token_is_accepted(self, gated_token_app, token):
        reason, credential = _web_server_chat._ws_auth_reason(
            _fake_ws(query={"token": token}))
        assert reason is None
        assert credential == "token"

    def test_foreign_token_is_rejected(self, gated_token_app, token):
        reason, _ = _web_server_chat._ws_auth_reason(
            _fake_ws(query={"token": secrets.token_urlsafe(32)}))
        assert reason == "token_invalid"

    def test_no_credential_is_rejected(self, gated_token_app):
        reason, credential = _web_server_chat._ws_auth_reason(_fake_ws(query={}))
        assert reason == "no_credential"
        assert credential == "none"


# ---------------------------------------------------------------------------
# Login surfaces never select a provisioned-credential provider (review P1)
# ---------------------------------------------------------------------------


def _password_provider():
    """A real interactive password provider (protocol-compliant subclass) to sit beside the
    session-token provider; only its routing flags matter to these tests."""
    cls = type("StubPasswordProvider", (session_token_plugin.SessionTokenProvider,), {
        "name": "basic", "display_name": "Username & Password",
        "supports_password": True, "interactive_login": True})
    return cls(token=secrets.token_urlsafe(32))


_NATIVE = {
    "code_challenge": "x" * 43, "code_challenge_method": "S256",
    "redirect_uri": "http://127.0.0.1:53682/callback", "state": "client-state",
}


class TestNativeAuthorize:
    def _pending_count(self):
        from hermes_cli.dashboard_auth import native_flow
        return len(native_flow._pending)

    def test_unnamed_request_auto_selects_the_one_interactive_provider(self, gated_token_app):
        """session-token + one password provider: the password provider is the ONLY interactive
        one, so an unnamed request selects it (was: 404, because both were counted)."""
        register_provider(_password_provider())
        r = gated_token_app.get("/auth/native/authorize", params=_NATIVE, follow_redirects=False)
        assert r.status_code == 302, (r.status_code, r.text)
        assert r.headers["location"].endswith("/login")

    def test_token_only_deployment_never_reaches_start_login(self, gated_token_app):
        """Only session-token registered: no interactive provider, a clean 4xx — never the
        NotImplementedError from its start_login stub, and no pending broker state."""
        before = self._pending_count()
        r = gated_token_app.get("/auth/native/authorize", params=_NATIVE, follow_redirects=False)
        assert 400 <= r.status_code < 500, (r.status_code, r.text)
        assert self._pending_count() == before

    def test_explicitly_named_token_provider_is_rejected_before_pending_state(
            self, gated_token_app):
        before = self._pending_count()
        r = gated_token_app.get(
            "/auth/native/authorize", params={**_NATIVE, "provider": "session-token"},
            follow_redirects=False)
        assert r.status_code == 400, (r.status_code, r.text)
        assert self._pending_count() == before

    def test_browser_login_route_rejects_the_token_provider(self, gated_token_app):
        r = gated_token_app.get(
            "/auth/login", params={"provider": "session-token"}, follow_redirects=False)
        assert r.status_code == 404


class TestStatusAdvertisement:
    def _flows(self, client):
        return client.get("/api/status").json()["auth_flows"]

    def test_token_only_deployment_does_not_advertise_native_pkce(self, gated_token_app):
        """Desktop picks the native browser flow off native_pkce; a token-only gate has no
        browser sign-in, so advertising it would send the desktop down a dead end."""
        flows = self._flows(gated_token_app)
        assert "cookie" in flows
        assert "native_pkce" not in flows

    def test_native_pkce_returns_with_an_interactive_provider(self, gated_token_app):
        register_provider(_password_provider())
        assert "native_pkce" in self._flows(gated_token_app)
