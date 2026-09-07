"""E2E + unit tests for the RFC 8252 native-app (system-browser + loopback +
PKCE) dashboard-auth flow.

Covers:
  * ``native_flow`` broker unit behaviour — PKCE binding, single-use codes,
    expiry, capacity, replay resistance.
  * The full ``/auth/native/authorize`` → ``/auth/callback`` →
    ``/auth/native/token`` round trip in-process against ``StubAuthProvider``.
  * ``/api/status`` capability advertisement (``auth_flows``).
  * Cookieless bearer authentication of a gated route (the whole point of the
    feature — a desktop authenticates REST with ``Authorization: Bearer`` and
    sets/needs no cookie).
  * ``/auth/native/refresh`` token rotation and terminal-expiry semantics.

Run: pytest tests/hermes_cli/test_dashboard_auth_native_flow.py
"""

from __future__ import annotations

import hashlib
import base64
import time
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import (
    clear_providers,
    register_provider,
)
from hermes_cli.dashboard_auth import native_flow
from hermes_cli.dashboard_auth.base import Session
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


# ---------------------------------------------------------------------------
# PKCE helpers (desktop side)
# ---------------------------------------------------------------------------


def _b64url_no_pad(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _make_pkce() -> tuple[str, str]:
    """Return ``(verifier, challenge)`` — the desktop's PKCE pair."""
    verifier = _b64url_no_pad(b"desktop-verifier-secret-material-0123456789abcd")
    challenge = _b64url_no_pad(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


class _PasswordOnlyProvider(StubAuthProvider):
    """Mirrors the bundled ``basic`` provider's flags: a session provider
    (``supports_session`` defaults True) that authenticates by username +
    password and can never be the target of the native OAuth broker flow.
    ``start_login`` raises to prove the route must reject it before ever
    attempting a redirect."""

    name = "pwonly"
    display_name = "Password Only (test)"
    supports_password = True

    def start_login(self, *, redirect_uri):
        raise AssertionError(
            "native authorize must reject a password provider before "
            "calling start_login"
        )


class _SecondStubProvider(StubAuthProvider):
    """A second brokerable OAuth provider, so tests can create an ambiguous
    multi-provider deployment."""

    name = "stub2"
    display_name = "Stub IdP Two (test only)"


# ---------------------------------------------------------------------------
# native_flow broker unit tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_broker():
    native_flow._reset_for_tests()
    # Snapshot the shared app.state auth fields + provider registry so a test
    # that flips auth_required / registers a stub provider can't leak into a
    # later test file (e.g. the MCP dashboard-oauth suite shares web_server.app).
    prev_required = getattr(web_server.app.state, "auth_required", None)
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    yield
    native_flow._reset_for_tests()
    clear_providers()
    web_server.app.state.auth_required = prev_required
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port


def _stub_session(exp_offset: int = 3600) -> Session:
    now = int(time.time())
    return Session(
        user_id="u1",
        email="u1@example.test",
        display_name="U One",
        org_id="org1",
        provider="stub",
        expires_at=now + exp_offset,
        access_token="at-opaque",
        refresh_token="rt-opaque",
    )








# ---------------------------------------------------------------------------
# Route-level E2E against StubAuthProvider
# ---------------------------------------------------------------------------


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
    # follow_redirects=False so we can inspect each 302 leg of the flow.
    client = TestClient(
        web_server.app, base_url="https://fly-app.fly.dev",
        follow_redirects=False,
    )
    yield client
    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def _walk_native_login(client, *, redirect_uri, challenge, state="cli-state"):
    """Drive authorize → (stub redirects to callback) → loopback code.

    Returns the ``code`` + ``state`` the gateway put on the loopback redirect.
    """
    # 1. Desktop opens the system browser at /auth/native/authorize.
    r = client.get(
        "/auth/native/authorize",
        params={
            "provider": "stub",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "redirect_uri": redirect_uri,
            "state": state,
        },
    )
    assert r.status_code == 302, r.text
    # Stub's start_login redirects straight to /auth/callback?code=stub_code.
    loc = r.headers["location"]
    parsed = urlparse(loc)
    cb_qs = parse_qs(parsed.query)
    # Carry the gateway PKCE cookie forward (holds broker_state + verifier).
    cookies = r.cookies
    # 2. Browser hits the gateway callback.
    r2 = client.get(
        "/auth/callback",
        params={"code": cb_qs["code"][0], "state": cb_qs["state"][0]},
        cookies=cookies,
    )
    assert r2.status_code == 302, r2.text
    # 3. The callback 302s to the desktop's loopback redirect_uri.
    loop = urlparse(r2.headers["location"])
    assert f"{loop.scheme}://{loop.netloc}" == redirect_uri.rsplit("/", 1)[0] or \
        loop.netloc in redirect_uri
    loop_qs = parse_qs(loop.query)
    # No session cookie must be set on the native callback response.
    set_cookie = r2.headers.get("set-cookie", "")
    assert "hermes_session_at" not in set_cookie, (
        f"native callback must NOT set a session cookie; got {set_cookie!r}"
    )
    return loop_qs["code"][0], loop_qs["state"][0]




def test_native_authorize_rejects_non_loopback_redirect(gated_client):
    _verifier, challenge = _make_pkce()
    r = gated_client.get(
        "/auth/native/authorize",
        params={
            "provider": "stub",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "redirect_uri": "https://evil.example.com/steal",
            "state": "s",
        },
    )
    assert r.status_code == 400
    assert "loopback" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Empty-provider auto-select (the desktop omits ``provider``; the gateway
# picks when there is exactly one brokerable candidate) — regression #78906
# ---------------------------------------------------------------------------


def _native_authorize_params(challenge, **overrides):
    params = {
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "redirect_uri": "http://127.0.0.1:53999/cb",
        "state": "s",
    }
    params.update(overrides)
    return params


def test_native_authorize_empty_provider_auto_selects_oauth_with_password_also_registered(
    gated_client,
):
    """Regression for #78906: a password provider is a session provider but
    can never be the target of the native OAuth broker flow, so it must not
    count toward the empty-provider auto-select. With one OAuth provider +
    one password provider (the normal SSO-with-password-fallback setup) the
    desktop's empty-provider request must auto-select the OAuth provider
    (302), not fail with ``Unknown provider: ''`` (404)."""
    register_provider(_PasswordOnlyProvider())
    _verifier, challenge = _make_pkce()
    r = gated_client.get(
        "/auth/native/authorize",
        params=_native_authorize_params(challenge),
    )
    assert r.status_code == 302, r.text
    assert "code=stub_code" in r.headers["location"]


def test_native_authorize_empty_provider_auto_selects_single_oauth(gated_client):
    """The common hosted case: exactly one brokerable provider; an empty
    ``provider`` auto-selects it (302), so the desktop needn't hardcode the
    name."""
    _verifier, challenge = _make_pkce()
    r = gated_client.get(
        "/auth/native/authorize",
        params=_native_authorize_params(challenge),
    )
    assert r.status_code == 302, r.text
    assert "code=stub_code" in r.headers["location"]


def test_native_authorize_empty_provider_ambiguous_multiple_oauth_404(gated_client):
    """Two brokerable providers: the empty-provider convenience cannot pick
    unambiguously, so the request still fails — the desktop must pass
    ``?provider=`` explicitly."""
    register_provider(_SecondStubProvider())
    _verifier, challenge = _make_pkce()
    r = gated_client.get(
        "/auth/native/authorize",
        params=_native_authorize_params(challenge),
    )
    assert r.status_code == 404


def test_native_authorize_empty_provider_password_only_brokers_to_login(
    gated_client,
):
    """Password-only deployment: an empty ``provider`` selects the lone
    session provider and — now that native sign-in brokers password
    providers through the system browser — 302s to ``/login`` with the
    broker in the PKCE cookie, rather than the old 400."""
    clear_providers()
    register_provider(_PasswordOnlyProvider())
    _verifier, challenge = _make_pkce()
    r = gated_client.get(
        "/auth/native/authorize",
        params=_native_authorize_params(challenge),
    )
    assert r.status_code == 302, r.text
    assert r.headers["location"].endswith("/login")
    set_cookie = r.headers.get("set-cookie", "")
    # The PKCE cookie value is URL-encoded on the wire; decode through
    # the real reader inverse before asserting the broker handle rides
    # in it.
    from hermes_cli.dashboard_auth.cookies import parse_pkce_payload
    wire_value = set_cookie.split("=", 1)[1].split(";", 1)[0]
    assert "broker" in parse_pkce_payload(wire_value)


# ---------------------------------------------------------------------------
# Cookieless bearer auth of a gated route — the core deliverable
# ---------------------------------------------------------------------------


def test_bearer_authenticates_gated_route_without_cookie(gated_client):
    """A desktop that redeemed tokens can call a gated route with only an
    ``Authorization: Bearer`` header — no cookie in the jar."""
    verifier, challenge = _make_pkce()
    code, _state = _walk_native_login(
        gated_client, redirect_uri="http://127.0.0.1:53999/cb",
        challenge=challenge,
    )
    tokens = gated_client.post(
        "/auth/native/token",
        json={"code": code, "code_verifier": verifier},
    ).json()
    at = tokens["access_token"]

    # /api/auth/me is gated; a cookieless request with the bearer must pass
    # and identify the user.
    r = gated_client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {at}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["user_id"] == "stub-user-1"




# ---------------------------------------------------------------------------
# Capability advertisement on /api/status
# ---------------------------------------------------------------------------




def test_status_loopback_mode_has_no_auth_flows():
    clear_providers()
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    try:
        client = TestClient(web_server.app, base_url="http://127.0.0.1:8080")
        body = client.get("/api/status").json()
        assert body["auth_required"] is False
        assert body["auth_flows"] == []
    finally:
        web_server.app.state.auth_required = prev_required


# ---------------------------------------------------------------------------
# Native flow for password providers (system-browser autofill path)
# ---------------------------------------------------------------------------
#
# A password provider has no IDP round trip, but the native flow still buys
# the desktop the one thing an embedded webview can never have: the system
# browser's OS-password-manager autofill. /auth/native/authorize lands the
# browser on /login (broker_state in the PKCE cookie) and a successful
# /auth/password-login completes the pending authorization exactly like the
# OAuth callback does.


@pytest.fixture
def pw_gated_client():
    from hermes_cli.dashboard_auth.routes import _reset_password_rate_limit
    from tests.hermes_cli.test_dashboard_auth_password_login import (
        PasswordProvider,
    )

    clear_providers()
    register_provider(PasswordProvider())
    _reset_password_rate_limit()
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(
        web_server.app, base_url="https://fly-app.fly.dev",
        follow_redirects=False,
    )
    yield client
    clear_providers()
    _reset_password_rate_limit()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def test_status_advertises_native_pkce_for_password_only_gateway(
    pw_gated_client,
):
    body = pw_gated_client.get("/api/status").json()
    assert body["auth_required"] is True
    assert "cookie" in body["auth_flows"]
    assert "native_pkce" in body["auth_flows"]


def test_native_authorize_password_provider_redirects_to_login(
    pw_gated_client,
):
    """Empty ``provider`` auto-picks the single password provider and lands
    the system browser on /login with the broker in the PKCE cookie."""
    _verifier, challenge = _make_pkce()
    r = pw_gated_client.get(
        "/auth/native/authorize",
        params={
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "redirect_uri": "http://127.0.0.1:53999/cb",
            "state": "desk-state",
        },
    )
    assert r.status_code == 302, r.text
    assert r.headers["location"].endswith("/login")
    set_cookie = r.headers.get("set-cookie", "")
    assert "pkce" in set_cookie
    # Wire value is URL-encoded; decode through the reader inverse.
    from hermes_cli.dashboard_auth.cookies import parse_pkce_payload
    wire_value = set_cookie.split("=", 1)[1].split(";", 1)[0]
    assert "broker" in parse_pkce_payload(wire_value)


def _start_native_password_login(client, *, challenge, state="desk-state"):
    r = client.get(
        "/auth/native/authorize",
        params={
            "provider": "testpw",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "redirect_uri": "http://127.0.0.1:53999/cb",
            "state": state,
        },
    )
    assert r.status_code == 302, r.text
    return r.cookies


def test_native_password_login_full_roundtrip(pw_gated_client):
    """authorize → /login → password-login → loopback code → bearer tokens."""
    verifier, challenge = _make_pkce()
    cookies = _start_native_password_login(pw_gated_client, challenge=challenge)

    # The browser form POSTs the credentials; the PKCE cookie rides along.
    r = pw_gated_client.post(
        "/auth/password-login",
        json={"provider": "testpw", "username": "admin", "password": "hunter2"},
        cookies=cookies,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    # ``next`` is the desktop's loopback redirect carrying code + state —
    # NOT a dashboard path.
    assert body["next"].startswith("http://127.0.0.1:53999/cb?")
    qs = parse_qs(urlparse(body["next"]).query)
    assert qs["state"][0] == "desk-state"
    code = qs["code"][0]
    # No browser session on the native branch; the PKCE cookie is cleared.
    set_cookie = r.headers.get("set-cookie", "")
    assert "hermes_session_at" not in set_cookie, (
        f"native password login must NOT set a session cookie; got {set_cookie!r}"
    )
    assert "pkce" in set_cookie  # the clearing Set-Cookie

    # Desktop redeems the loopback code with its PKCE verifier.
    tokens = pw_gated_client.post(
        "/auth/native/token",
        json={"code": code, "code_verifier": verifier},
    ).json()
    assert tokens["provider"] == "testpw"
    assert tokens["user_id"] == "admin"

    # Cookieless bearer auth of a gated route — the point of the flow.
    r2 = pw_gated_client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["user_id"] == "admin"


def test_native_password_login_wrong_password_keeps_pending(pw_gated_client):
    """A failed credential attempt must not consume the pending
    authorization — the user retypes and succeeds on the same broker."""
    verifier, challenge = _make_pkce()
    cookies = _start_native_password_login(pw_gated_client, challenge=challenge)

    r = pw_gated_client.post(
        "/auth/password-login",
        json={"provider": "testpw", "username": "admin", "password": "wrong"},
        cookies=cookies,
    )
    assert r.status_code == 401

    r2 = pw_gated_client.post(
        "/auth/password-login",
        json={"provider": "testpw", "username": "admin", "password": "hunter2"},
        cookies=cookies,
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["next"].startswith("http://127.0.0.1:53999/cb?")


def test_native_password_login_expired_broker_returns_400(pw_gated_client):
    """A broker cookie whose pending entry lapsed (TTL) is a clean 400
    telling the user to restart sign-in — never a silent cookie login."""
    _verifier, challenge = _make_pkce()
    cookies = _start_native_password_login(pw_gated_client, challenge=challenge)

    native_flow._reset_for_tests()  # simulate the pending TTL lapsing

    r = pw_gated_client.post(
        "/auth/password-login",
        json={"provider": "testpw", "username": "admin", "password": "hunter2"},
        cookies=cookies,
    )
    assert r.status_code == 400
    assert "restart" in r.json()["detail"].lower()


def test_native_password_login_rejects_cross_provider_completion(
    pw_gated_client,
):
    """A native flow started for provider A must not be completable with
    provider B's credentials: /login renders every provider's form, and the
    pending authorization is bound to the provider recorded in the
    server-set PKCE cookie. The mismatch is rejected BEFORE credential
    verification and preserves the pending entry, so the user can still
    submit the form the flow was started for."""
    from tests.hermes_cli.test_dashboard_auth_password_login import (
        PasswordProvider,
    )

    class SecondPasswordProvider(PasswordProvider):
        name = "testpw2"
        display_name = "Test Password 2"

    register_provider(SecondPasswordProvider())

    verifier, challenge = _make_pkce()
    # Native flow initiated for provider A ("testpw").
    cookies = _start_native_password_login(pw_gated_client, challenge=challenge)

    # Valid credentials for provider B ("testpw2") must NOT complete A's
    # pending authorization.
    r = pw_gated_client.post(
        "/auth/password-login",
        json={
            "provider": "testpw2", "username": "admin", "password": "hunter2",
        },
        cookies=cookies,
    )
    assert r.status_code == 400, r.text
    assert "different provider" in r.json()["detail"]
    set_cookie = r.headers.get("set-cookie", "")
    assert "hermes_session_at" not in set_cookie

    # The pending entry survived — provider A completes normally.
    r2 = pw_gated_client.post(
        "/auth/password-login",
        json={
            "provider": "testpw", "username": "admin", "password": "hunter2",
        },
        cookies=cookies,
    )
    assert r2.status_code == 200, r2.text
    qs = parse_qs(urlparse(r2.json()["next"]).query)
    tokens = pw_gated_client.post(
        "/auth/native/token",
        json={"code": qs["code"][0], "code_verifier": verifier},
    ).json()
    assert tokens["provider"] == "testpw"


def test_password_login_without_broker_still_mints_cookies(pw_gated_client):
    """Guard: an ordinary browser password login (no native broker cookie)
    keeps the existing cookie-minting behaviour."""
    r = pw_gated_client.post(
        "/auth/password-login",
        json={"provider": "testpw", "username": "admin", "password": "hunter2"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["next"] == "/"
    set_cookie = r.headers.get("set-cookie", "")
    assert "hermes_session_at" in set_cookie


# ---------------------------------------------------------------------------
# Native refresh
# ---------------------------------------------------------------------------


def test_native_refresh_dead_token_returns_401(gated_client):
    r = gated_client.post(
        "/auth/native/refresh",
        json={"refresh_token": "garbage-not-a-real-rt", "provider": "stub"},
    )
    assert r.status_code == 401
    assert r.json()["error"] == "session_expired"
