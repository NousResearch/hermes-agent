"""End-to-end + unit tests for the RFC 8252 native-app (desktop loopback) login.

Covers the gateway-side broker that lets the desktop app authenticate via the
user's SYSTEM browser and hold the tokens itself (``Authorization: Bearer``)
instead of the embedded-webview + HttpOnly-cookie flow:

  * ``native_auth`` broker unit behaviour (loopback validation, PKCE verify,
    single-use code, TTL expiry).
  * Full ``/auth/native/start`` → ``/auth/callback`` → ``/auth/native/token``
    round trip against the in-process ``StubAuthProvider``.
  * The desktop-held bearer authenticates a gated REST route (``/api/auth/me``)
    and mints a ws-ticket (``/api/auth/ws-ticket``) — identically to a cookie
    session — while an invalid bearer is rejected 401.
  * The cookieless ``/api/auth/refresh`` rotates tokens.
  * ``/api/status`` advertises ``native_loopback_auth`` so the desktop can
    choose the flow vs. the embedded-webview fallback.

Uses ``StubAuthProvider`` so the OAuth round trip completes in-process with no
external IDP. The stub's ``start_login`` bounces straight back to the callback
with ``code=stub_code``, which is exactly what the broker needs.
"""
from __future__ import annotations

import base64
import hashlib
import time
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import (
    clear_providers,
    register_provider,
)
from hermes_cli.dashboard_auth import native_auth
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


# ---------------------------------------------------------------------------
# PKCE helper (desktop side)
# ---------------------------------------------------------------------------


def _pkce_pair() -> tuple[str, str]:
    """Return ``(verifier, challenge)`` — S256, matching native_auth's verify."""
    verifier = base64.urlsafe_b64encode(b"x" * 48).rstrip(b"=").decode()
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    return verifier, challenge


# ===========================================================================
# native_auth broker — unit
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_broker():
    native_auth._reset_for_tests()
    yield
    native_auth._reset_for_tests()


@pytest.mark.parametrize(
    "uri,ok",
    [
        ("http://127.0.0.1:8765/callback", True),
        ("http://localhost:8765/callback", True),
        ("http://[::1]:8765/callback", True),
        ("https://127.0.0.1:8765/callback", False),  # https not allowed for loopback
        ("http://evil.example.com/callback", False),  # public host
        ("http://127.0.0.1:8765", False),  # no path
        ("hermes://callback", False),  # custom scheme
        ("http://10.0.0.5/callback", False),  # private but not loopback
        ("not a url", False),
    ],
)
def test_is_loopback_redirect_uri(uri, ok):
    assert native_auth.is_loopback_redirect_uri(uri) is ok


def test_start_broker_rejects_non_s256():
    _, challenge = _pkce_pair()
    with pytest.raises(native_auth.BrokerError):
        native_auth.start_broker(
            upstream_state="us",
            upstream_verifier="uv",
            provider="stub",
            code_challenge=challenge,
            code_challenge_method="plain",
            redirect_uri="http://127.0.0.1:9/cb",
            desktop_state="ds",
        )


def test_start_broker_rejects_non_loopback_redirect():
    _, challenge = _pkce_pair()
    with pytest.raises(native_auth.BrokerError):
        native_auth.start_broker(
            upstream_state="us",
            upstream_verifier="uv",
            provider="stub",
            code_challenge=challenge,
            code_challenge_method="S256",
            redirect_uri="https://evil.example.com/cb",
            desktop_state="ds",
        )


def _mk_session():
    p = StubAuthProvider()
    ls = p.start_login(redirect_uri="https://gw/auth/callback")
    pkce = dict(
        item.split("=", 1)
        for item in ls.cookie_payload["hermes_session_pkce"].split(";")
    )
    return p.complete_login(
        code="stub_code",
        state=pkce["state"],
        code_verifier=pkce["verifier"],
        redirect_uri="https://gw/auth/callback",
    )


def test_redeem_happy_path():
    verifier, challenge = _pkce_pair()
    native_auth.start_broker(
        upstream_state="us1",
        upstream_verifier="uv",
        provider="stub",
        code_challenge=challenge,
        code_challenge_method="S256",
        redirect_uri="http://127.0.0.1:9/cb",
        desktop_state="ds",
    )
    session = _mk_session()
    code = native_auth.complete_broker("us1", session)
    got = native_auth.redeem_code(
        code=code, code_verifier=verifier, redirect_uri="http://127.0.0.1:9/cb"
    )
    assert got.user_id == session.user_id
    assert got.access_token == session.access_token


def test_redeem_pkce_mismatch_rejected_and_code_burned():
    _, challenge = _pkce_pair()
    native_auth.start_broker(
        upstream_state="us2",
        upstream_verifier="uv",
        provider="stub",
        code_challenge=challenge,
        code_challenge_method="S256",
        redirect_uri="http://127.0.0.1:9/cb",
        desktop_state="ds",
    )
    code = native_auth.complete_broker("us2", _mk_session())
    # Wrong verifier → invalid_grant.
    with pytest.raises(native_auth.BrokerError):
        native_auth.redeem_code(
            code=code,
            code_verifier="wrong-verifier",
            redirect_uri="http://127.0.0.1:9/cb",
        )
    # And the code is burned — a subsequent correct attempt also fails.
    verifier, _ = _pkce_pair()
    with pytest.raises(native_auth.BrokerError):
        native_auth.redeem_code(
            code=code, code_verifier=verifier, redirect_uri="http://127.0.0.1:9/cb"
        )


def test_redeem_is_single_use():
    verifier, challenge = _pkce_pair()
    native_auth.start_broker(
        upstream_state="us3",
        upstream_verifier="uv",
        provider="stub",
        code_challenge=challenge,
        code_challenge_method="S256",
        redirect_uri="http://127.0.0.1:9/cb",
        desktop_state="ds",
    )
    code = native_auth.complete_broker("us3", _mk_session())
    native_auth.redeem_code(
        code=code, code_verifier=verifier, redirect_uri="http://127.0.0.1:9/cb"
    )
    with pytest.raises(native_auth.BrokerError):
        native_auth.redeem_code(
            code=code, code_verifier=verifier, redirect_uri="http://127.0.0.1:9/cb"
        )


def test_redeem_redirect_uri_mismatch_rejected():
    verifier, challenge = _pkce_pair()
    native_auth.start_broker(
        upstream_state="us4",
        upstream_verifier="uv",
        provider="stub",
        code_challenge=challenge,
        code_challenge_method="S256",
        redirect_uri="http://127.0.0.1:9/cb",
        desktop_state="ds",
    )
    code = native_auth.complete_broker("us4", _mk_session())
    with pytest.raises(native_auth.BrokerError):
        native_auth.redeem_code(
            code=code,
            code_verifier=verifier,
            redirect_uri="http://127.0.0.1:9999/cb",  # different port
        )


def test_broker_record_expiry(monkeypatch):
    _, challenge = _pkce_pair()
    native_auth.start_broker(
        upstream_state="us5",
        upstream_verifier="uv",
        provider="stub",
        code_challenge=challenge,
        code_challenge_method="S256",
        redirect_uri="http://127.0.0.1:9/cb",
        desktop_state="ds",
    )
    # Jump past the broker TTL — get_broker evicts on read.
    real_time = time.time
    monkeypatch.setattr(
        native_auth.time, "time",
        lambda: real_time() + native_auth.BROKER_TTL_SECONDS + 5,
    )
    assert native_auth.get_broker("us5") is None


# ===========================================================================
# HTTP round trip against the gated app
# ===========================================================================


@pytest.fixture
def gated_app():
    """web_server.app in gated mode with the stub OAuth provider registered."""
    clear_providers()
    register_provider(StubAuthProvider())
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "gw.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    native_auth._reset_for_tests()
    client = TestClient(web_server.app, base_url="https://gw.fly.dev")
    yield client
    clear_providers()
    native_auth._reset_for_tests()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def _run_native_flow(client) -> dict:
    """Drive start → callback → token; return the token JSON the desktop gets."""
    verifier, challenge = _pkce_pair()
    redirect_uri = "http://127.0.0.1:8765/callback"
    desktop_state = "desktop-state-xyz"

    # 1) Desktop POSTs /auth/native/start (its own HTTP client, no cookie).
    r_start = client.post(
        "/auth/native/start",
        json={
            "provider": "stub",
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": desktop_state,
        },
    )
    assert r_start.status_code == 200, r_start.text
    authorization_url = r_start.json()["authorization_url"]

    # The stub's authorize URL bounces straight back to the gateway callback
    # with ?code=stub_code&state=<upstream_state>. Extract those params and
    # hit /auth/callback the way the system browser would (NO pkce cookie).
    q = parse_qs(urlparse(authorization_url).query)
    upstream_code = q["code"][0]
    upstream_state = q["state"][0]

    # 2) System browser → /auth/callback. The broker recognises the login by
    #    upstream state and 302s to the desktop's loopback with a one-time code.
    r_cb = client.get(
        f"/auth/callback?code={upstream_code}&state={upstream_state}",
        follow_redirects=False,
    )
    assert r_cb.status_code == 302, r_cb.text
    loc = r_cb.headers["location"]
    assert loc.startswith(redirect_uri), loc
    # No session cookie is set on the native callback (cookieless flow).
    assert "set-cookie" not in {k.lower() for k in r_cb.headers.keys()}
    cb_q = parse_qs(urlparse(loc).query)
    assert cb_q["state"][0] == desktop_state  # desktop CSRF check
    one_time_code = cb_q["code"][0]

    # 3) Desktop redeems the code + its PKCE verifier for JSON tokens.
    r_tok = client.post(
        "/auth/native/token",
        json={
            "code": one_time_code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        },
    )
    assert r_tok.status_code == 200, r_tok.text
    return r_tok.json()


def test_full_native_round_trip_returns_json_tokens(gated_app):
    tokens = _run_native_flow(gated_app)
    assert tokens["token_type"] == "Bearer"
    assert tokens["access_token"]
    assert tokens["refresh_token"]
    assert tokens["user_id"] == "stub-user-1"
    assert tokens["provider"] == "stub"
    assert isinstance(tokens["expires_at"], int)


def test_bearer_unlocks_gated_api_me(gated_app):
    """The desktop-held bearer authenticates a gated route with NO cookie."""
    tokens = _run_native_flow(gated_app)
    r = gated_app.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert r.status_code == 200, r.text
    me = r.json()
    assert me["user_id"] == "stub-user-1"
    assert me["provider"] == "stub"


def test_bearer_mints_ws_ticket(gated_app):
    """POST /api/auth/ws-ticket works under a bearer exactly like under a cookie."""
    tokens = _run_native_flow(gated_app)
    r = gated_app.post(
        "/api/auth/ws-ticket",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert r.status_code == 200, r.text
    assert r.json().get("ticket")


def test_invalid_bearer_is_401(gated_app):
    r = gated_app.get(
        "/api/auth/me",
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert r.status_code == 401
    assert r.json().get("error") == "session_expired"


def test_native_start_rejects_non_loopback_redirect(gated_app):
    _, challenge = _pkce_pair()
    r = gated_app.post(
        "/auth/native/start",
        json={
            "provider": "stub",
            "redirect_uri": "https://evil.example.com/cb",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": "s",
        },
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_request"


def test_native_start_unknown_provider_404(gated_app):
    _, challenge = _pkce_pair()
    r = gated_app.post(
        "/auth/native/start",
        json={
            "provider": "nope",
            "redirect_uri": "http://127.0.0.1:9/cb",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": "s",
        },
    )
    assert r.status_code == 404


def test_native_token_rejects_bad_code(gated_app):
    verifier, _ = _pkce_pair()
    r = gated_app.post(
        "/auth/native/token",
        json={
            "code": "never-issued",
            "code_verifier": verifier,
            "redirect_uri": "http://127.0.0.1:9/cb",
        },
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_grant"


def test_cookieless_refresh_rotates_tokens(gated_app):
    tokens = _run_native_flow(gated_app)
    r = gated_app.post(
        "/api/auth/refresh",
        json={"refresh_token": tokens["refresh_token"], "provider": "stub"},
    )
    assert r.status_code == 200, r.text
    rotated = r.json()
    assert rotated["access_token"]
    assert rotated["user_id"] == "stub-user-1"
    # The rotated access token verifies against a gated route.
    r2 = gated_app.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {rotated['access_token']}"},
    )
    assert r2.status_code == 200


def test_refresh_bad_token_401(gated_app):
    r = gated_app.post(
        "/api/auth/refresh",
        json={"refresh_token": "garbage", "provider": "stub"},
    )
    assert r.status_code == 401
    assert r.json()["error"] == "invalid_grant"


def test_status_advertises_native_loopback_capability(gated_app):
    r = gated_app.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["auth_required"] is True
    assert body["native_loopback_auth"] is True


def test_status_native_capability_false_in_loopback_mode():
    """Loopback (no gate) → native_loopback_auth is False."""
    clear_providers()
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    web_server.app.state.auth_required = False
    try:
        client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
        body = client.get("/api/status").json()
        assert body["auth_required"] is False
        assert body["native_loopback_auth"] is False
    finally:
        web_server.app.state.bound_host = prev_host
        web_server.app.state.bound_port = prev_port
        web_server.app.state.auth_required = prev_required
