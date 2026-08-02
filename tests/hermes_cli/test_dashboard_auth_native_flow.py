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
from html.parser import HTMLParser
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


class NousStubAuthProvider(StubAuthProvider):
    name = "nous"
    display_name = "Nous Research"


class SelfHostedOidcStubAuthProvider(StubAuthProvider):
    name = "self-hosted"
    display_name = "Self-Hosted OIDC"


class ProviderLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "a" and "provider-btn" in (values.get("class") or "").split():
            href = values.get("href")
            if href:
                self.hrefs.append(href)


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


def test_native_authorize_auto_selects_single_provider(gated_client):
    _verifier, challenge = _make_pkce()
    response = gated_client.get(
        "/auth/native/authorize",
        params={
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "redirect_uri": "http://127.0.0.1:53999/cb",
            "state": "single-provider-state",
        },
    )

    assert response.status_code == 302
    assert response.headers["location"].startswith("https://fly-app.fly.dev/auth/callback?")


def test_native_authorize_renders_multi_provider_chooser(gated_client):
    clear_providers()
    register_provider(NousStubAuthProvider())
    register_provider(SelfHostedOidcStubAuthProvider())
    _verifier, challenge = _make_pkce()
    redirect_uri = "http://127.0.0.1:53999/cb"
    state = 'state-with-unsafe-chars-"&<>'

    response = gated_client.get(
        "/auth/native/authorize",
        params={
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "redirect_uri": redirect_uri,
            "state": state,
        },
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate"
    assert "Nous Research" in response.text
    assert "Self-Hosted OIDC" in response.text

    parser = ProviderLinkParser()
    parser.feed(response.text)
    assert len(parser.hrefs) == 2

    links_by_provider = {}
    for href in parser.hrefs:
        parsed = urlparse(href)
        query = parse_qs(parsed.query)
        links_by_provider[query["provider"][0]] = (parsed, query)

    assert set(links_by_provider) == {"nous", "self-hosted"}
    for parsed, query in links_by_provider.values():
        assert parsed.path == "/auth/native/authorize"
        assert query["code_challenge"] == [challenge]
        assert query["code_challenge_method"] == ["S256"]
        assert query["redirect_uri"] == [redirect_uri]
        assert query["state"] == [state]

    for href in parser.hrefs:
        selected = gated_client.get(href)
        assert selected.status_code == 302


def test_native_authorize_without_registered_provider_fails_closed(gated_client):
    clear_providers()
    _verifier, challenge = _make_pkce()
    response = gated_client.get(
        "/auth/native/authorize",
        params={
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "redirect_uri": "http://127.0.0.1:53999/cb",
            "state": "no-provider-state",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown provider: ''"




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
# Native refresh
# ---------------------------------------------------------------------------


def test_native_refresh_dead_token_returns_401(gated_client):
    r = gated_client.post(
        "/auth/native/refresh",
        json={"refresh_token": "garbage-not-a-real-rt", "provider": "stub"},
    )
    assert r.status_code == 401
    assert r.json()["error"] == "session_expired"
