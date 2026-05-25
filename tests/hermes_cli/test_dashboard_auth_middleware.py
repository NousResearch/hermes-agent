"""End-to-end behavioural tests for the dashboard auth gate.

Uses ``StubAuthProvider`` so the OAuth round trip can complete in-process
without any external IDP.  Exercises:

  * `/api/status` flips from public (loopback) to gated (auth_required)
  * `/` redirects to /login when no cookie present
  * `/api/auth/providers` is the public bootstrap endpoint
  * `/login` renders HTML listing all providers
  * /assets/* still passes through unauthenticated
  * Full /auth/login → /auth/callback → / round trip with the stub
  * Invalid / missing cookies return 401 (api) or 302 (html)
  * Zero-providers + gate-on fails closed
"""
from __future__ import annotations

import pytest

# Phase 5 / Phase 6: these tests mutate ``web_server.app.state.auth_required``
# at module level. Run them in the same xdist worker so they don't race
# against each other (and against any other file that also touches
# ``app.state``) — the marker name is shared across all dashboard-auth test
# files that gate the app.
pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE
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
    # Use https base_url so cookies pick up Secure flag and host_header
    # matches the bound interface.
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


# ---------------------------------------------------------------------------
# Allowlist (public) routes
# ---------------------------------------------------------------------------


def test_gated_status_now_requires_auth(gated_app):
    """When gate is on, /api/status is NOT public — login bootstrap uses /api/auth/providers."""
    r = gated_app.get("/api/status")
    assert r.status_code == 401


def test_gated_html_redirects_to_login(gated_app):
    r = gated_app.get("/", follow_redirects=False)
    assert r.status_code == 302
    # Phase 6: gate carries a ``next=`` so post-login bounces back to /.
    assert r.headers["location"] in ("/login", "/login?next=%2F")


def test_gated_auth_providers_is_public(gated_app):
    r = gated_app.get("/api/auth/providers")
    assert r.status_code == 200
    body = r.json()
    assert any(p["name"] == "stub" for p in body["providers"])
    assert body["providers"][0]["display_name"] == "Stub IdP (test only)"


def test_gated_login_html_is_public_and_lists_providers(gated_app):
    r = gated_app.get("/login")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Stub IdP" in r.text
    assert 'href="/auth/login?provider=stub"' in r.text


def test_gated_static_asset_path_is_public(gated_app):
    """``/assets/*`` is allowlisted so the SPA's CSS/JS loads pre-login."""
    r = gated_app.get("/assets/_nonexistent.css")
    # 404 not 401 — proves middleware let the request through to the
    # static-files mount, which then 404'd because the file isn't there.
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# OAuth round trip
# ---------------------------------------------------------------------------


def test_full_login_round_trip_unlocks_api_status(gated_app):
    # 1) Click "Sign in with Stub IdP" — /auth/login redirects to the stub
    #    with a PKCE cookie on the response.
    r1 = gated_app.get("/auth/login?provider=stub", follow_redirects=False)
    assert r1.status_code == 302
    pkce = next(
        (c for c in r1.headers.get_list("set-cookie")
         if "hermes_session_pkce" in c),
        None,
    )
    assert pkce and "HttpOnly" in pkce

    redirect = r1.headers["location"]
    # Stub bounces back to {redirect_uri}?code=stub_code&state=<s>
    assert "code=stub_code" in redirect
    assert "state=" in redirect
    state = redirect.split("state=")[1]

    # 2) The browser would now follow the redirect to /auth/callback.
    #    TestClient automatically carries the PKCE cookie forward.
    r2 = gated_app.get(
        f"/auth/callback?code=stub_code&state={state}",
        follow_redirects=False,
    )
    assert r2.status_code == 302
    assert r2.headers["location"] == "/"
    set_cookies = r2.headers.get_list("set-cookie")
    assert any("hermes_session_at" in c for c in set_cookies)
    assert any("hermes_session_rt" in c for c in set_cookies)

    # 3) /api/status now succeeds because we're authenticated.
    r3 = gated_app.get("/api/status")
    assert r3.status_code == 200
    body = r3.json()
    assert "version" in body


def test_login_unknown_provider_returns_404(gated_app):
    r = gated_app.get("/auth/login?provider=nonexistent", follow_redirects=False)
    assert r.status_code == 404


def test_callback_without_pkce_cookie_returns_400(gated_app):
    # No prior /auth/login → no PKCE cookie.
    r = gated_app.get(
        "/auth/callback?code=stub_code&state=anything",
        follow_redirects=False,
    )
    assert r.status_code == 400


def test_callback_state_mismatch_returns_400(gated_app):
    # Walk through /auth/login first to plant the PKCE cookie.
    r1 = gated_app.get("/auth/login?provider=stub", follow_redirects=False)
    # ...then pretend the IDP returned a different state.
    r2 = gated_app.get(
        "/auth/callback?code=stub_code&state=WRONG",
        follow_redirects=False,
    )
    assert r2.status_code == 400


def test_callback_invalid_code_returns_400(gated_app):
    r1 = gated_app.get("/auth/login?provider=stub", follow_redirects=False)
    state = r1.headers["location"].split("state=")[1]
    r2 = gated_app.get(
        f"/auth/callback?code=BAD_CODE&state={state}",
        follow_redirects=False,
    )
    assert r2.status_code == 400


# ---------------------------------------------------------------------------
# Cookie validation
# ---------------------------------------------------------------------------


def test_invalid_cookie_returns_401_on_api(gated_app):
    gated_app.cookies.set(SESSION_AT_COOKIE, "garbage-not-a-real-token")
    r = gated_app.get("/api/sessions")
    assert r.status_code == 401


def test_invalid_cookie_redirects_on_html(gated_app):
    gated_app.cookies.set(SESSION_AT_COOKIE, "garbage")
    r = gated_app.get("/", follow_redirects=False)
    assert r.status_code == 302
    # Phase 6: gate carries a ``next=`` so post-login bounces back to /.
    assert r.headers["location"] in ("/login", "/login?next=%2F")


def test_logout_clears_cookies_and_redirects_to_login(gated_app):
    # First log in.
    r1 = gated_app.get("/auth/login?provider=stub", follow_redirects=False)
    state = r1.headers["location"].split("state=")[1]
    gated_app.get(
        f"/auth/callback?code=stub_code&state={state}",
        follow_redirects=False,
    )
    # Now log out.
    r = gated_app.post("/auth/logout", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/login"
    set_cookies = r.headers.get_list("set-cookie")
    assert any(
        c.startswith("hermes_session_at=") and "Max-Age=0" in c
        for c in set_cookies
    )
    assert any(
        c.startswith("hermes_session_rt=") and "Max-Age=0" in c
        for c in set_cookies
    )


# ---------------------------------------------------------------------------
# Identity probe
# ---------------------------------------------------------------------------


def test_api_auth_me_returns_session_after_login(gated_app):
    r1 = gated_app.get("/auth/login?provider=stub", follow_redirects=False)
    state = r1.headers["location"].split("state=")[1]
    gated_app.get(
        f"/auth/callback?code=stub_code&state={state}",
        follow_redirects=False,
    )
    r = gated_app.get("/api/auth/me")
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == "stub-user-1"
    assert body["email"] == "stub@example.test"
    assert body["display_name"] == "Stub User"
    assert body["provider"] == "stub"
    assert body["org_id"] == "stub-org-1"
    assert "expires_at" in body


def test_api_auth_me_requires_auth(gated_app):
    # No cookies.
    r = gated_app.get("/api/auth/me")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Zero-providers fail-closed
# ---------------------------------------------------------------------------


def test_gated_zero_providers_fails_closed_on_api_auth_providers():
    """If gate is on but no providers are registered, /api/auth/providers 503s."""
    clear_providers()
    prev_required = getattr(web_server.app.state, "auth_required", None)
    prev_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.auth_required = True
    try:
        client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
        r = client.get("/api/auth/providers")
        assert r.status_code == 503
        assert "no auth providers" in r.text.lower()
    finally:
        web_server.app.state.auth_required = prev_required
        web_server.app.state.bound_host = prev_host


def test_gated_zero_providers_login_page_renders_help_text():
    clear_providers()
    prev_required = getattr(web_server.app.state, "auth_required", None)
    prev_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.auth_required = True
    try:
        client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
        r = client.get("/login")
        assert r.status_code == 200
        # Empty-provider HTML mentions the fix-up path.  (HTML wraps text
        # so we can't grep for the exact phrase; check for the canonical
        # fragments instead.)
        text = r.text.lower()
        assert "sign-in unavailable" in text
        assert "no authentication" in text
        assert "providers are installed" in text
        assert "--insecure" in text
    finally:
        web_server.app.state.auth_required = prev_required
        web_server.app.state.bound_host = prev_host
