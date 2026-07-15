"""Tests for the password (non-redirect) dashboard-auth login flow.

Covers the protocol extension (``supports_password`` +
``complete_password_login``), the ``/auth/password-login`` route end-to-end
through the REAL ``gated_auth_middleware`` (session-cookie mint →
authenticated request → transparent refresh), the login-page credential
form rendering, and the route's rate limiter.

The E2E harness mirrors ``test_dashboard_auth_401_reauth.py``: register a
provider, flip ``app.state.auth_required = True``, drive a ``TestClient``.
"""

from __future__ import annotations

import time

import pytest

from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    InvalidCredentialsError,
    ProviderError,
    Session,
    assert_protocol_compliance,
    clear_providers,
    register_provider,
)
from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE, SESSION_RT_COOKIE
from hermes_cli.dashboard_auth.login_page import render_login_html
from hermes_cli.dashboard_auth.routes import _reset_password_rate_limit
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


# ---------------------------------------------------------------------------
# Test password provider — minimal, in-memory, signed tokens.
# ---------------------------------------------------------------------------


def _sign(secret: bytes, sub: str, kind: str, ttl: int) -> str:
    import base64
    import hashlib
    import hmac
    import json

    raw = json.dumps(
        {"sub": sub, "kind": kind, "exp": int(time.time()) + ttl},
        separators=(",", ":"),
    ).encode()
    sig = hmac.new(secret, raw, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(raw + sig).decode()


def _unsign(secret: bytes, token: str):
    import base64
    import hashlib
    import hmac
    import json

    try:
        blob = base64.urlsafe_b64decode(token.encode())
        raw, sig = blob[:-32], blob[-32:]
        if not hmac.compare_digest(
            sig, hmac.new(secret, raw, hashlib.sha256).digest()
        ):
            return None
        return json.loads(raw)
    except Exception:
        return None


class PasswordProvider(DashboardAuthProvider):
    """In-test username/password provider (admin / hunter2)."""

    name = "testpw"
    display_name = "Test Password"
    supports_password = True

    def __init__(self, *, ttl: int = 3600, secret: bytes = b"test-secret-1234567890"):
        self._ttl = ttl
        self._secret = secret
        self.unreachable = False  # flip to simulate a ProviderError

    def start_login(self, *, redirect_uri: str):
        raise NotImplementedError

    def complete_login(self, **kwargs):
        raise NotImplementedError

    def complete_password_login(self, *, username: str, password: str) -> Session:
        if self.unreachable:
            raise ProviderError("backing store down")
        if username != "admin" or password != "hunter2":
            raise InvalidCredentialsError("bad creds")
        exp = int(time.time()) + self._ttl
        return Session(
            user_id="admin",
            email="",
            display_name="admin",
            org_id="",
            provider=self.name,
            expires_at=exp,
            access_token=_sign(self._secret, "admin", "access", self._ttl),
            refresh_token=_sign(self._secret, "admin", "refresh", 30 * 86400),
        )

    def verify_session(self, *, access_token: str):
        p = _unsign(self._secret, access_token)
        if not p or p.get("kind") != "access" or p["exp"] <= int(time.time()):
            return None
        return Session(
            user_id=p["sub"], email="", display_name=p["sub"], org_id="",
            provider=self.name, expires_at=p["exp"],
            access_token=access_token, refresh_token="",
        )

    def refresh_session(self, *, refresh_token: str) -> Session:
        from hermes_cli.dashboard_auth import RefreshExpiredError

        p = _unsign(self._secret, refresh_token)
        if not p or p.get("kind") != "refresh" or p["exp"] <= int(time.time()):
            raise RefreshExpiredError("dead rt")
        exp = int(time.time()) + self._ttl
        return Session(
            user_id=p["sub"], email="", display_name=p["sub"], org_id="",
            provider=self.name, expires_at=exp,
            access_token=_sign(self._secret, p["sub"], "access", self._ttl),
            refresh_token=_sign(self._secret, p["sub"], "refresh", 30 * 86400),
        )

    def revoke_session(self, *, refresh_token: str) -> None:
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pw_provider():
    return PasswordProvider()


@pytest.fixture
def gated_app(pw_provider):
    clear_providers()
    register_provider(pw_provider)
    _reset_password_rate_limit()
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    _reset_password_rate_limit()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


# ---------------------------------------------------------------------------
# Protocol extension
# ---------------------------------------------------------------------------


class TestProtocolExtension:
    def test_password_provider_is_protocol_compliant(self):
        assert assert_protocol_compliance(PasswordProvider) is None

    def test_default_supports_password_is_false(self):
        # OAuth providers (the Stub) inherit the False default.
        assert StubAuthProvider.supports_password is False

    def test_default_complete_password_login_raises_not_implemented(self):
        # A provider that doesn't override the method (the Stub) raises,
        # rather than silently accepting any credentials.
        with pytest.raises(NotImplementedError):
            StubAuthProvider().complete_password_login(
                username="x", password="y"
            )


# ---------------------------------------------------------------------------
# /api/auth/providers exposes the supports_password flag
# ---------------------------------------------------------------------------


class TestProviderListFlag:
    def test_providers_endpoint_reports_supports_password(self, gated_app):
        resp = gated_app.get("/api/auth/providers")
        assert resp.status_code == 200
        prov = {p["name"]: p for p in resp.json()["providers"]}
        assert prov["testpw"]["supports_password"] is True

    def test_password_provider_html_redirects_to_login_form(self, gated_app):
        resp = gated_app.get("/", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/login?next=%2F"

        login = gated_app.get(resp.headers["location"])
        assert login.status_code == 200
        assert '<form class="provider-form" data-provider="testpw"' in login.text
        assert "/auth/password-login" in login.text

    def test_password_provider_auth_login_redirects_to_login_form(self, gated_app):
        resp = gated_app.get(
            "/auth/login?provider=testpw&next=%2F",
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert resp.headers["location"] == "/login?next=%2F"

    def test_oauth_provider_reports_false(self):
        clear_providers()
        register_provider(StubAuthProvider())
        prev = getattr(web_server.app.state, "auth_required", None)
        web_server.app.state.auth_required = True
        try:
            client = TestClient(
                web_server.app, base_url="https://fly-app.fly.dev"
            )
            resp = client.get("/api/auth/providers")
            prov = {p["name"]: p for p in resp.json()["providers"]}
            assert prov["stub"]["supports_password"] is False
        finally:
            clear_providers()
            web_server.app.state.auth_required = prev


# ---------------------------------------------------------------------------
# /auth/password-login — end-to-end through the real middleware
# ---------------------------------------------------------------------------


class TestPasswordLoginRoute:
    def test_valid_credentials_set_session_cookies_and_return_next(
        self, gated_app
    ):
        resp = gated_app.post(
            "/auth/password-login",
            json={
                "provider": "testpw",
                "username": "admin",
                "password": "hunter2",
                "next": "/sessions",
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "next": "/sessions"}
        set_cookie = resp.headers.get("set-cookie", "")
        # HTTPS request → __Host- prefixed access-token cookie is set.
        assert SESSION_AT_COOKIE in set_cookie
        assert SESSION_RT_COOKIE in set_cookie

    def test_session_cookie_then_grants_authenticated_access(self, gated_app):
        # Log in, then hit an auth-required endpoint with the cookie jar
        # the TestClient retains — proving the minted session is accepted
        # by the real gated_auth_middleware.
        login = gated_app.post(
            "/auth/password-login",
            json={"provider": "testpw", "username": "admin", "password": "hunter2"},
        )
        assert login.status_code == 200
        me = gated_app.get("/api/auth/me")
        assert me.status_code == 200
        assert me.json()["user_id"] == "admin"
        assert me.json()["provider"] == "testpw"

    def test_wrong_password_returns_generic_401(self, gated_app):
        resp = gated_app.post(
            "/auth/password-login",
            json={"provider": "testpw", "username": "admin", "password": "WRONG"},
        )
        assert resp.status_code == 401
        # Generic detail — no user-vs-password distinction.
        assert resp.json()["detail"] == "Invalid credentials"
        assert "set-cookie" not in {k.lower() for k in resp.headers}

    def test_unknown_user_returns_same_generic_401(self, gated_app):
        resp = gated_app.post(
            "/auth/password-login",
            json={"provider": "testpw", "username": "ghost", "password": "hunter2"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid credentials"

    def test_unknown_provider_returns_404(self, gated_app):
        resp = gated_app.post(
            "/auth/password-login",
            json={"provider": "nope", "username": "admin", "password": "hunter2"},
        )
        assert resp.status_code == 404

    def test_oauth_provider_rejects_password_login_with_404(self):
        # An OAuth-only provider (supports_password False) must not be
        # reachable via the password route — same 404 as unknown, so the
        # endpoint isn't a provider-capability oracle.
        clear_providers()
        register_provider(StubAuthProvider())
        _reset_password_rate_limit()
        prev = getattr(web_server.app.state, "auth_required", None)
        web_server.app.state.auth_required = True
        try:
            client = TestClient(
                web_server.app, base_url="https://fly-app.fly.dev"
            )
            resp = client.post(
                "/auth/password-login",
                json={"provider": "stub", "username": "x", "password": "y"},
            )
            assert resp.status_code == 404
        finally:
            clear_providers()
            _reset_password_rate_limit()
            web_server.app.state.auth_required = prev

    def test_provider_unreachable_returns_503(self, gated_app, pw_provider):
        pw_provider.unreachable = True
        resp = gated_app.post(
            "/auth/password-login",
            json={"provider": "testpw", "username": "admin", "password": "hunter2"},
        )
        assert resp.status_code == 503

    def test_open_redirect_next_is_dropped(self, gated_app):
        resp = gated_app.post(
            "/auth/password-login",
            json={
                "provider": "testpw",
                "username": "admin",
                "password": "hunter2",
                "next": "https://evil.example/phish",
            },
        )
        assert resp.status_code == 200
        # Malicious absolute URL dropped → lands at root.
        assert resp.json()["next"] == "/"

    def test_route_is_public_unauthenticated(self, gated_app):
        # The login route itself must be reachable without a session —
        # otherwise you could never log in.
        resp = gated_app.post(
            "/auth/password-login",
            json={"provider": "testpw", "username": "admin", "password": "hunter2"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Transparent refresh — expired access token, live refresh token
# ---------------------------------------------------------------------------


class TestPasswordSessionRefresh:
    def test_expired_access_token_refreshes_via_rt_cookie(self):
        # TTL=0 → access token born expired; the RT cookie should drive a
        # transparent refresh on the next request (the same machinery the
        # OAuth provider uses).
        clear_providers()
        provider = PasswordProvider(ttl=0)
        register_provider(provider)
        _reset_password_rate_limit()
        prev = getattr(web_server.app.state, "auth_required", None)
        web_server.app.state.auth_required = True
        try:
            client = TestClient(
                web_server.app, base_url="https://fly-app.fly.dev"
            )
            login = client.post(
                "/auth/password-login",
                json={"provider": "testpw", "username": "admin", "password": "hunter2"},
            )
            assert login.status_code == 200
            # Give the provider a live TTL so the refreshed token verifies.
            provider._ttl = 3600
            me = client.get("/api/auth/me")
            assert me.status_code == 200
            assert me.json()["user_id"] == "admin"
        finally:
            clear_providers()
            _reset_password_rate_limit()
            web_server.app.state.auth_required = prev


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_repeated_failures_eventually_429(self, gated_app):
        # The limiter caps attempts per IP per window (default 10). After
        # the budget is exhausted, even a VALID credential gets 429.
        last = None
        for _ in range(15):
            last = gated_app.post(
                "/auth/password-login",
                json={"provider": "testpw", "username": "admin", "password": "WRONG"},
            )
        assert last.status_code == 429
        # Even correct creds are throttled once the window is saturated.
        good = gated_app.post(
            "/auth/password-login",
            json={"provider": "testpw", "username": "admin", "password": "hunter2"},
        )
        assert good.status_code == 429


# ---------------------------------------------------------------------------
# Login page rendering
# ---------------------------------------------------------------------------


class TestLoginPageRender:
    def test_password_provider_renders_credential_form_and_script(self):
        clear_providers()
        register_provider(PasswordProvider())
        try:
            html = render_login_html(next_path="/sessions")
            assert '<form class="provider-form" data-provider="testpw"' in html
            assert 'name="username"' in html
            assert 'name="password"' in html
            assert 'value="/sessions"' in html
            assert "<script>" in html
            assert "/auth/password-login" in html
        finally:
            clear_providers()

    def test_oauth_only_page_stays_script_free(self):
        clear_providers()
        register_provider(StubAuthProvider())
        try:
            html = render_login_html()
            assert "provider-btn" in html
            assert "<script>" not in html
            # No password FORM element rendered (the .provider-form CSS
            # rule lives in the template's <style> block unconditionally;
            # what must be absent is an actual rendered form + its script).
            assert '<form class="provider-form"' not in html
            assert "/auth/password-login" not in html
        finally:
            clear_providers()

    def test_mixed_providers_render_both(self):
        clear_providers()
        register_provider(StubAuthProvider())
        register_provider(PasswordProvider())
        try:
            html = render_login_html()
            # OAuth redirect button AND a password form, both present.
            assert "/auth/login?provider=stub" in html
            assert 'data-provider="testpw"' in html
            assert "<script>" in html
        finally:
            clear_providers()
