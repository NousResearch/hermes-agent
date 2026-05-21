"""Phase 6 — 401 re-auth + ``next=`` propagation tests.

Verifies the contract documented in Phase 6 v2 of the plan:

  - API 401 responses carry ``{"error", "login_url", ...}`` so the SPA
    fetch wrapper can ``window.location.assign(body.login_url)``.
  - The ``login_url`` embeds a ``next=<original-path>`` query string so
    re-auth lands the user back where they were.
  - HTML redirects ALSO carry ``next=``.
  - ``next=`` validation: protocol-relative paths, absolute URLs, and
    loops back to ``/login`` / ``/auth/*`` are dropped.
  - Invalid/expired cookies are cleared on 401 so the browser doesn't
    keep replaying them.
  - ``set_session_cookies(refresh_token="")`` does NOT emit the
    ``hermes_session_rt`` cookie (contract V1: no RT to persist).
  - ``/auth/callback?next=…`` honours the same-origin landing path.
"""

from __future__ import annotations

from urllib.parse import quote

import pytest

# Phase 5 / Phase 6: these tests mutate ``web_server.app.state.auth_required``
# at module level. Run them in the same xdist worker so they don't race
# against each other (and against any other file that also touches
# ``app.state``) — the marker name is shared across all dashboard-auth test
# files that gate the app.
pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.cookies import (
    SESSION_AT_COOKIE,
    SESSION_RT_COOKIE,
    clear_session_cookies,
    set_session_cookies,
)
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gated_app():
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


# ---------------------------------------------------------------------------
# set_session_cookies(refresh_token="") skips the RT cookie
# ---------------------------------------------------------------------------


class TestRefreshTokenCookieDeprecation:
    def _build_app(self, *, refresh_token: str):
        app = FastAPI()

        @app.get("/set")
        def _set():
            r = Response("ok")
            set_session_cookies(
                r, access_token="AT", refresh_token=refresh_token,
                access_token_expires_in=3600, use_https=True,
            )
            return r

        return app

    def test_empty_refresh_token_does_not_emit_rt_cookie(self):
        client = TestClient(self._build_app(refresh_token=""))
        r = client.get("/set")
        cookies = r.headers.get_list("set-cookie")
        rt_cookies = [c for c in cookies if c.startswith(f"{SESSION_RT_COOKIE}=")]
        assert rt_cookies == []
        # AT cookie still set.
        at_cookies = [c for c in cookies if c.startswith(f"{SESSION_AT_COOKIE}=")]
        assert len(at_cookies) == 1

    def test_present_refresh_token_still_emits_rt_cookie(self):
        client = TestClient(self._build_app(refresh_token="forward-compat"))
        r = client.get("/set")
        cookies = r.headers.get_list("set-cookie")
        rt_cookies = [c for c in cookies if c.startswith(f"{SESSION_RT_COOKIE}=")]
        assert len(rt_cookies) == 1
        assert "forward-compat" in rt_cookies[0]

    def test_clear_session_cookies_still_emits_rt_deletion(self):
        """Even when we never wrote the RT cookie, logout/clear should
        emit a Max-Age=0 deletion to flush stale cookies from old
        deployments."""
        app = FastAPI()

        @app.get("/clear")
        def _clear():
            r = Response("ok")
            clear_session_cookies(r)
            return r

        client = TestClient(app)
        r = client.get("/clear")
        cookies = r.headers.get_list("set-cookie")
        assert any(
            c.startswith(f"{SESSION_RT_COOKIE}=") and "Max-Age=0" in c
            for c in cookies
        )


# ---------------------------------------------------------------------------
# Gate middleware: 401 envelope + next= propagation
# ---------------------------------------------------------------------------


class TestApi401Envelope:
    def test_no_cookie_returns_unauthenticated_envelope(self, gated_app):
        r = gated_app.get("/api/status")
        assert r.status_code == 401
        body = r.json()
        assert body["error"] == "unauthenticated"
        assert "login_url" in body
        assert body["login_url"].startswith("/login")

    def test_invalid_cookie_returns_session_expired_envelope(self, gated_app):
        gated_app.cookies.set(SESSION_AT_COOKIE, "garbage")
        r = gated_app.get("/api/status")
        assert r.status_code == 401
        body = r.json()
        assert body["error"] == "session_expired"
        assert body["login_url"].startswith("/login")

    def test_invalid_cookie_clears_dead_cookie(self, gated_app):
        """Dead-cookie cleanup — Phase 6 requirement so the browser
        doesn't keep replaying the stale token on every request."""
        gated_app.cookies.set(SESSION_AT_COOKIE, "garbage")
        r = gated_app.get("/api/status")
        set_cookies = r.headers.get_list("set-cookie")
        assert any(
            c.startswith(f"{SESSION_AT_COOKIE}=") and "Max-Age=0" in c
            for c in set_cookies
        )

    def test_login_url_carries_next_for_deep_api_path(self, gated_app):
        r = gated_app.get("/api/sessions?page=2")
        body = r.json()
        # next= is URL-encoded.
        assert "next=" in body["login_url"]
        assert quote("/api/sessions?page=2", safe="") in body["login_url"]


class TestHtmlRedirectNext:
    def test_deep_html_path_redirects_with_next(self, gated_app):
        r = gated_app.get("/sessions", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"] == "/login?next=%2Fsessions"

    def test_root_path_redirects_with_next(self, gated_app):
        r = gated_app.get("/", follow_redirects=False)
        assert r.headers["location"] in ("/login", "/login?next=%2F")

    def test_login_loop_avoided(self, gated_app):
        """A request to /login itself must not produce ``?next=/login``
        because that'd be a loop after re-auth."""
        # /login is on the public allowlist so it doesn't go through the
        # 401 path. But sanity: the page renders.
        r = gated_app.get("/login")
        assert r.status_code == 200

    def test_auth_loop_avoided(self, gated_app):
        """A failed cookie on /auth/me (auth-required path) must drop
        the next= rather than risk a /login?next=/api/auth/me loop."""
        # /api/auth/me requires auth. Without cookie → 401 with login_url
        # but next= must NOT point at /api/auth/.
        r = gated_app.get("/api/auth/me")
        assert r.status_code == 401
        body = r.json()
        assert "next=" not in body["login_url"]


# ---------------------------------------------------------------------------
# Gate middleware: same-origin next= validation
# ---------------------------------------------------------------------------


class TestNextSameOriginValidation:
    def test_protocol_relative_path_dropped(self, gated_app):
        # `//evil.com/foo` parses to a protocol-relative URL — browser
        # would treat as cross-origin. We drop it at the gate; the path
        # we redirect to should NOT contain `//evil.com`.
        r = gated_app.get("//evil.com", follow_redirects=False)
        # Starlette likely normalizes the path before we see it, so the
        # gate may see "/evil.com" — either way the encoded value
        # in next= must be safe to feed to window.location.assign.
        # Just assert no protocol-relative form survives.
        assert r.status_code == 302
        location = r.headers["location"]
        assert "%2F%2Fevil" not in location  # urlencoded // form
        assert "//evil" not in location

    def test_safe_next_validator_accepts_same_origin(self):
        from hermes_cli.dashboard_auth.middleware import _safe_next_target

        class FakeRequest:
            def __init__(self, path, query=""):
                self.url = type("URL", (), {"path": path, "query": query})()

        assert _safe_next_target(FakeRequest("/sessions")) == "%2Fsessions"
        assert (
            _safe_next_target(FakeRequest("/sessions", "page=2"))
            == "%2Fsessions%3Fpage%3D2"
        )

    def test_safe_next_validator_rejects_protocol_relative(self):
        from hermes_cli.dashboard_auth.middleware import _safe_next_target

        class FakeRequest:
            def __init__(self, path):
                self.url = type("URL", (), {"path": path, "query": ""})()

        assert _safe_next_target(FakeRequest("//evil.com")) == ""

    def test_safe_next_validator_rejects_login_loop(self):
        from hermes_cli.dashboard_auth.middleware import _safe_next_target

        class FakeRequest:
            def __init__(self, path):
                self.url = type("URL", (), {"path": path, "query": ""})()

        assert _safe_next_target(FakeRequest("/login")) == ""
        assert _safe_next_target(FakeRequest("/auth/login")) == ""
        assert _safe_next_target(FakeRequest("/api/auth/me")) == ""


# ---------------------------------------------------------------------------
# /auth/callback honours next= and validates it
# ---------------------------------------------------------------------------


class TestAuthCallbackNext:
    def _drive_oauth(self, gated_app, *, next_path: str = ""):
        next_qs = f"&next={quote(next_path, safe='')}" if next_path else ""
        r1 = gated_app.get(
            f"/auth/login?provider=stub{next_qs}", follow_redirects=False
        )
        state = r1.headers["location"].split("state=")[1]
        # next is preserved by the route (it's in the original URL — but the
        # stub IDP returns to /auth/callback. We need to pass next as a
        # separate query param on the callback URL to simulate what a real
        # IDP would do via state-bound storage. For this test, the
        # /auth/callback handler reads `next` directly from its own query
        # string, so just append it.
        return gated_app.get(
            f"/auth/callback?code=stub_code&state={state}{next_qs}",
            follow_redirects=False,
        )

    def test_callback_without_next_lands_at_root(self, gated_app):
        r = self._drive_oauth(gated_app)
        assert r.status_code == 302
        assert r.headers["location"] == "/"

    def test_callback_with_safe_next_lands_there(self, gated_app):
        r = self._drive_oauth(gated_app, next_path="/sessions")
        assert r.status_code == 302
        assert r.headers["location"] == "/sessions"

    def test_callback_with_query_string_in_next(self, gated_app):
        r = self._drive_oauth(gated_app, next_path="/sessions?page=2")
        assert r.status_code == 302
        assert r.headers["location"] == "/sessions?page=2"

    def test_callback_rejects_open_redirect(self, gated_app):
        # Attacker provides ``next=//evil.com`` hoping for an open redirect
        # after successful auth. Validator drops it; user lands at "/".
        r = self._drive_oauth(gated_app, next_path="//evil.com/steal")
        assert r.status_code == 302
        assert r.headers["location"] == "/"

    def test_callback_rejects_login_loop(self, gated_app):
        r = self._drive_oauth(gated_app, next_path="/login")
        assert r.status_code == 302
        assert r.headers["location"] == "/"
