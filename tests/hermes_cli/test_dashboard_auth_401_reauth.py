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
        rt_cookies = [c for c in cookies if SESSION_RT_COOKIE in c]
        assert rt_cookies == []
        # AT cookie still set (whichever variant the request resolves to).
        at_cookies = [c for c in cookies if SESSION_AT_COOKIE in c]
        assert len(at_cookies) == 1

    def test_present_refresh_token_still_emits_rt_cookie(self):
        client = TestClient(self._build_app(refresh_token="forward-compat"))
        r = client.get("/set")
        cookies = r.headers.get_list("set-cookie")
        rt_cookies = [c for c in cookies if SESSION_RT_COOKIE in c]
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
            SESSION_RT_COOKIE in c and "Max-Age=0" in c
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
    """End-to-end next= propagation through a full OAuth round trip.

    These tests drive the real flow exactly as the gate produces it:

      1. unauth GET /sessions  → 302 /login?next=%2Fsessions
      2. GET /login?next=%2Fsessions → HTML with provider buttons that
         carry next=%2Fsessions in their hrefs
      3. GET /auth/login?provider=stub&next=%2Fsessions → 302 to IDP +
         PKCE cookie carrying provider/state/verifier/next
      4. IDP returns to /auth/callback?code=...&state=... (NO next on
         the callback URL — real IDPs only echo back code+state)
      5. /auth/callback reads next from the PKCE cookie, validates it,
         and redirects there.

    Discrimination: each test drives the flow without smuggling
    ``next=`` onto the callback URL. Under the pre-fix code paths
    (/login ignored next=, /auth/login dropped it, /auth/callback read
    it from the wrong place), the callback always lands on ``/``. Only
    PKCE-cookie carriage produces the correct landing.
    """

    def _drive_oauth_via_login(
        self, gated_app, *, next_path: str = "",
        expect_next_in_button: bool = True,
    ):
        """Walk /login → /auth/login → IDP-bounce → /auth/callback like
        a real browser. ``next_path`` is the path the gate would have
        encoded for the user; nothing about the callback URL is
        smuggled. ``expect_next_in_button`` controls whether the
        rendered /login page is expected to thread next= into the
        provider button — False for cases where the same-origin
        validator drops the value (e.g. //evil.com, /login)."""
        login_path = "/login"
        if next_path:
            login_path = f"/login?next={quote(next_path, safe='')}"
        r_login = gated_app.get(login_path, follow_redirects=False)
        assert r_login.status_code == 200
        # Click the stub provider button. Real browsers parse the HTML;
        # we extract the href the page emitted, so a regression that
        # forgets to thread next= through the button will surface here.
        body = r_login.text
        # Each provider button is emitted as an <a class="provider-btn"
        # href="/auth/login?provider=stub..."> line.
        marker = 'href="'
        i = body.find('class="provider-btn"')
        assert i != -1, "no provider button in /login HTML"
        h = body.find(marker, i) + len(marker)
        j = body.find('"', h)
        href = body[h:j]
        # Critical: the href must carry next= when /login was given
        # next= AND the validator accepted it. (This is the property the
        # pre-fix render_login_html didn't satisfy.) For rejected
        # next= values, the validator drops them at the /login boundary
        # and the button href must NOT carry the rogue value.
        if next_path and expect_next_in_button:
            assert "next=" in href, (
                f"login button dropped next= (href={href!r})"
            )
        if next_path and not expect_next_in_button:
            assert "next=" not in href, (
                f"login button leaked rejected next= "
                f"(next_path={next_path!r}, href={href!r})"
            )

        r_to_idp = gated_app.get(href, follow_redirects=False)
        assert r_to_idp.status_code == 302
        # Stub IDP "returns" code+state on the callback URL — same shape
        # as a real IDP. Critical: we do NOT append next= here.
        state = r_to_idp.headers["location"].split("state=")[1]
        return gated_app.get(
            f"/auth/callback?code=stub_code&state={state}",
            follow_redirects=False,
        )

    def test_callback_without_next_lands_at_root(self, gated_app):
        r = self._drive_oauth_via_login(gated_app)
        assert r.status_code == 302
        assert r.headers["location"] == "/"

    def test_callback_with_safe_next_lands_there(self, gated_app):
        r = self._drive_oauth_via_login(gated_app, next_path="/sessions")
        assert r.status_code == 302
        assert r.headers["location"] == "/sessions"

    def test_callback_with_query_string_in_next(self, gated_app):
        r = self._drive_oauth_via_login(
            gated_app, next_path="/sessions?page=2"
        )
        assert r.status_code == 302
        assert r.headers["location"] == "/sessions?page=2"

    def test_callback_rejects_open_redirect(self, gated_app):
        # Attacker tries to inject ``next=//evil.com`` at the /login
        # boundary, hoping it survives to the callback redirect. The
        # /login validator drops it before it reaches the button href
        # (and therefore the cookie), so the callback never sees it and
        # the user lands at "/".
        r = self._drive_oauth_via_login(
            gated_app, next_path="//evil.com/steal",
            expect_next_in_button=False,
        )
        assert r.status_code == 302
        assert r.headers["location"] == "/"

    def test_callback_rejects_login_loop(self, gated_app):
        r = self._drive_oauth_via_login(
            gated_app, next_path="/login",
            expect_next_in_button=False,
        )
        assert r.status_code == 302
        assert r.headers["location"] == "/"

    def test_attacker_callback_next_param_is_ignored(self, gated_app):
        """Hardening: even if an attacker crafts a callback URL with a
        rogue ``next=`` query parameter, the server reads from the PKCE
        cookie (server-set) and ignores the URL value. This pins the
        fix against a regression that re-introduces the URL read."""
        # Drive a clean login with no next=.
        r_login = gated_app.get("/login", follow_redirects=False)
        assert r_login.status_code == 200
        r_to_idp = gated_app.get(
            "/auth/login?provider=stub", follow_redirects=False
        )
        state = r_to_idp.headers["location"].split("state=")[1]
        # Attacker appends next=/internal-admin to the callback URL.
        r = gated_app.get(
            f"/auth/callback?code=stub_code&state={state}"
            f"&next={quote('/internal-admin', safe='')}",
            follow_redirects=False,
        )
        assert r.status_code == 302
        # No next= was in the PKCE cookie, so landing must be "/" —
        # NOT /internal-admin.
        assert r.headers["location"] == "/"


# ---------------------------------------------------------------------------
# Unit-level coverage: render_login_html threads next= into provider buttons
# ---------------------------------------------------------------------------


class TestRenderLoginHtmlNext:
    """Cover ``render_login_html`` directly so a regression that drops
    the ``next_path`` parameter is caught at the function boundary, not
    only via the full integration walk."""

    def setup_method(self):
        clear_providers()
        register_provider(StubAuthProvider())

    def teardown_method(self):
        clear_providers()

    def test_no_next_emits_plain_button(self):
        from hermes_cli.dashboard_auth.login_page import render_login_html
        html_out = render_login_html()
        assert 'href="/auth/login?provider=stub"' in html_out
        assert "next=" not in html_out

    def test_next_threaded_url_encoded(self):
        from hermes_cli.dashboard_auth.login_page import render_login_html
        html_out = render_login_html(next_path="/sessions?page=2")
        # next= is URL-encoded — quote(safe='') turns "/" into "%2F",
        # "?" into "%3F", "=" into "%3D". The encoded value never
        # contains an "&" so the raw "&" separator in the href is
        # unambiguous.
        assert "next=%2Fsessions%3Fpage%3D2" in html_out
        assert "provider=stub&next=" in html_out

    def test_next_with_html_metacharacters_is_escaped(self):
        """Defence in depth: even though the caller validates next_path,
        we still HTML-escape the rendered value so a regression in the
        caller can't trivially produce an HTML-injection sink."""
        from hermes_cli.dashboard_auth.login_page import render_login_html
        # `"` in a path is already URL-encoded by quote() to %22, so it
        # never reaches the HTML escaper as a raw quote. This test pins
        # both layers: quote() does its job AND escape() does its.
        html_out = render_login_html(next_path='/x"injected')
        assert '"injected' not in html_out
        assert "%22injected" in html_out


# ---------------------------------------------------------------------------
# Unit-level coverage: /auth/login persists next= into the PKCE cookie
# ---------------------------------------------------------------------------


class TestAuthLoginPkceCookieNext:
    """Cover the ``/auth/login`` route's PKCE cookie payload directly.

    The cookie is the round-trip carrier for ``next=``; if /auth/login
    forgets to encode it, the callback has no path to honour even when
    everything else is wired correctly.
    """

    def test_no_next_query_omits_next_segment(self, gated_app):
        r = gated_app.get(
            "/auth/login?provider=stub", follow_redirects=False
        )
        assert r.status_code == 302
        cookies = r.headers.get_list("set-cookie")
        pkce = next(c for c in cookies if "hermes_session_pkce" in c)
        assert "next=" not in pkce

    def test_safe_next_query_encoded_into_cookie(self, gated_app):
        r = gated_app.get(
            f"/auth/login?provider=stub&next={quote('/sessions', safe='')}",
            follow_redirects=False,
        )
        cookies = r.headers.get_list("set-cookie")
        pkce = next(c for c in cookies if "hermes_session_pkce" in c)
        # ``next=`` segment present, URL-encoded.
        assert "next=%2Fsessions" in pkce

    def test_unsafe_next_query_dropped_from_cookie(self, gated_app):
        """The validator at /auth/login refuses //evil.com BEFORE
        storing it. Defence in depth: even if a regression leaks next=
        through /login's button rendering, /auth/login is the second
        boundary."""
        r = gated_app.get(
            f"/auth/login?provider=stub&next={quote('//evil.com/x', safe='')}",
            follow_redirects=False,
        )
        cookies = r.headers.get_list("set-cookie")
        pkce = next(c for c in cookies if "hermes_session_pkce" in c)
        assert "next=" not in pkce
