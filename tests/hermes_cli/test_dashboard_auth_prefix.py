"""Path-prefix (X-Forwarded-Prefix) awareness for the dashboard-auth gate.

Mission-control style deployments reverse-proxy the dashboard at a path
prefix (e.g. ``mission-control.tilos.com/hermes/*`` -> local Caddy ->
:9119), injecting ``X-Forwarded-Prefix: /hermes`` on every request.

The dashboard already honours this for the SPA bundle (rewriting asset
URLs and the bootstrap ``__HERMES_BASE_PATH__``). The OAuth gate must
honour it too:

  1. The gate's ``Location:`` redirect to /login (in
     ``_unauth_response``) needs to be ``/hermes/login`` so the browser
     follows it through the proxy.
  2. The 401 JSON envelope's ``login_url`` needs the same prefix so the
     SPA's full-page navigation lands at the proxied login page.
  3. ``_redirect_uri`` (the OAuth callback URL handed to the IDP) must
     reconstruct the public URL including the prefix, otherwise the IDP
     redirects back to ``/auth/callback`` instead of
     ``/hermes/auth/callback`` and the user gets 404.
  4. Cookies must use ``Path=/hermes`` when behind a prefix so they
     don't leak to other apps on the same origin AND so they get sent
     back to the dashboard on subsequent requests under the prefix.
  5. The ``__Host-`` cookie prefix requires ``Path=/`` — when behind an
     X-Forwarded-Prefix we use ``__Secure-`` instead (matches every
     hardening property except scope, which the explicit ``Path``
     covers).

These tests document the wire-level contract so a regression in any of
those rules surfaces before a Mission Control deploy.
"""
from __future__ import annotations

import pytest

# Same xdist group as the other dashboard-auth tests — they all mutate
# web_server.app.state.auth_required at module level.
pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")

from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture
def gated_app_proxied():
    """web_server.app configured for gated mode with proxy_headers + a
    public Host that simulates the Mission Control reverse proxy.

    The ``base_url`` sets ``host:scheme`` defaults so we don't have to
    pass them on every request. ``X-Forwarded-Prefix`` is passed
    per-request because the TestClient doesn't have a way to default
    request headers.
    """
    clear_providers()
    register_provider(StubAuthProvider())
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "mission-control.tilos.com"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(
        web_server.app,
        base_url="https://mission-control.tilos.com",
    )
    yield client
    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


@pytest.fixture
def gated_app_direct():
    """web_server.app configured for gated mode WITHOUT a proxy prefix,
    for the Fly-direct deploy shape (no path mounting).
    """
    clear_providers()
    register_provider(StubAuthProvider())
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(
        web_server.app,
        base_url="https://fly-app.fly.dev",
    )
    yield client
    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


# ---------------------------------------------------------------------------
# Gate middleware: Location: header and 401 envelope respect prefix
# ---------------------------------------------------------------------------


class TestGateRedirectsCarryPrefix:
    def test_html_redirect_to_login_carries_prefix(self, gated_app_proxied):
        r = gated_app_proxied.get(
            "/sessions",
            headers={"x-forwarded-prefix": "/hermes"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        # /login redirect must include the prefix or the browser will
        # follow it to mission-control.tilos.com/login (which the proxy
        # doesn't route to the dashboard).
        assert r.headers["location"].startswith("/hermes/login"), (
            f"Location header lost prefix: {r.headers['location']!r}"
        )

    def test_api_401_envelope_login_url_carries_prefix(self, gated_app_proxied):
        r = gated_app_proxied.get(
            "/api/sessions",
            headers={"x-forwarded-prefix": "/hermes"},
            follow_redirects=False,
        )
        assert r.status_code == 401
        body = r.json()
        # SPA does window.location.assign(body.login_url); this MUST
        # include the prefix.
        assert body["login_url"].startswith("/hermes/login"), (
            f"401 envelope login_url lost prefix: {body['login_url']!r}"
        )

    def test_no_prefix_header_keeps_unprefixed_paths(self, gated_app_direct):
        """When no X-Forwarded-Prefix is sent, the Location header must
        NOT gain a phantom prefix — the Fly-direct deploy shape has no
        proxy at all."""
        r = gated_app_direct.get("/sessions", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"] == "/login?next=%2Fsessions"

    def test_malformed_prefix_header_is_ignored(self, gated_app_proxied):
        """A hostile proxy injects ``X-Forwarded-Prefix: <script>``;
        the normaliser rejects it and the gate falls back to unprefixed
        URLs. Defence against header-injection HTML inside Location."""
        r = gated_app_proxied.get(
            "/sessions",
            headers={"x-forwarded-prefix": "<script>alert(1)</script>"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        assert "<script>" not in r.headers["location"]
        assert r.headers["location"].startswith("/login")


# ---------------------------------------------------------------------------
# /auth/login: the OAuth redirect_uri reflects the proxy prefix
# ---------------------------------------------------------------------------


class TestOAuthRedirectUriRespectsPrefix:
    def test_redirect_uri_includes_prefix_in_authorize_url(
        self, gated_app_proxied
    ):
        """The IDP returns the user to the redirect_uri we sent. If we
        don't include the prefix, the IDP redirects to
        ``https://mission-control.tilos.com/auth/callback`` instead of
        ``https://mission-control.tilos.com/hermes/auth/callback`` — the
        former routes to the MC frontend, not the dashboard, so the
        user gets 404."""
        r = gated_app_proxied.get(
            "/auth/login?provider=stub",
            headers={"x-forwarded-prefix": "/hermes"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        location = r.headers["location"]
        # The stub IDP's redirect_url echoes the redirect_uri back. The
        # real IDP would consume it and later use it to redirect the
        # user, so the byte-exact value MUST include the prefix.
        from urllib.parse import urlparse, parse_qs, unquote
        # Stub returns ``{redirect_uri}?code=stub_code&state=...`` — so
        # we read up to the first ``?``.
        redirect_uri = location.split("?", 1)[0]
        # Absolute https URL including prefix.
        parsed = urlparse(redirect_uri)
        assert parsed.scheme == "https"
        assert parsed.netloc == "mission-control.tilos.com"
        assert parsed.path == "/hermes/auth/callback", (
            f"redirect_uri dropped prefix: {redirect_uri!r}"
        )

    def test_redirect_uri_no_prefix_when_direct_deploy(
        self, gated_app_direct
    ):
        r = gated_app_direct.get(
            "/auth/login?provider=stub", follow_redirects=False
        )
        assert r.status_code == 302
        redirect_uri = r.headers["location"].split("?", 1)[0]
        from urllib.parse import urlparse
        parsed = urlparse(redirect_uri)
        assert parsed.netloc == "fly-app.fly.dev"
        assert parsed.path == "/auth/callback"


# ---------------------------------------------------------------------------
# Cookies: Path attribute + __Host- / __Secure- prefix rules
# ---------------------------------------------------------------------------


class TestCookiePathRespectsPrefix:
    """Cookies must use ``Path=<prefix>`` when behind a proxy so they:

      a) get sent back to the dashboard on subsequent requests (browser
         only sends a cookie if the request path starts with the cookie's
         Path attribute);
      b) don't leak to other apps mounted alongside the dashboard
         (e.g. ``mission-control.tilos.com/billing/...``).

    When the cookie's Path can be ``/`` (no prefix, Fly-direct), we use
    the ``__Host-`` cookie prefix for additional hardening — it binds
    the cookie to the exact host (no Domain attribute) and requires Secure.
    """

    def test_pkce_cookie_uses_prefix_path(self, gated_app_proxied):
        r = gated_app_proxied.get(
            "/auth/login?provider=stub",
            headers={"x-forwarded-prefix": "/hermes"},
            follow_redirects=False,
        )
        cookies = r.headers.get_list("set-cookie")
        pkce = next(c for c in cookies if "hermes_session_pkce" in c)
        # Browser only sends cookie back if the request path is under
        # the cookie's Path attribute, so we need /hermes here. Bare
        # /-rooted cookies would still be sent but would also be sent
        # to /billing/... etc.
        assert "Path=/hermes" in pkce, (
            f"PKCE cookie has wrong Path: {pkce!r}"
        )

    def test_pkce_cookie_uses_secure_prefix_when_proxied(
        self, gated_app_proxied
    ):
        """Behind a proxy with Path != /, ``__Host-`` is disallowed
        (the spec requires Path=/). Fall back to ``__Secure-``, which
        carries the same Secure-required guarantee but allows any Path.
        """
        r = gated_app_proxied.get(
            "/auth/login?provider=stub",
            headers={"x-forwarded-prefix": "/hermes"},
            follow_redirects=False,
        )
        cookies = r.headers.get_list("set-cookie")
        # The PKCE cookie name carries the __Secure- prefix.
        pkce_candidates = [
            c for c in cookies
            if c.startswith("__Secure-hermes_session_pkce=")
        ]
        assert pkce_candidates, (
            f"PKCE cookie missing __Secure- prefix: {cookies!r}"
        )

    def test_pkce_cookie_uses_host_prefix_when_direct(
        self, gated_app_direct
    ):
        """Fly-direct deploy: Path=/ is available, so we can use the
        stricter ``__Host-`` prefix. This binds the cookie to the
        exact origin (no Domain attribute) — best practice for
        single-host single-app deploys."""
        r = gated_app_direct.get(
            "/auth/login?provider=stub", follow_redirects=False
        )
        cookies = r.headers.get_list("set-cookie")
        pkce_candidates = [
            c for c in cookies
            if c.startswith("__Host-hermes_session_pkce=")
        ]
        assert pkce_candidates, (
            f"PKCE cookie missing __Host- prefix on direct deploy: "
            f"{cookies!r}"
        )
        # __Host- requires Path=/ and Secure (cookies spec); both must
        # be present even if a regression flips one off.
        pkce = pkce_candidates[0]
        assert "Path=/" in pkce
        assert "Secure" in pkce

    def test_loopback_cookies_unprefixed(self):
        """Loopback HTTP dev: no Secure, no __Host- / __Secure-.
        The bare cookie name is the right choice — neither prefix is
        spec-compatible without Secure."""
        from fastapi import FastAPI
        from fastapi.responses import Response
        from hermes_cli.dashboard_auth.cookies import set_pkce_cookie

        app = FastAPI()

        @app.get("/set")
        def _set():
            r = Response("ok")
            set_pkce_cookie(r, payload="x", use_https=False)
            return r

        client = TestClient(app)
        r = client.get("/set")
        cookies = r.headers.get_list("set-cookie")
        # Bare cookie name, no prefix.
        assert any(c.startswith("hermes_session_pkce=") for c in cookies), (
            f"Loopback cookie should be bare-named: {cookies!r}"
        )
        # And no __Host- / __Secure- variant accidentally emitted.
        assert not any(
            c.startswith("__Host-") or c.startswith("__Secure-")
            for c in cookies
        )

    def test_cookies_read_back_round_trip_through_prefix(
        self, gated_app_proxied
    ):
        """The end-to-end property: after a successful OAuth round
        trip via the proxy, the session-AT cookie carries the
        __Secure- prefix AND Path=/hermes, so the next request under
        the same prefix is authenticated.

        Note on TestClient semantics: starlette's TestClient sees the
        literal request path (``/auth/login``, ``/auth/callback``) —
        not the public path the proxy displays to the browser
        (``/hermes/auth/login``, ``/hermes/auth/callback``). A cookie
        set with ``Path=/hermes`` would therefore NOT be sent back on
        the second request through TestClient even though it WOULD be
        sent by a real browser hitting ``/hermes/auth/callback``. To
        avoid baking that mismatch into the test, we inspect the
        ``Set-Cookie`` header on the callback's response WITHOUT
        depending on the PKCE cookie round-tripping through
        TestClient's jar — we drive /auth/callback with an explicit
        Cookie header that carries the PKCE value from /auth/login.
        """
        # /auth/login sets the PKCE cookie. Capture it from Set-Cookie.
        r1 = gated_app_proxied.get(
            "/auth/login?provider=stub",
            headers={"x-forwarded-prefix": "/hermes"},
            follow_redirects=False,
        )
        pkce_set = next(
            c for c in r1.headers.get_list("set-cookie")
            if "hermes_session_pkce" in c
        )
        # Parse "__Secure-hermes_session_pkce=...; HttpOnly; ...".
        pkce_kv = pkce_set.split(";", 1)[0]  # "__Secure-hermes_session_pkce=value"
        state = r1.headers["location"].split("state=")[1]

        # Round-trip the cookie by hand because TestClient's jar won't
        # automatically send a Path=/hermes cookie to a /auth/callback
        # request path.
        r2 = gated_app_proxied.get(
            f"/auth/callback?code=stub_code&state={state}",
            headers={
                "x-forwarded-prefix": "/hermes",
                "cookie": pkce_kv,
            },
            follow_redirects=False,
        )
        assert r2.status_code == 302, r2.text
        cookies = r2.headers.get_list("set-cookie")
        at_cookies = [
            c for c in cookies
            if c.startswith("__Secure-hermes_session_at=")
        ]
        assert at_cookies, (
            f"session_at missing __Secure- prefix: {cookies!r}"
        )
        assert "Path=/hermes" in at_cookies[0]
        assert "Secure" in at_cookies[0]
        assert "HttpOnly" in at_cookies[0]
