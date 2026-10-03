"""Regression tests for #126061: the PKCE state cookie outliving its TTL.

Two facets, both from the issue:

1. The cookie's ``Max-Age`` must cover IDP round trips that wait on an emailed
   verification code — 10 minutes was regularly too short (observed failures at
   78/195/43 minutes elapsed vs successes always under 48 s).
2. When the cookie is nonetheless missing/expired at ``/auth/callback``, the
   user must land on a readable login page, not a raw ``400`` JSON body that
   reads as a crash and re-400s on every reload.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient

from hermes_cli.dashboard_auth.cookies import (
    PKCE_COOKIE,
    set_pkce_cookie,
)


def _pkce_setter_app() -> FastAPI:
    """Minimal app exposing one route that mints a PKCE cookie (wire-shape pin)."""
    app = FastAPI()

    @app.get("/set-pkce")
    def set_pkce():
        r = Response("ok")
        set_pkce_cookie(
            r,
            payload={"provider": "stub", "state": "s", "verifier": "v"},
            use_https=True, prefix="",
        )
        return r

    return app


def _auth_routes_app() -> FastAPI:
    """The real dashboard-auth router, no middleware (callback is public)."""
    from hermes_cli.dashboard_auth.routes import router

    app = FastAPI()
    app.include_router(router)
    return app


def test_pkce_cookie_max_age_covers_emailed_code_round_trips():
    """The PKCE cookie wire ``Max-Age`` must be 30 minutes.

    The cookie is single-use, bound to one ``state``, and cleared on callback,
    so the TTL is a guess about the user's pace, not a security control: IDP
    flows that wait on an emailed verification code regularly exceed 10 minutes
    (#126061). Pin the wire value so a silent regression to a shorter window
    fails loudly.
    """
    client = TestClient(_pkce_setter_app())
    r = client.get("/set-pkce")
    pkce = next(
        c for c in r.headers.get_list("set-cookie")
        if c.startswith(f"__Host-{PKCE_COOKIE}=")
    )
    assert "max-age=1800" in pkce.lower(), (
        f"PKCE cookie Max-Age regressed below the 30-minute window: {pkce!r}"
    )


def test_callback_without_pkce_cookie_redirects_to_login_notice():
    """A missing/expired PKCE cookie at ``/auth/callback`` must bounce the user
    to a readable login page instead of returning ``400 {"detail": ...}``.

    The raw JSON body was indistinguishable from a server crash and parked the
    browser (or desktop login window) on a URL that 400s again on reload
    (#126061)."""
    client = TestClient(_auth_routes_app())
    r = client.get(
        "/auth/callback?code=stub_code&state=some_state",
        follow_redirects=False,
    )
    assert r.status_code == 302, f"expected a redirect, got {r.status_code}: {r.text!r}"
    assert r.headers["location"] == "/login?notice=signin_expired", (
        f"callback must redirect to the login page with the expiry notice: "
        f"{r.headers.get('location')!r}"
    )


def test_login_page_renders_whitelisted_notice():
    """``/login?notice=signin_expired`` renders the server-authored expiry line."""
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    clear_providers()
    register_provider(StubAuthProvider())
    try:
        client = TestClient(_auth_routes_app())
        r = client.get("/login?notice=signin_expired")
        assert r.status_code == 200
        assert "Your sign-in session expired" in r.text, (
            "login page must surface the expired-sign-in notice"
        )
    finally:
        clear_providers()


def test_login_page_notice_is_whitelist_only():
    """An unmapped ``notice=`` value renders NO notice.

    The query param is user-controlled; echoing it (or rendering markup from
    it) would make the login page a phishing/XSS vector. Only the fixed
    server-authored phrases in ``_LOGIN_NOTICES`` may render."""
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    clear_providers()
    register_provider(StubAuthProvider())
    try:
        client = TestClient(_auth_routes_app())
        r = client.get("/login?notice=Please+sign+in+at+evil.example")
        assert r.status_code == 200
        assert "evil.example" not in r.text, (
            "unmapped notice values must never be echoed into the login page"
        )
        assert '<p class="login-notice"' not in r.text, (
            "no notice element should render for an unmapped key"
        )
    finally:
        clear_providers()
