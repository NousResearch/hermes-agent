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
    _PKCE_MAX_AGE,
    set_pkce_cookie,
)


def _pkce_setter_app(provider: str = "stub") -> FastAPI:
    """Minimal app exposing one route that mints a PKCE cookie (wire-shape pin)."""
    app = FastAPI()

    @app.get("/set-pkce")
    def set_pkce():
        r = Response("ok")
        set_pkce_cookie(
            r,
            payload={"provider": provider, "state": "s", "verifier": "v"},
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


def test_native_pending_ttl_matches_the_pkce_cookie_window():
    """The native-flow pending entry is keyed by the broker_state riding in the PKCE
    cookie, so both TTLs guard one interactive login and must move together. With a
    shorter pending TTL, t ∈ (600, 1800] still validates the cookie, the upstream
    callback succeeds — and then the desktop gets ``Native login expired`` anyway."""
    from hermes_cli.dashboard_auth import native_flow

    assert native_flow._PENDING_TTL_SECONDS == _PKCE_MAX_AGE, (
        "_PENDING_TTL_SECONDS drifted from _PKCE_MAX_AGE: the broker_state outlives "
        "its pending entry, turning late-but-valid callbacks into 400s"
    )
    assert _PKCE_MAX_AGE >= 30 * 60


def _start_web_login(client: TestClient) -> tuple[str, object]:
    """Drive ``/auth/login?provider=stub``; the stub bounces straight back to the
    callback, so the 302 carries both the PKCE cookie and the one-shot ``state``."""
    r = client.get("/auth/login?provider=stub", follow_redirects=False)
    assert r.status_code == 302, r.text
    state = r.headers["location"].split("state=")[1]
    return state, r.cookies


def test_callback_failures_land_on_the_login_notice_page():
    """Every dead-end failure at ``/auth/callback`` must land the browser on the
    readable login page, not a raw 400 JSON body (#126061): IDP-reported errors,
    a state mismatch (stale cookie / cross-tab replay), and an invalid code are
    all normal user-facing outcomes, same as the missing-cookie case."""
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    clear_providers()
    register_provider(StubAuthProvider())
    try:
        client = TestClient(_auth_routes_app())
        state, cookies = _start_web_login(client)

        cases = [
            (f"/auth/callback?error=access_denied&state={state}", "signin_failed"),
            (f"/auth/callback?code=stub_code&state=wrong_state", "signin_failed"),
            (f"/auth/callback?code=not_the_code&state={state}", "signin_failed"),
        ]
        for url, notice in cases:
            r = client.get(url, cookies=cookies, follow_redirects=False)
            assert r.status_code == 302, f"{url!r}: expected a redirect, got {r.status_code}"
            assert r.headers["location"] == f"/login?notice={notice}", (
                f"{url!r} must land on the login notice page: {r.headers.get('location')!r}"
            )
    finally:
        clear_providers()


def test_callback_unknown_cookie_provider_lands_on_expiry_notice():
    """A cookie naming a provider this gateway no longer serves is the same dead end
    as an expired cookie (config change / stale browser state): land it on the
    expiry notice rather than a raw 400."""
    setter = TestClient(_pkce_setter_app(provider="no-longer-registered"))
    cookies = setter.get("/set-pkce").cookies
    r = TestClient(_auth_routes_app()).get(
        "/auth/callback?code=x&state=s", cookies=cookies, follow_redirects=False
    )
    assert r.status_code == 302, r.text
    assert r.headers["location"] == "/login?notice=signin_expired"
