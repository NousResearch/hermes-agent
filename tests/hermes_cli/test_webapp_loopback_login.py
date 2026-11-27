"""Loopback Webapp stays strictly launch-link-only (no browser login cookies).

Regression suite for https://github.com/NousResearch/hermes-agent/issues/159500:
Cookies on ``127.0.0.1`` are not isolated by port (RFC 6265 §8.5), so a loopback
gateway-hosted Webapp (``ui_surface="webapp"``, ``auth_required=False``,
``withhold_session_token=True`` — i.e. ``private_launch(app.state)``) must NEVER mint or
accept browser login cookies even when dashboard auth providers (e.g. ``dashboard.basic_auth``
or OAuth) are configured:
  - ``GET /login`` renders ``hermes webapp`` launch-link guidance with no login form or cookies
  - ``GET /api/auth/providers`` answers ``{"providers": []}`` when authenticated (and stays 401
    unauthenticated on loopback)
  - ``POST /auth/password-login``, ``GET /auth/login``, and ``GET /auth/callback`` refuse
    browser cookie sign-ins cleanly (403 with ``hermes webapp`` guidance) and set no cookies
  - replaying valid provider + sign-in cookies (with no ``Origin`` header) returns 401 on
    ``/api/auth/me``, ``/api/profiles``, and ``/api/webapp/socket-ticket?socket=host-terminal``
  - ``hermes webapp`` launch-link tokens, gated non-loopback password/OAuth logins, and
    standalone loopback dashboards continue to work unchanged
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import types
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from tests.hermes_cli.test_web_host_terminal_persistence import metadata
from tests.hermes_cli.test_webapp_socket_tickets import _TOKEN, serve_webapp

WEBAPP = "http://127.0.0.1:9119"
GATED_WEBAPP = "https://webapp.example.com"
USER, PASSWORD = "owner", "throwaway-test-password"


@pytest.fixture
def loopback_webapp(tmp_path, monkeypatch):
    """Loopback gateway-hosted Webapp with basic_auth and stub OAuth registered."""
    from hermes_cli import webapp_sign_in
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from hermes_cli.dashboard_auth.routes import _reset_password_rate_limit
    from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests
    from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    webapp = serve_webapp(tmp_path, monkeypatch)
    dist = tmp_path / "dist-webapp"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><html><head></head><body>WEBAPP</body></html>", encoding="utf-8")

    state = webapp.server.app.state
    monkeypatch.setattr(state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(state, "bound_port", 9119, raising=False)
    monkeypatch.setattr(state, "auth_required", False, raising=False)
    monkeypatch.setattr(state, "withhold_session_token", True, raising=False)
    monkeypatch.setattr(state, "web_dist", dist, raising=False)
    monkeypatch.setattr(state, "webapp_home", tmp_path / ".hermes", raising=False)
    monkeypatch.setattr(state, "webapp_window_tickets", {}, raising=False)
    monkeypatch.setattr(state, "listener_public_url", "", raising=False)
    monkeypatch.setattr("hermes_cli.dashboard_auth.prefix.resolve_public_url", lambda *_a, **_kw: "")

    _reset_for_tests()
    webapp_sign_in._reset_for_tests()
    _reset_password_rate_limit()
    clear_providers()
    basic = BasicAuthProvider(username=USER, password_hash=hash_password(PASSWORD), secret=b"k" * 32)
    stub = StubAuthProvider()
    register_provider(basic)
    register_provider(stub)
    webapp.basic = basic
    webapp.stub = stub
    webapp.browser = lambda base_url=WEBAPP: TestClient(webapp.server.app, base_url=base_url)
    try:
        yield webapp
    finally:
        clear_providers()
        _reset_for_tests()
        webapp_sign_in._reset_for_tests()
        _reset_password_rate_limit()


def _gateway_suffix(app) -> str:
    from hermes_cli import webapp_sign_in

    dummy = Request({"type": "http", "method": "GET", "path": "/", "headers": [], "app": app})
    return webapp_sign_in.cookie_gateway(dummy)


def _mint_replayed_cookies(loopback_webapp, *, provider: str) -> dict[str, str]:
    """Build the exact four host-only session cookies a browser login sets on a gated Webapp."""
    if provider == "basic":
        session = loopback_webapp.basic.complete_password_login(username=USER, password=PASSWORD)
    else:
        started = loopback_webapp.stub.start_login(redirect_uri=f"{WEBAPP}/auth/callback")
        pkce = dict(
            seg.split("=", 1)
            for seg in started.cookie_payload["hermes_session_pkce"].split(";")
            if "=" in seg
        )
        session = loopback_webapp.stub.complete_login(
            code="stub_code",
            state=pkce["state"],
            code_verifier=pkce["verifier"],
            redirect_uri=f"{WEBAPP}/auth/callback",
        )
    gw = _gateway_suffix(loopback_webapp.server.app)
    token = loopback_webapp.server._SESSION_TOKEN
    binding = hmac.new(token.encode("utf-8"), b"webapp-sign-in", hashlib.sha256).hexdigest()
    return {
        f"hermes_session_at_{gw}": session.access_token,
        f"hermes_session_rt_{gw}": session.refresh_token,
        f"hermes_session_provider_{gw}": session.provider,
        f"hermes_webapp_signin_{gw}": binding,
    }


def test_loopback_webapp_login_page_explains_hermes_webapp_without_forms(loopback_webapp):
    from hermes_cli.dashboard_auth import clear_providers

    browser = loopback_webapp.browser()
    for path in ("/login", "/login?next=%2Fsessions"):
        page = browser.get(path, follow_redirects=False)
        assert page.status_code == 200
        assert "no-store" in page.headers.get("cache-control", "")
        assert "set-cookie" not in page.headers
        assert "hermes webapp" in page.text
        assert "<form" not in page.text
        assert "provider-form" not in page.text
        assert "provider-btn" not in page.text
        assert "/auth/login?provider=" not in page.text
        assert "/auth/password-login" not in page.text

    # Provider bootstrap requires the loopback token when unauthenticated and advertises no
    # browser sign-in providers when authenticated (with or without registered providers).
    assert browser.get("/api/auth/providers").status_code == 401
    token_headers = {"X-Hermes-Session-Token": _TOKEN}
    assert browser.get("/api/auth/providers", headers=token_headers).status_code == 200
    assert browser.get("/api/auth/providers", headers=token_headers).json() == {"providers": []}

    # GET / stays non-redirecting, withholds _SESSION_TOKEN, and stays authRequired=false.
    root = browser.get("/", follow_redirects=False)
    assert root.status_code == 200
    assert "window.__HERMES_AUTH_REQUIRED__=false;" in root.text
    assert loopback_webapp.server._SESSION_TOKEN not in root.text

    clear_providers()
    assert browser.get("/api/auth/providers", headers=token_headers).status_code == 200
    assert browser.get("/api/auth/providers", headers=token_headers).json() == {"providers": []}


def test_loopback_webapp_password_login_refuses_without_setting_cookies(loopback_webapp):
    from hermes_cli.dashboard_auth.cookies import encode_pkce_payload

    gw = _gateway_suffix(loopback_webapp.server.app)

    for extra_cookie in (None, encode_pkce_payload({"provider": "basic", "broker": "forged-unknown-broker"})):
        browser = loopback_webapp.browser()
        if extra_cookie is not None:
            browser.cookies.set(f"hermes_session_pkce_{gw}", extra_cookie)
        resp = browser.post(
            "/auth/password-login",
            json={"provider": "basic", "username": USER, "password": PASSWORD, "next": "/"},
        )
        assert resp.status_code == 403, resp.text
        assert "hermes webapp" in resp.json()["detail"]
        assert "set-cookie" not in resp.headers
        assert not any(name.startswith("hermes_session_at") or name.startswith("hermes_session_rt")
                       for name in browser.cookies.keys())


@pytest.mark.parametrize("provider", ["basic", "stub", "nonexistent"])
def test_loopback_webapp_oauth_login_start_refuses_without_setting_cookies(loopback_webapp, provider):
    browser = loopback_webapp.browser()
    resp = browser.get(f"/auth/login?provider={provider}", follow_redirects=False)
    assert resp.status_code == 403, resp.text
    assert "hermes webapp" in resp.json()["detail"]
    assert "location" not in resp.headers
    assert "set-cookie" not in resp.headers
    assert not dict(browser.cookies)


def test_loopback_webapp_oauth_callback_refuses_without_setting_cookies(loopback_webapp):
    from hermes_cli.dashboard_auth.cookies import encode_pkce_payload

    gw = _gateway_suffix(loopback_webapp.server.app)
    started = loopback_webapp.stub.start_login(redirect_uri=f"{WEBAPP}/auth/callback")
    pkce = dict(
        seg.split("=", 1)
        for seg in started.cookie_payload["hermes_session_pkce"].split(";")
        if "=" in seg
    )
    real_state, real_verifier = pkce["state"], pkce["verifier"]

    for extra_cookie in (
        encode_pkce_payload({"provider": "stub", "state": real_state, "verifier": real_verifier}),
        None,
        encode_pkce_payload({
            "provider": "stub",
            "state": real_state,
            "verifier": real_verifier,
            "broker": "forged-unknown-broker",
        }),
    ):
        browser = loopback_webapp.browser()
        if extra_cookie is not None:
            browser.cookies.set(f"hermes_session_pkce_{gw}", extra_cookie)
        resp = browser.get(f"/auth/callback?code=stub_code&state={real_state}", follow_redirects=False)
        assert resp.status_code == 403, resp.text
        assert "hermes webapp" in resp.json()["detail"]
        assert "location" not in resp.headers
        assert "set-cookie" not in resp.headers
        assert not any(name.startswith("hermes_session_at") or name.startswith("hermes_session_rt")
                       for name in browser.cookies.keys())


@pytest.mark.parametrize("provider", ["basic", "stub"])
def test_loopback_webapp_rejects_replayed_session_cookies_across_rest_and_host_terminal(loopback_webapp, provider):
    """RFC 6265 §8.5 defense: valid login cookies (even with matching webapp_signin binding and no
    Origin/Sec-Fetch-Site headers) must never authorize REST or host-terminal sockets on loopback."""
    replayed = _mint_replayed_cookies(loopback_webapp, provider=provider)
    peer = loopback_webapp.browser()
    for name, value in replayed.items():
        peer.cookies.set(name, value)

    assert peer.get("/api/auth/me").status_code == 401
    assert peer.get("/api/profiles").status_code == 401
    assert peer.post("/api/webapp/socket-ticket?socket=host-terminal").status_code == 401
    assert peer.post("/api/webapp/socket-ticket?socket=gateway").status_code == 401
    assert peer.post("/api/webapp/window-ticket").status_code == 401

    root = peer.get("/", follow_redirects=False)
    assert root.status_code == 200
    assert "window.__HERMES_AUTH_REQUIRED__=false;" in root.text
    assert loopback_webapp.server._SESSION_TOKEN not in root.text


def test_loopback_webapp_launch_link_token_works_beside_registered_providers_and_stale_cookies(loopback_webapp):
    browser = loopback_webapp.browser()
    for name, value in _mint_replayed_cookies(loopback_webapp, provider="basic").items():
        browser.cookies.set(name, value)

    headers = {"X-Hermes-Session-Token": _TOKEN}
    me = browser.get("/api/auth/me", headers=headers)
    assert me.status_code == 200, me.text
    assert me.json()["provider"] == "loopback"
    assert browser.get("/api/profiles", headers=headers).status_code == 200
    assert browser.post("/api/webapp/socket-ticket?socket=gateway", headers=headers).status_code == 200
    assert browser.post("/api/webapp/window-ticket", headers=headers).status_code == 200

    term = browser.post("/api/webapp/socket-ticket?socket=host-terminal", headers=headers)
    assert term.status_code == 200, term.text
    term_ticket = term.json()["ticket"]
    with TestClient(loopback_webapp.host, base_url=WEBAPP) as shells:
        with shells.websocket_connect(f"ws://127.0.0.1:9119/api/host-terminal?ticket={term_ticket}") as ws:
            assert metadata(ws)["terminalId"]


@pytest.mark.parametrize("method", ["password", "oauth"])
def test_gated_non_loopback_webapp_password_and_oauth_login_still_work(loopback_webapp, monkeypatch, method):
    state = loopback_webapp.server.app.state
    monkeypatch.setattr(state, "bound_host", "0.0.0.0", raising=False)
    monkeypatch.setattr(state, "auth_required", True, raising=False)

    browser = loopback_webapp.browser(base_url=GATED_WEBAPP)
    page = browser.get("/login")
    assert page.status_code == 200
    assert '<form class="provider-form" data-provider="basic"' in page.text
    assert '/auth/login?provider=stub' in page.text

    providers = browser.get("/api/auth/providers").json()["providers"]
    assert {p["name"] for p in providers} == {"basic", "stub"}

    if method == "password":
        resp = browser.post(
            "/auth/password-login",
            json={"provider": "basic", "username": USER, "password": PASSWORD, "next": "/"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"ok": True, "next": "/"}
    else:
        started = browser.get("/auth/login?provider=stub", follow_redirects=False)
        assert started.status_code == 302
        oauth_state = parse_qs(urlsplit(started.headers["location"]).query)["state"][0]
        done = browser.get(f"/auth/callback?code=stub_code&state={oauth_state}", follow_redirects=False)
        assert done.status_code == 302

    assert any("hermes_webapp_signin" in name for name in browser.cookies.keys())
    me = browser.get("/api/auth/me")
    assert me.status_code == 200, me.text
    assert me.json()["provider"] == ("basic" if method == "password" else "stub")
    assert browser.get("/api/profiles").status_code == 200


def test_standalone_loopback_dashboard_is_unaffected(loopback_webapp, monkeypatch):
    """Off the Webapp surface, standalone loopback dashboard keeps main's behavior."""
    monkeypatch.setattr(loopback_webapp.server.app.state, "ui_surface", "dashboard")
    browser = loopback_webapp.browser()

    page = browser.get("/login")
    assert page.status_code == 200
    assert '<form class="provider-form" data-provider="basic"' in page.text
    assert "Open with hermes webapp" not in page.text

    token_headers = {"X-Hermes-Session-Token": _TOKEN}
    providers = browser.get("/api/auth/providers", headers=token_headers).json()["providers"]
    assert {p["name"] for p in providers} == {"basic", "stub"}

    assert browser.get("/api/profiles").status_code == 401
    assert browser.get("/api/profiles", headers=token_headers).status_code == 200


def test_gateway_dashboard_mount_loopback_webapp_stays_launch_link_only(tmp_path, monkeypatch):
    """Real GatewayDashboardMount on 127.0.0.1 sets private_launch=True, refuses browser cookie
    sign-ins with hermes webapp guidance, and authorizes launch-link tokens."""
    from gateway.run_api import GatewayDashboardMount
    from gateway.webapp_listener import WebappMount
    from hermes_cli import web_server, webapp_sign_in
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from hermes_cli.dashboard_auth.routes import _reset_password_rate_limit
    from hermes_constants import get_hermes_home
    from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password

    for key in ("bound_host", "bound_port", "auth_required", "trusted_public_hosts"):
        monkeypatch.setattr(web_server.app.state, key, getattr(web_server.app.state, key, None), raising=False)
    dist = tmp_path / "dist-webapp"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><html><head></head><body>WEBAPP</body></html>", encoding="utf-8")
    state = types.SimpleNamespace(
        bound_host="127.0.0.1", bound_port=9119,
        webapp=WebappMount(dist, get_hermes_home()),
    )
    mount = GatewayDashboardMount(types.SimpleNamespace(session_authority=None), state)
    clear_providers()
    webapp_sign_in._reset_for_tests()
    _reset_password_rate_limit()
    register_provider(BasicAuthProvider(username=USER, password_hash=hash_password(PASSWORD), secret=b"k" * 32))

    async def scenario():
        web = await mount.ensure()
        try:
            assert (web.app.state.auth_required, web.app.state.ui_surface, web.app.state.withhold_session_token) == (
                False, "webapp", True
            )
            transport = httpx.ASGITransport(app=mount)
            async with httpx.AsyncClient(transport=transport, base_url=WEBAPP) as client:
                login_page = await client.get("/login")
                assert login_page.status_code == 200
                assert "hermes webapp" in login_page.text
                assert "<form" not in login_page.text
                assert "set-cookie" not in login_page.headers

                headers = {"X-Hermes-Session-Token": web._SESSION_TOKEN}
                assert (await client.get("/api/auth/providers")).status_code == 401
                assert (await client.get("/api/auth/providers", headers=headers)).json() == {"providers": []}

                pw = await client.post(
                    "/auth/password-login",
                    json={"provider": "basic", "username": USER, "password": PASSWORD, "next": "/"},
                )
                assert pw.status_code == 403, pw.text
                assert "hermes webapp" in pw.json()["detail"]
                assert "set-cookie" not in pw.headers

                oauth_start = await client.get("/auth/login?provider=basic", follow_redirects=False)
                assert oauth_start.status_code == 403, oauth_start.text
                assert "hermes webapp" in oauth_start.json()["detail"]
                assert "set-cookie" not in oauth_start.headers

                assert (await client.get("/api/profiles")).status_code == 401
                assert (await client.get("/api/auth/me")).status_code == 401
                assert (await client.post("/api/webapp/socket-ticket?socket=host-terminal")).status_code == 401

                root = await client.get("/")
                assert root.status_code == 200
                assert "window.__HERMES_AUTH_REQUIRED__=false;" in root.text
                assert web._SESSION_TOKEN not in root.text

                assert (await client.get("/api/profiles", headers=headers)).status_code == 200
                ticket = await client.post("/api/webapp/socket-ticket?socket=gateway", headers=headers)
                assert ticket.status_code == 200, ticket.text
                assert ticket.json()["ticket"]
        finally:
            await mount.close()

    try:
        asyncio.run(scenario())
    finally:
        clear_providers()
        webapp_sign_in._reset_for_tests()
        _reset_password_rate_limit()
