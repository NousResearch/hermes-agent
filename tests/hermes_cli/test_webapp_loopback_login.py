"""Loopback Webapp browser sign-in (/login) authorizes REST, socket-tickets, and WS end-to-end.

Regression suite for https://github.com/NousResearch/hermes-agent/issues/159500:
On a loopback gateway-hosted Webapp (``ui_surface="webapp"``, ``auth_required=False``,
``withhold_session_token=True``) with a configured dashboard auth provider (e.g.
``dashboard.basic_auth``), signing in via ``/login`` / ``POST /auth/password-login`` (or OAuth
``/auth/callback``) must authorize subsequent ``/api/*`` calls (``/api/profiles``,
``/api/auth/me``, ``/api/webapp/socket-ticket``) and bootstrap the SPA into cookie mode while
preserving all existing security invariants:
  - ``withhold_session_token`` stays enforced (``_SESSION_TOKEN`` never appears in HTML)
  - unauthenticated and forged-cookie calls get 401
  - cookie-authenticated writes enforce Webapp own-origin CSRF protection (403 on foreign origin)
  - cookie-authenticated tabs cannot extract ``_SESSION_TOKEN`` via ``/api/webapp/window-ticket``
  - ``hermes webapp --sign-out`` (credential rotation) revokes loopback cookie logins and tickets
  - launch-link (``X-Hermes-Session-Token``) sessions still work, even beside stale cookies
"""
from __future__ import annotations

import asyncio
import json
import types
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.hermes_cli.test_web_host_terminal_persistence import metadata
from tests.hermes_cli.test_webapp_socket_tickets import _TOKEN, serve_webapp

WEBAPP = "http://127.0.0.1:9119"
USER, PASSWORD = "owner", "throwaway-test-password"
OWN_PAGE = {"Origin": WEBAPP, "Sec-Fetch-Site": "same-origin"}
FOREIGN_PAGES = (
    {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"},
    {"Origin": "http://127.0.0.1:8080", "Sec-Fetch-Site": "same-site"},
    {"Origin": "https://evil.example"},
    {"Origin": "null", "Sec-Fetch-Site": "cross-site"},
)


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
    monkeypatch.setattr("hermes_cli.dashboard_auth.prefix.resolve_public_url", lambda *_a, **_kw: "")

    _reset_for_tests()
    webapp_sign_in._reset_for_tests()
    _reset_password_rate_limit()
    clear_providers()
    register_provider(BasicAuthProvider(username=USER, password_hash=hash_password(PASSWORD), secret=b"k" * 32))
    register_provider(StubAuthProvider())
    webapp.browser = lambda: TestClient(webapp.server.app, base_url=WEBAPP)
    try:
        yield webapp
    finally:
        clear_providers()
        _reset_for_tests()
        webapp_sign_in._reset_for_tests()
        _reset_password_rate_limit()


def _sign_in_password(client: TestClient) -> None:
    resp = client.post(
        "/auth/password-login",
        json={"provider": "basic", "username": USER, "password": PASSWORD, "next": "/"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True, "next": "/"}


def _sign_in_oauth(client: TestClient) -> None:
    started = client.get("/auth/login?provider=stub", follow_redirects=False)
    assert started.status_code == 302
    state = parse_qs(urlsplit(started.headers["location"]).query)["state"][0]
    done = client.get(f"/auth/callback?code=stub_code&state={state}", follow_redirects=False)
    assert done.status_code == 302


SIGN_INS = {"password": _sign_in_password, "oauth": _sign_in_oauth}


@pytest.mark.parametrize("method", SIGN_INS)
def test_loopback_webapp_login_authorizes_rest_socket_tickets_and_spa(loopback_webapp, method):
    browser = loopback_webapp.browser()
    SIGN_INS[method](browser)

    # Cookie jar carries the Webapp sign-in binding alongside the session cookies.
    assert any(name.startswith("hermes_webapp_signin") for name in browser.cookies.keys())

    # REST routes work with the session cookie (no X-Hermes-Session-Token header).
    profiles = browser.get("/api/profiles")
    assert profiles.status_code == 200, profiles.text
    me = browser.get("/api/auth/me")
    assert me.status_code == 200, me.text
    assert me.json()["provider"] == ("basic" if method == "password" else "stub")

    # Socket tickets mint for both gateway and host-terminal sockets.
    gw = browser.post("/api/webapp/socket-ticket?socket=gateway", headers=OWN_PAGE)
    assert gw.status_code == 200, gw.text
    assert gw.json()["ticket"]

    term = browser.post("/api/webapp/socket-ticket?socket=host-terminal", headers=OWN_PAGE)
    assert term.status_code == 200, term.text
    term_ticket = term.json()["ticket"]

    # The minted host-terminal ticket opens a real WebSocket once.
    with TestClient(loopback_webapp.host, base_url=WEBAPP) as shells:
        with shells.websocket_connect(f"ws://127.0.0.1:9119/api/host-terminal?ticket={term_ticket}") as ws:
            assert metadata(ws)["terminalId"]

    # GET / switches the SPA to cookie-auth mode while STILL withholding _SESSION_TOKEN.
    page = browser.get("/")
    assert page.status_code == 200
    assert "window.__HERMES_AUTH_REQUIRED__=true;" in page.text
    assert loopback_webapp.server._SESSION_TOKEN not in page.text


def test_loopback_webapp_cookie_refresh_rotates_and_authorizes(loopback_webapp):
    browser = loopback_webapp.browser()
    _sign_in_password(browser)

    # Evict the access-token cookie (normal TTL expiry), leaving RT + webapp_signin binding.
    for name in list(browser.cookies.keys()):
        if name.startswith("hermes_session_at"):
            browser.cookies.delete(name)

    profiles = browser.get("/api/profiles")
    assert profiles.status_code == 200, profiles.text
    assert any(name.startswith("hermes_session_at") for name in browser.cookies.keys())


def test_loopback_webapp_unauthenticated_and_forged_cookies_stay_401_and_token_withheld(loopback_webapp):
    anon = loopback_webapp.browser()

    # GET / never publishes _SESSION_TOKEN and stays authRequired=false for anonymous callers.
    page = anon.get("/")
    assert page.status_code == 200
    assert "window.__HERMES_AUTH_REQUIRED__=false;" in page.text
    assert loopback_webapp.server._SESSION_TOKEN not in page.text

    # Unauthenticated protected REST and socket-ticket calls remain 401.
    assert anon.get("/api/profiles").status_code == 401
    assert anon.get("/api/auth/me").status_code == 401
    assert anon.post("/api/webapp/socket-ticket?socket=gateway").status_code == 401

    # Forged session cookie without valid binding or provider token is rejected with 401.
    from hermes_cli import webapp_sign_in
    from starlette.requests import Request

    dummy = Request({"type": "http", "method": "GET", "path": "/", "headers": [], "app": loopback_webapp.server.app})
    gw = webapp_sign_in.cookie_gateway(dummy)
    anon.cookies.set(f"hermes_session_at_{gw}", "forged-token")
    assert anon.get("/api/profiles").status_code == 401
    assert anon.post("/api/webapp/socket-ticket?socket=gateway").status_code == 401


def test_loopback_webapp_cookie_session_cannot_extract_session_token_via_window_ticket(loopback_webapp):
    browser = loopback_webapp.browser()
    _sign_in_password(browser)

    denied = browser.post("/api/webapp/window-ticket", headers=OWN_PAGE)
    assert denied.status_code == 403, denied.text
    assert loopback_webapp.server._SESSION_TOKEN not in denied.text


def test_loopback_webapp_cookie_writes_enforce_own_origin_csrf(loopback_webapp):
    browser = loopback_webapp.browser()
    _sign_in_password(browser)

    for foreign in FOREIGN_PAGES:
        assert browser.post("/api/webapp/socket-ticket?socket=gateway", headers=foreign).status_code == 403, foreign
        assert browser.post("/auth/logout", headers=foreign, follow_redirects=False).status_code == 403, foreign

    # Same-origin browser and non-browser curl (no Origin, no Sec-Fetch-Site) succeed.
    assert browser.post("/api/webapp/socket-ticket?socket=gateway", headers=OWN_PAGE).status_code == 200
    assert browser.post("/api/webapp/socket-ticket?socket=gateway").status_code == 200


def test_loopback_webapp_sign_out_revokes_cookie_session_and_tickets(loopback_webapp, monkeypatch):
    browser = loopback_webapp.browser()
    _sign_in_password(browser)
    assert browser.get("/api/profiles").status_code == 200
    minted = browser.post("/api/webapp/socket-ticket?socket=host-terminal", headers=OWN_PAGE).json()["ticket"]

    # Simulate `hermes webapp --sign-out` rotating the browser credential.
    monkeypatch.setattr(loopback_webapp.server, "_SESSION_TOKEN", "z" * 43)

    # API routes reject with structured 401 (reason=signed_out); old socket ticket is rejected.
    rejected = browser.get("/api/profiles")
    assert rejected.status_code == 401
    assert rejected.json()["reason"] == "signed_out"
    assert browser.post("/api/webapp/socket-ticket?socket=gateway", headers=OWN_PAGE).status_code == 401

    with TestClient(loopback_webapp.host, base_url=WEBAPP) as shells:
        with pytest.raises(WebSocketDisconnect) as closed:
            with shells.websocket_connect(f"ws://127.0.0.1:9119/api/host-terminal?ticket={minted}") as ws:
                ws.receive_text()
        assert closed.value.code == 4401

    # GET / does NOT trap a later launch link in a /login redirect: it serves index.html with authRequired=false.
    page = browser.get("/", follow_redirects=False)
    assert page.status_code == 200
    assert "window.__HERMES_AUTH_REQUIRED__=false;" in page.text
    assert loopback_webapp.server._SESSION_TOKEN not in page.text


def test_loopback_webapp_launch_link_token_still_works_beside_stale_cookies(loopback_webapp, monkeypatch):
    browser = loopback_webapp.browser()
    _sign_in_password(browser)
    # Rotate credential so the cookies in this jar are stale/signed-out, then authenticate with the new token.
    fresh = "n" * 43
    monkeypatch.setattr(loopback_webapp.server, "_SESSION_TOKEN", fresh)
    headers = {"X-Hermes-Session-Token": fresh}

    assert browser.get("/api/profiles", headers=headers).status_code == 200
    assert browser.post("/api/webapp/socket-ticket?socket=gateway", headers=headers).status_code == 200
    assert browser.post("/api/webapp/window-ticket", headers=headers).status_code == 200


def test_standalone_loopback_dashboard_is_unaffected(loopback_webapp, monkeypatch):
    """Off the Webapp surface, loopback dashboard keeps main's session-token-only behavior."""
    monkeypatch.setattr(loopback_webapp.server.app.state, "ui_surface", "dashboard")
    browser = loopback_webapp.browser()
    _sign_in_password(browser)
    assert not any(name.startswith("hermes_webapp_signin") for name in browser.cookies.keys())
    rejected = browser.get("/api/profiles")
    assert rejected.status_code == 401
    assert rejected.json() == {"detail": "Unauthorized"}
    assert browser.get("/api/profiles", headers={"X-Hermes-Session-Token": _TOKEN}).status_code == 200


def _sign_out_via_control_verb(loopback_webapp, tmp_path) -> dict:
    """Invoke the gateway's ``webapp-sign-out`` control verb against *loopback_webapp*."""
    from gateway.control_socket import GatewayControlServer
    from gateway.webapp_listener import register_control_verbs

    home = tmp_path / ".hermes"
    runner = types.SimpleNamespace(
        session_control_server=GatewayControlServer(home),
        webapp_status={"state": "serving", "url": WEBAPP, "reason": None},
        session_api=types.SimpleNamespace(app=types.SimpleNamespace(web=loopback_webapp.server)),
    )
    register_control_verbs(runner, home)
    raw = json.dumps({"protocol": 1, "verb": "webapp-sign-out", "id": 1, "params": {}}).encode()
    answer = json.loads(runner.session_control_server.handle_request_line(raw, "same-user"))
    assert answer["ok"], answer
    assert loopback_webapp.server._SESSION_TOKEN != _TOKEN
    return answer["result"]


def _mint_native_file_ticket(loopback_webapp, monkeypatch, home, route: str, path: str) -> str:
    """Mint a file ticket through HTTP using a real ``native-http`` gateway grant."""
    from gateway.runtime_bootstrap import TicketStore
    from gateway.session_authorities import SessionAuthorities
    from hermes_cli.dashboard_auth.native_http import HEADER as NATIVE_HEADER

    resolved_home = home.resolve()
    resolved_home.mkdir(parents=True, exist_ok=True)
    authorities = SessionAuthorities(resolved_home, multiplexed=False)
    authorities.add(resolved_home, types.SimpleNamespace(profile_id=str(resolved_home), instance_id="daemon"))
    store = TicketStore("daemon", authorities.profile_ids())
    runner = types.SimpleNamespace(
        session_authorities=authorities,
        session_ticket_store=store,
        _draining=False,
        session_runtime_descriptor={
            "state": "ready",
            "instance_id": "daemon",
            "served_profiles": authorities.served_profiles(),
        },
    )
    monkeypatch.setattr(loopback_webapp.server.app.state, "gateway_runner", runner, raising=False)
    grant = store.mint(
        profile_id=str(resolved_home),
        subject="desktop",
        purpose="native-http",
    )

    async def loopback_peer(scope, receive, send):
        scope["hermes.gateway_socket_peer"] = ("127.0.0.1", 50000)
        await loopback_webapp.server.app(scope, receive, send)

    native_client = TestClient(loopback_peer, base_url=WEBAPP)
    resp = native_client.post(
        "/api/files/ticket",
        headers={NATIVE_HEADER: grant},
        json={"route": route, "path": path},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["ticket"]


def _mint_service_file_ticket(loopback_webapp, route: str, path: str) -> str:
    """Mint a file ticket through HTTP using a registered bearer token-auth service provider."""
    from hermes_cli.dashboard_auth import clear_providers, list_providers, register_provider
    from hermes_cli.dashboard_auth.base import TokenPrincipal
    from hermes_cli.dashboard_auth.token_auth import clear_token_routes, register_token_route
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    class _ServiceTokenProvider(StubAuthProvider):
        name = "test-service-token"
        display_name = "Test Service Token"
        supports_session = False
        supports_token = True

        def verify_token(self, *, token: str):
            if token == "svc-secret":
                return TokenPrincipal(principal="svc", provider=self.name, scopes=("files",))
            return None

    prior_providers = list_providers()
    register_provider(_ServiceTokenProvider())
    register_token_route("/api/files/ticket")
    try:
        svc_client = loopback_webapp.browser()
        resp = svc_client.post(
            "/api/files/ticket",
            headers={"Authorization": "Bearer svc-secret"},
            json={"route": route, "path": path},
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["ticket"]
    finally:
        clear_token_routes()
        clear_providers()
        for provider in prior_providers:
            register_provider(provider)


@pytest.mark.parametrize("route", ["download", "stream"])
def test_loopback_webapp_sign_out_revokes_cookie_file_tickets_only(loopback_webapp, tmp_path, monkeypatch, route):
    """Cookie-authenticated file tickets (download and stream) inherit the Webapp sign-in
    generation and are purged on sign-out; native and service file tickets remain intact."""
    from hermes_cli import web_server_file_tickets as tickets

    root = tmp_path / "owned"
    root.mkdir()
    target = root / "clip.mp4"
    target.write_bytes(b"fixture")
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))

    browser = loopback_webapp.browser()
    _sign_in_password(browser)
    resp = browser.post("/api/files/ticket", headers=OWN_PAGE, json={"route": route, "path": str(target)})
    assert resp.status_code == 200, resp.text
    browser_ticket = resp.json()["ticket"]
    native_ticket = _mint_native_file_ticket(loopback_webapp, monkeypatch, tmp_path / ".hermes", route, str(target))
    service_ticket = _mint_service_file_ticket(loopback_webapp, route, str(target))

    endpoint = tickets.ROUTES[route]
    anon = loopback_webapp.browser()
    assert browser_ticket in tickets._tickets
    assert anon.get(endpoint, params={"path": str(target), "ticket": browser_ticket}).content == b"fixture"

    _sign_out_via_control_verb(loopback_webapp, tmp_path)

    assert browser_ticket not in tickets._tickets
    assert anon.get(endpoint, params={"path": str(target), "ticket": browser_ticket}).status_code == 401
    assert browser.get(endpoint, params={"path": str(target), "ticket": browser_ticket}).status_code == 401

    assert native_ticket in tickets._tickets
    assert anon.get(endpoint, params={"path": str(target), "ticket": native_ticket}).content == b"fixture"
    assert service_ticket in tickets._tickets
    assert anon.get(endpoint, params={"path": str(target), "ticket": service_ticket}).content == b"fixture"


def test_loopback_webapp_expired_cookie_jar_returns_structured_401_without_redirecting_html(loopback_webapp):
    """Once a loopback cookie-mode tab loses its cookies, protected API calls get the structured
    401 envelope with ``login_url`` so the SPA navigates back to ``/login``, while HTML loads stay
    non-redirecting and launch-link tokens keep working."""
    browser = loopback_webapp.browser()
    _sign_in_password(browser)
    assert "window.__HERMES_AUTH_REQUIRED__=true;" in browser.get("/").text

    browser.cookies.clear()

    socket_resp = browser.post("/api/webapp/socket-ticket?socket=gateway", headers=OWN_PAGE)
    assert socket_resp.status_code == 401
    assert socket_resp.json() == {
        "error": "unauthenticated",
        "detail": "Unauthorized",
        "reason": "no_cookie",
        "login_url": "/login",
    }

    profiles_resp = browser.get("/api/profiles")
    assert profiles_resp.status_code == 401
    assert profiles_resp.json()["login_url"] == "/login"
    assert profiles_resp.json()["error"] == "unauthenticated"

    prefixed = browser.get("/api/profiles", headers={"X-Forwarded-Prefix": "/hermes"})
    assert prefixed.status_code == 401
    assert prefixed.json()["login_url"] == "/hermes/login"

    # Public routes and HTML loads stay non-redirecting so launch-link tabs are never trapped.
    assert browser.get("/api/status").status_code == 200
    root = browser.get("/", follow_redirects=False)
    assert root.status_code == 200
    assert "window.__HERMES_AUTH_REQUIRED__=false;" in root.text
    assert loopback_webapp.server._SESSION_TOKEN not in root.text

    # Launch-link token admission remains intact.
    launch_headers = {"X-Hermes-Session-Token": _TOKEN}
    assert browser.get("/api/profiles", headers=launch_headers).status_code == 200
    assert browser.post("/api/webapp/socket-ticket?socket=gateway", headers=launch_headers).status_code == 200


def test_gateway_dashboard_mount_loopback_webapp_login_end_to_end(tmp_path, monkeypatch):
    """Real GatewayDashboardMount on 127.0.0.1 sets withhold_session_token=True, auth_required=False,
    and still honors /auth/password-login cookies on /api/profiles and /api/webapp/socket-ticket."""
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
                assert (await client.get("/api/profiles")).status_code == 401
                login = await client.post(
                    "/auth/password-login",
                    json={"provider": "basic", "username": USER, "password": PASSWORD, "next": "/"},
                )
                assert login.status_code == 200, login.text
                assert (await client.get("/api/profiles")).status_code == 200
                ticket = await client.post("/api/webapp/socket-ticket?socket=gateway", headers=OWN_PAGE)
                assert ticket.status_code == 200, ticket.text
                assert ticket.json()["ticket"]
                root = await client.get("/")
                assert root.status_code == 200
                assert "window.__HERMES_AUTH_REQUIRED__=true;" in root.text
                assert web._SESSION_TOKEN not in root.text
        finally:
            await mount.close()

    try:
        asyncio.run(scenario())
    finally:
        clear_providers()
        webapp_sign_in._reset_for_tests()
        _reset_password_rate_limit()
