"""Tests for the paired extension's Webapp tickets.

Covers:
- ``mint_window_ticket`` / ``mint_socket_ticket`` against stand-ins for the Webapp's own modules
  (the ticket stores, the surface policy, the profile lookup), so they run on any checkout.
- Every refusal and its ``error.code``: Webapp absent, off, starting, unavailable, gated or public.
- The two HTTP routes: loopback only (403), the paired token or the API key (401 otherwise, even on a
  server with no key), ``Cache-Control: no-store`` on every reply, and registration by ``connect()``.
- Against the real Webapp modules, when this checkout has them: a window ticket redeems into the browser
  credential exactly once, and a socket ticket is the Webapp's own and dies with a sign-out.
"""
from __future__ import annotations

import importlib.util
import socket
import sys
import types
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from fastapi import HTTPException

from gateway import browser_webapp_tickets as tickets
from gateway.browser_webapp_tickets import WebappTicketRefused, mint_socket_ticket, mint_window_ticket
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

API_KEY = "-".join(("fixture", "neutral", "api", "key", "123"))
WEBAPP_URL = "http://127.0.0.1:9120"
STATE = types.SimpleNamespace(marker="dashboard-app-state", web_dist="/webapp/dist")


def _runner(status=None, *, state=STATE, mounted=True):
    web_module = types.SimpleNamespace(app=types.SimpleNamespace(state=state)) if mounted else None
    return types.SimpleNamespace(
        webapp_status={"state": "serving", "url": WEBAPP_URL, "reason": None} if status is None else status,
        session_api=types.SimpleNamespace(app=types.SimpleNamespace(web=web_module)))


class _Fakes:
    """What the Webapp's modules would answer, and what was minted through them."""

    def __init__(self):
        self.private = True
        self.gateway_ticket = True
        self.bundle_missing = None
        self.window_error = None
        self.profiles = {"default", "work"}
        self.window_calls = []
        self.socket_calls = []


@pytest.fixture
def fakes(monkeypatch):
    fake = _Fakes()

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)

    def mint_handoff(state, ttl, *, launch):
        if fake.window_error is not None:
            raise fake.window_error
        fake.window_calls.append((state, ttl, launch))
        return "w" * 43

    def mint_socket(identity, socket_name, *, session_token=None, sign_in=None, profile_home=None):
        fake.socket_calls.append(
            {"identity": identity, "socket": socket_name, "session_token": session_token,
             "sign_in": sign_in, "profile_home": profile_home})
        return "s" * 43

    def profile_home(profile):
        name = profile or "default"
        if name == "reserved":
            raise HTTPException(400, detail="'reserved' is a reserved name")
        if name not in fake.profiles:
            raise HTTPException(404, detail=f"Profile '{name}' does not exist.")
        return name, Path("/homes") / name

    module("gateway.webapp_listener", bundle_missing_reason=lambda dist: fake.bundle_missing)
    module("hermes_cli.web_server_surface",
           private_launch=lambda state: fake.private,
           policy=lambda state: types.SimpleNamespace(gateway_ticket=fake.gateway_ticket))
    module("hermes_cli.web_routers.webapp",
           WINDOW_TICKET_TTL_SECONDS=30, _mint_handoff_ticket=mint_handoff,
           SESSION_TOKEN_OWNER={"user_id": "legacy-token-owner", "provider": "session-token", "issuer": ""},
           mint_socket_ticket=mint_socket)
    module("hermes_cli.browser_credential", current=lambda: "credential-in-use")
    module("hermes_cli.dashboard_auth.ws_tickets", TTL_SECONDS=30)
    module("hermes_cli.web_server_cron", _cron_profile_home=profile_home)
    module("hermes_cli.web_session_sockets", GATEWAY_SOCKET="gateway")
    return fake


# ---------------------------------------------------------------------------
# Window ticket
# ---------------------------------------------------------------------------


class TestWindowTicket:

    def test_mints_a_one_use_non_launch_window_ticket_and_names_the_webapp(self, fakes):
        body = mint_window_ticket(_runner(), "work")
        assert body == {"ticket": "w" * 43, "webapp_url": WEBAPP_URL, "expires_in_seconds": 30}
        # Minted in this process, on the Webapp's own app state, as a window ticket (launch=False).
        assert fakes.window_calls == [(STATE, 30, False)]

    def test_a_trailing_slash_is_not_part_of_the_webapp_url(self, fakes):
        runner = _runner({"state": "serving", "url": WEBAPP_URL + "/", "reason": None})
        assert mint_window_ticket(runner)["webapp_url"] == WEBAPP_URL

    @pytest.mark.parametrize("profile", ["Bad Name", "../x", "a" * 65, "-x", "UPPER"])
    def test_a_profile_that_is_not_a_name_is_a_400(self, fakes, profile):
        with pytest.raises(WebappTicketRefused) as raised:
            mint_window_ticket(_runner(), profile)
        assert (raised.value.status, raised.value.code) == (400, "invalid_profile")
        assert fakes.window_calls == []

    def test_too_many_pending_windows_is_a_429(self, fakes):
        fakes.window_error = HTTPException(429, "Too many pending Webapp windows")
        with pytest.raises(WebappTicketRefused) as raised:
            mint_window_ticket(_runner())
        assert (raised.value.status, raised.value.code) == (429, "webapp_tickets_exhausted")


# ---------------------------------------------------------------------------
# Socket ticket
# ---------------------------------------------------------------------------


class TestSocketTicket:

    def test_mints_a_gateway_socket_ticket_under_the_browser_credential(self, fakes):
        body = mint_socket_ticket(_runner())
        assert body == {"ticket": "s" * 43, "ws_url": "ws://127.0.0.1:9120/api/ws", "ttl_seconds": 30}
        assert fakes.socket_calls == [{
            "identity": {"user_id": "legacy-token-owner", "provider": "session-token", "issuer": ""},
            "socket": "gateway", "session_token": "credential-in-use", "sign_in": None,
            "profile_home": None}]

    def test_a_ticket_with_a_profile_is_bound_to_that_profile_home(self, fakes):
        mint_socket_ticket(_runner(), "work")
        assert fakes.socket_calls[0]["profile_home"] == str(Path("/homes/work"))

    def test_an_unknown_profile_is_a_404_and_mints_nothing(self, fakes):
        with pytest.raises(WebappTicketRefused) as raised:
            mint_socket_ticket(_runner(), "ghost")
        assert (raised.value.status, raised.value.code) == (404, "profile_not_found")
        assert fakes.socket_calls == []

    def test_a_profile_the_webapp_refuses_is_a_400(self, fakes):
        with pytest.raises(WebappTicketRefused) as raised:
            mint_socket_ticket(_runner(), "reserved")
        assert (raised.value.status, raised.value.code) == (400, "invalid_profile")

    def test_a_surface_that_offers_no_gateway_tickets_is_a_404(self, fakes):
        fakes.gateway_ticket = False
        with pytest.raises(WebappTicketRefused) as raised:
            mint_socket_ticket(_runner())
        assert (raised.value.status, raised.value.code) == (404, "webapp_unavailable")

    def test_the_websocket_url_follows_the_webapp_origin(self, fakes):
        assert tickets._websocket_url("http://[::1]:9120") == "ws://[::1]:9120/api/ws"
        assert tickets._websocket_url("https://example.test/hermes/") == "wss://example.test/hermes/api/ws"


# ---------------------------------------------------------------------------
# Refusals both tickets share
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mint", [mint_window_ticket, mint_socket_ticket])
class TestRefusals:

    def test_no_runner_or_no_webapp_status_is_unavailable(self, fakes, mint):
        for runner in (None, types.SimpleNamespace(), types.SimpleNamespace(webapp_status=None)):
            with pytest.raises(WebappTicketRefused) as raised:
                mint(runner)
            assert (raised.value.status, raised.value.code) == (404, "webapp_unavailable")

    def test_a_webapp_that_is_off_says_how_to_turn_it_on(self, fakes, mint):
        with pytest.raises(WebappTicketRefused) as raised:
            mint(_runner({"state": "off", "url": None, "reason": None}))
        assert (raised.value.status, raised.value.code) == (404, "webapp_off")
        assert "webapp.enabled" in raised.value.message

    def test_a_webapp_that_is_still_starting_is_a_503(self, fakes, mint):
        with pytest.raises(WebappTicketRefused) as raised:
            mint(_runner({"state": "starting", "url": None, "reason": None}))
        assert (raised.value.status, raised.value.code) == (503, "webapp_starting")

    def test_an_unmounted_dashboard_is_still_starting(self, fakes, mint):
        with pytest.raises(WebappTicketRefused) as raised:
            mint(_runner(mounted=False))
        assert (raised.value.status, raised.value.code) == (503, "webapp_starting")

    def test_an_unavailable_webapp_carries_its_reason(self, fakes, mint):
        with pytest.raises(WebappTicketRefused) as raised:
            mint(_runner({"state": "unavailable", "url": None, "reason": "could not listen on 127.0.0.1:9120"}))
        assert (raised.value.status, raised.value.code) == (404, "webapp_unavailable")
        assert "could not listen" in raised.value.message

    def test_a_missing_bundle_is_unavailable(self, fakes, mint):
        fakes.bundle_missing = "The Webapp is not built"
        with pytest.raises(WebappTicketRefused) as raised:
            mint(_runner())
        assert (raised.value.status, raised.value.code) == (404, "webapp_unavailable")
        assert "not built" in raised.value.message

    def test_a_gated_or_public_webapp_is_not_private(self, fakes, mint):
        fakes.private = False
        with pytest.raises(WebappTicketRefused) as raised:
            mint(_runner())
        assert (raised.value.status, raised.value.code) == (404, "webapp_not_private")

    def test_nothing_is_minted_on_a_refusal(self, fakes, mint):
        fakes.private = False
        with pytest.raises(WebappTicketRefused):
            mint(_runner())
        assert fakes.window_calls == [] and fakes.socket_calls == []


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


def _adapter(tmp_path, monkeypatch, *, key=API_KEY, runner=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key} if key else {}))
    adapter.gateway_runner = runner if runner is not None else _runner()
    return adapter


def _app(adapter) -> web.Application:
    app = web.Application()
    for method, path, handler in tickets.http_routes(adapter):
        app.router.add_route(method, path, handler)
    return app


def _paired_token(adapter) -> str:
    store = adapter._browser_pairing
    return store.grant_pairing(store.create_pairing()["pairing_id"])["token"]


WINDOW = "/api/browser-extension/webapp-ticket"
SOCKET = "/api/browser-extension/webapp-socket-ticket"


class TestRoutes:

    def test_the_routes_are_the_two_posts(self, tmp_path, monkeypatch):
        rows = [(method, path) for method, path, _handler in tickets.http_routes(_adapter(tmp_path, monkeypatch))]
        assert rows == [("POST", WINDOW), ("POST", SOCKET)]

    @pytest.mark.asyncio
    async def test_the_paired_token_gets_a_window_ticket(self, fakes, tmp_path, monkeypatch):
        adapter = _adapter(tmp_path, monkeypatch)
        token = _paired_token(adapter)
        async with TestClient(TestServer(_app(adapter))) as client:
            resp = await client.post(WINDOW + "?profile=work", headers={"Authorization": f"Bearer {token}"})
            assert resp.status == 200
            assert await resp.json() == {"ticket": "w" * 43, "webapp_url": WEBAPP_URL, "expires_in_seconds": 30}
            assert resp.headers["Cache-Control"] == "no-store"

    @pytest.mark.asyncio
    async def test_a_paired_token_is_enough_on_a_server_with_no_api_key(self, fakes, tmp_path, monkeypatch):
        adapter = _adapter(tmp_path, monkeypatch, key=None)
        token = _paired_token(adapter)
        async with TestClient(TestServer(_app(adapter))) as client:
            resp = await client.post(SOCKET, headers={"Authorization": f"Bearer {token}"})
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_the_api_key_gets_a_socket_ticket_bound_to_the_profile(self, fakes, tmp_path, monkeypatch):
        adapter = _adapter(tmp_path, monkeypatch)
        async with TestClient(TestServer(_app(adapter))) as client:
            resp = await client.post(SOCKET + "?profile=work", headers={"Authorization": f"Bearer {API_KEY}"})
            assert resp.status == 200
            body = await resp.json()
            assert body["ticket"] == "s" * 43 and body["ws_url"] == "ws://127.0.0.1:9120/api/ws"
            assert fakes.socket_calls[0]["profile_home"] == str(Path("/homes/work"))
            assert resp.headers["Cache-Control"] == "no-store"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", [WINDOW, SOCKET])
    @pytest.mark.parametrize("key", [API_KEY, None])
    async def test_no_credential_or_a_wrong_one_is_a_401_even_without_an_api_key(
            self, fakes, tmp_path, monkeypatch, path, key):
        adapter = _adapter(tmp_path, monkeypatch, key=key)
        async with TestClient(TestServer(_app(adapter))) as client:
            assert (await client.post(path)).status == 401
            assert (await client.post(path, headers={"Authorization": "Bearer " + "x" * 64})).status == 401
        assert fakes.window_calls == [] and fakes.socket_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", [WINDOW, SOCKET])
    async def test_a_peer_off_the_loopback_is_a_403_before_any_credential_is_read(
            self, fakes, tmp_path, monkeypatch, path):
        adapter = _adapter(tmp_path, monkeypatch)
        adapter._request_is_loopback = lambda request: False
        async with TestClient(TestServer(_app(adapter))) as client:
            resp = await client.post(path, headers={"Authorization": f"Bearer {API_KEY}"})
            assert resp.status == 403
            assert (await resp.json())["error"]["code"] == "webapp_ticket_loopback_only"
        assert fakes.window_calls == [] and fakes.socket_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", [WINDOW, SOCKET])
    async def test_a_refusal_keeps_its_status_and_code(self, fakes, tmp_path, monkeypatch, path):
        runner = _runner({"state": "off", "url": None, "reason": None})
        adapter = _adapter(tmp_path, monkeypatch, runner=runner)
        async with TestClient(TestServer(_app(adapter))) as client:
            resp = await client.post(path, headers={"Authorization": f"Bearer {API_KEY}"})
            assert resp.status == 404
            body = await resp.json()
            assert body["error"]["code"] == "webapp_off"
            assert "webapp.enabled" in body["error"]["message"]
            assert resp.headers["Cache-Control"] == "no-store"

    @pytest.mark.asyncio
    async def test_a_bad_profile_is_a_400_with_its_code(self, fakes, tmp_path, monkeypatch):
        adapter = _adapter(tmp_path, monkeypatch)
        async with TestClient(TestServer(_app(adapter))) as client:
            resp = await client.post(SOCKET + "?profile=Bad%20Name", headers={"Authorization": f"Bearer {API_KEY}"})
            assert resp.status == 400
            assert (await resp.json())["error"]["code"] == "invalid_profile"

    @pytest.mark.asyncio
    async def test_the_runner_comes_from_the_app_when_the_adapter_has_none(self, fakes, tmp_path, monkeypatch):
        adapter = _adapter(tmp_path, monkeypatch)
        adapter.gateway_runner = None
        app = _app(adapter)
        app["gateway_runner"] = _runner()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(WINDOW, headers={"Authorization": f"Bearer {API_KEY}"})
            assert resp.status == 200


class TestConnectRegistersTheRoutes:

    @pytest.mark.asyncio
    async def test_a_started_api_server_answers_the_routes(self, fakes, tmp_path, monkeypatch):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = APIServerAdapter(PlatformConfig(
            enabled=True, extra={"host": "127.0.0.1", "port": port, "key": API_KEY}))
        adapter.gateway_runner = _runner()
        assert await adapter.connect() is True
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                for path in (WINDOW, SOCKET):
                    unauth = await session.post(f"http://127.0.0.1:{port}{path}")
                    assert unauth.status == 401
                    ok = await session.post(
                        f"http://127.0.0.1:{port}{path}", headers={"Authorization": f"Bearer {API_KEY}"})
                    assert ok.status == 200
                    assert (await ok.json())["ticket"]
        finally:
            await adapter.disconnect()


# ---------------------------------------------------------------------------
# Against the real Webapp, where this checkout has it
# ---------------------------------------------------------------------------


def _has_webapp() -> bool:
    try:
        return all(importlib.util.find_spec(name) is not None for name in (
            "hermes_cli.web_routers.webapp", "gateway.webapp_listener", "hermes_cli.browser_credential",
            "hermes_cli.web_session_sockets", "hermes_cli.web_server_cron"))
    except (ImportError, ValueError):
        return False


@pytest.mark.skipif(not _has_webapp(), reason="this checkout has no Webapp")
class TestAgainstTheRealWebapp:

    @pytest.fixture
    def webapp(self, tmp_path, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient as FastClient
        from hermes_cli import browser_credential
        from hermes_cli.web_routers import webapp as webapp_routes

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")
        app = FastAPI()
        app.include_router(webapp_routes.router)
        app.state.ui_surface = "webapp"
        app.state.auth_required = False
        app.state.web_dist = dist
        original = browser_credential.current()
        yield types.SimpleNamespace(app=app, client=FastClient(app), credential=browser_credential,
                                    runner=_runner(state=app.state))
        browser_credential.rotate(original)

    def _redeem(self, webapp, ticket):
        return webapp.client.post(
            "/webapp/window-session",
            headers={"Origin": "http://testserver", "X-Hermes-Window-Ticket": ticket})

    def test_the_window_ticket_redeems_once_into_the_browser_credential(self, webapp):
        body = mint_window_ticket(webapp.runner, "work")
        first = self._redeem(webapp, body["ticket"])
        assert first.status_code == 200
        assert first.json() == {"token": webapp.credential.current()}
        assert self._redeem(webapp, body["ticket"]).status_code == 403

    def test_a_signed_out_browser_credential_voids_a_window_ticket(self, webapp):
        body = mint_window_ticket(webapp.runner)
        webapp.credential.rotate("n" * 43)
        assert self._redeem(webapp, body["ticket"]).status_code == 403

    def test_the_socket_ticket_is_the_webapps_own_and_a_sign_out_revokes_it(self, webapp):
        from hermes_cli.dashboard_auth.ws_tickets import consume_ticket
        from hermes_cli.web_session_sockets import GATEWAY_SOCKET, ticket_refusal

        body = mint_socket_ticket(webapp.runner)
        info = consume_ticket(body["ticket"])
        assert info["socket"] == GATEWAY_SOCKET
        assert info["session_token"] == webapp.credential.generation()
        assert "profile_home" not in info
        assert ticket_refusal(info, GATEWAY_SOCKET, auth_required=False) is None
        assert ticket_refusal(info, "host-terminal", auth_required=False) is not None
        webapp.credential.rotate("n" * 43)
        assert ticket_refusal(info, GATEWAY_SOCKET, auth_required=False) is not None

    def test_the_socket_ticket_names_the_profile_home(self, webapp):
        from hermes_cli.dashboard_auth.ws_tickets import consume_ticket
        from hermes_cli.profiles import get_profile_dir

        info = consume_ticket(mint_socket_ticket(webapp.runner, "default")["ticket"])
        assert info["profile_home"] == str(get_profile_dir("default"))

    def test_an_unknown_profile_is_refused(self, webapp):
        with pytest.raises(WebappTicketRefused) as raised:
            mint_socket_ticket(webapp.runner, "ghost")
        assert (raised.value.status, raised.value.code) == (404, "profile_not_found")

    def test_a_gated_webapp_is_refused(self, webapp):
        webapp.app.state.auth_required = True
        for mint in (mint_window_ticket, mint_socket_ticket):
            with pytest.raises(WebappTicketRefused) as raised:
                mint(webapp.runner)
            assert raised.value.code == "webapp_not_private"

    def test_a_missing_bundle_is_refused(self, webapp):
        (webapp.app.state.web_dist / "index.html").unlink()
        with pytest.raises(WebappTicketRefused) as raised:
            mint_window_ticket(webapp.runner)
        assert raised.value.code == "webapp_unavailable"
