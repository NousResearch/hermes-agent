"""A Desktop plugin socket's native grant decides its consent and its scope before plugin code (PLG-1).

The one-use ``native-http`` grant names the profile the socket acts for. The handler used to
redeem it, after the runtime gate had checked consent in the launch profile's ambient scope and
the route dependency had bound the launch profile's secrets: a secondary profile's socket ran a
plugin that profile never enabled (or had disabled), with its home but the launch profile's
secrets. The grant is now redeemed once, ahead of routing; the grant profile's consent holds before
accept, and the handler resolves home and secrets in the grant's profile.

The probe is a bundled plugin: since PLG-3 a secondary profile reaches user plugins only through its
own plugin host (``test_plugin_secondary_profile_host``), so bundled plugins (kanban's socket) are
the in-process sockets a secondary grant still opens. Driven through the real app on the
multiplexed daemon stand-in of the consent-gate tests.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack

import pytest
from starlette.status import WS_1008_POLICY_VIOLATION
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli import plugins as plugins_mod
from hermes_cli import web_server
from hermes_cli.web_server_dashboard import PluginApiWebSocketGate
from hermes_cli.web_server_profiles import _config_profile_scope
from tests.hermes_cli import test_plugin_api_consent_gate as consent_gate

daemon = consent_gate.daemon  # the multiplexed daemon stand-in fixture
_plugins = consent_gate._plugins

_PROBE_API = '''
from pathlib import Path

from fastapi import APIRouter, WebSocket

router = APIRouter()


@router.get("/ping")
async def ping():
    return {"pong": True}


@router.websocket("/events")
async def events(ws: WebSocket):
    with Path(__file__).with_name("ENTERED").open("a") as entered:
        entered.write("x")
    from hermes_cli.web_server_chat import _ws_auth_ok
    if not _ws_auth_ok(ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    from agent.secret_scope import get_secret
    from hermes_constants import get_hermes_home
    while True:
        await ws.send_json({"home": str(get_hermes_home()), "secret": get_secret("PLG_WS_SENTINEL")})
        if await ws.receive_text() == "bye":
            break
    await ws.close()
'''

NAME = "scope-probe"
EVENTS = f"/api/plugins/{NAME}/events"


@pytest.fixture(autouse=True)
def _bundled_root(tmp_path, monkeypatch):
    root = tmp_path / "bundled"
    root.mkdir()
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: root)


def _setup(daemon, *, beta_disabled=False):
    dashboard = plugins_mod.get_bundled_plugins_dir() / NAME / "dashboard"
    dashboard.mkdir(parents=True)
    (dashboard / "manifest.json").write_text(json.dumps({"name": NAME, "api": "plugin_api.py"}))
    (dashboard / "plugin_api.py").write_text(_PROBE_API)
    (dashboard / "index.js").write_text("export {}\n")
    (daemon.launch / ".env").write_text("PLG_WS_SENTINEL=launch-sentinel\n")
    (daemon.beta / ".env").write_text("PLG_WS_SENTINEL=beta-sentinel\n")
    _plugins(daemon.launch)
    _beta(daemon, disabled=beta_disabled)
    assert daemon.client.get(f"/api/plugins/{NAME}/ping", headers=daemon.owner).status_code == 200  # mounts it
    return dashboard / "ENTERED"


def _beta(daemon, *, disabled=False):
    _plugins(daemon.beta, disabled=[NAME] if disabled else [])


def _store():
    return web_server.app.state.gateway_runner.session_ticket_store


def _grant(home):
    return _store().mint(profile_id=str(home), subject="desktop", purpose="native-http")


def _headers(grant):
    return {"X-Hermes-Gateway-Ticket": grant}


def _refused(client, path, grant):
    """True when the upgrade closes with 1008 before accept, False when the socket opens and serves.
    Only entering ``websocket_connect`` may count as a refusal: a socket accepted and then closed
    fails the test."""
    with ExitStack() as stack:
        try:
            ws = stack.enter_context(client.websocket_connect(path, headers=_headers(grant)))
        except WebSocketDisconnect as refusal:
            assert refusal.code == WS_1008_POLICY_VIOLATION
            return True
        ws.receive_json()
        ws.send_text("bye")
    return False


SELECTORS = ["", "?profile=current", "?profile=beta"]


@pytest.mark.parametrize("selector", SELECTORS)
def test_secondary_grant_needs_its_own_profiles_consent(daemon, selector):
    """Disabled in the secondary profile only: its grant is refused before any plugin code runs,
    with the profile selector omitted (the Desktop's shape), current or explicit."""
    entered = _setup(daemon, beta_disabled=True)

    assert _refused(daemon.client, EVENTS + selector, _grant(daemon.beta))
    assert not entered.exists()
    assert not _refused(daemon.client, EVENTS, _grant(daemon.launch))  # the launch profile's own socket opens


def test_runtime_disable_refuses_new_grant_sockets_in_that_profile_only(daemon):
    _setup(daemon)
    assert not _refused(daemon.client, EVENTS, _grant(daemon.beta))

    _beta(daemon, disabled=True)
    assert _refused(daemon.client, EVENTS, _grant(daemon.beta))
    assert not _refused(daemon.client, EVENTS, _grant(daemon.launch))

    _beta(daemon)
    _plugins(daemon.launch, disabled=[NAME])  # PLG-3: the launch profile's lists no longer gate beta
    assert not _refused(daemon.client, EVENTS, _grant(daemon.beta))
    assert _refused(daemon.client, EVENTS, _grant(daemon.launch))


def test_grant_is_redeemed_once(daemon, monkeypatch):
    _setup(daemon)
    store, redeemed = _store(), []
    redeem = store.redeem

    def counting(ticket, **kwargs):
        redeemed.append(ticket)
        return redeem(ticket, **kwargs)

    monkeypatch.setattr(store, "redeem", counting)
    grant = _grant(daemon.beta)
    assert not _refused(daemon.client, EVENTS, grant)
    assert redeemed == [grant]
    assert _refused(daemon.client, EVENTS, grant)  # a second dial with the same grant


@pytest.mark.parametrize("selector", SELECTORS)
def test_handler_resolves_home_and_secrets_in_the_grants_profile(daemon, selector):
    _setup(daemon)
    with daemon.client.websocket_connect(EVENTS + selector, headers=_headers(_grant(daemon.beta))) as ws:
        assert ws.receive_json() == {"home": str(daemon.beta), "secret": "beta-sentinel"}
        ws.send_text("bye")
    with daemon.client.websocket_connect(EVENTS, headers=_headers(_grant(daemon.launch))) as ws:
        assert ws.receive_json() == {"home": str(daemon.launch), "secret": "launch-sentinel"}
        ws.send_text("bye")


def test_concurrent_sockets_keep_their_own_profiles(daemon):
    """Two live sockets for different profiles on ONE event loop never see each other's home or
    secrets, before or after the other one opened."""
    _setup(daemon)

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":  # one shared loop for both sockets, without app startup
            for done in ("lifespan.startup.complete", "lifespan.shutdown.complete"):
                await receive()
                await send({"type": done})
            return
        scope["hermes.gateway_socket_peer"] = ("127.0.0.1", 50000)
        await web_server.app(scope, receive, send)

    beta = {"home": str(daemon.beta), "secret": "beta-sentinel"}
    launch = {"home": str(daemon.launch), "secret": "launch-sentinel"}
    with TestClient(app) as client, \
            client.websocket_connect(EVENTS, headers=_headers(_grant(daemon.beta))) as a:
        assert a.receive_json() == beta
        with client.websocket_connect(EVENTS, headers=_headers(_grant(daemon.launch))) as b:
            assert b.receive_json() == launch
            for ws, expected in ((a, beta), (b, launch), (a, beta)):
                ws.send_text("again")
                assert ws.receive_json() == expected
            b.send_text("bye")
        a.send_text("again")
        assert a.receive_json() == beta
        a.send_text("bye")


def _ambient():
    from agent.secret_scope import get_secret
    from hermes_constants import get_hermes_home
    return str(get_hermes_home()), get_secret("PLG_WS_SENTINEL")


@pytest.mark.parametrize("ending", ["raises", "cancelled"])
def test_grant_scope_is_restored_when_the_connection_ends_abnormally(daemon, ending):
    """The grant's home and secret scope last only as long as its connection: when the plugin
    handler raises, or the connection task is cancelled, the launch profile's are back."""
    _setup(daemon)
    seen = []

    async def handler(scope, receive, send):
        seen.append(_ambient())
        if ending == "raises":
            raise RuntimeError("plugin handler failed")
        await asyncio.Event().wait()  # held open until cancelled

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        raise AssertionError(f"an admitted socket was answered by the gate: {message}")

    scope = {"type": "websocket", "path": EVENTS, "raw_path": EVENTS.encode(), "root_path": "",
             "query_string": b"", "app": web_server.app,
             "headers": [(b"x-hermes-gateway-ticket", _grant(daemon.beta).encode())],
             "hermes.gateway_socket_peer": ("127.0.0.1", 50000)}

    async def connection():
        with _config_profile_scope(None):  # the launch profile's ambient scope
            before = _ambient()
            with pytest.raises(RuntimeError if ending == "raises" else asyncio.CancelledError):
                await PluginApiWebSocketGate(handler)(scope, receive, send)
            return before, _ambient()

    async def run():
        task = asyncio.create_task(connection())
        if ending == "cancelled":
            while not seen:
                await asyncio.sleep(0)
            task.cancel()
        return await task

    launch = (str(daemon.launch), "launch-sentinel")
    assert asyncio.run(run()) == (launch, launch)
    assert seen == [(str(daemon.beta), "beta-sentinel")]
