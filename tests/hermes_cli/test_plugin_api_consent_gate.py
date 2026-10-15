"""The runtime plugin gate holds on every transport and for every profile a multiplexed daemon serves.

Regressions for #134889:

- PLG-8: ``_plugin_api_runtime_gate`` is HTTP middleware, so a plugin disabled while the server ran
  kept accepting new connections on its websocket routes until restart.
- PLG-9: under a secondary profile's native ticket the gate read only that profile's plugin lists,
  while dashboard discovery only ever finds the launch profile's copy of a plugin: the secondary
  profile kept reaching a plugin the launch profile had disabled, and its first request for a plugin
  only it enabled imported that copy here or handed it to its plugin host.

Driven through the real app (native-ticket auth, the gates, the hot mount, the plugin's own router)
against a temp HERMES_HOME. The daemon is a stand-in runner carrying a real ``TicketStore``; the
plugin host, another process, is a recorder of what this process would ask it to run.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocketDisconnect

from hermes_cli import web_server

_PLUGIN_API = '''
from pathlib import Path

from fastapi import APIRouter, WebSocket

Path(__file__).with_name("IMPORTED").touch()
router = APIRouter()


@router.get("/ping")
async def ping():
    return {"pong": True}


@router.websocket("/events")
async def events(ws: WebSocket):
    # The canonical WS credential check every dashboard plugin socket makes (kanban's too).
    from hermes_cli.web_server_chat import _ws_auth_ok
    if not _ws_auth_ok(ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    await ws.send_text("hello")
    await ws.close()
'''


def _install(home, name):
    dashboard = home / "plugins" / name / "dashboard"
    dashboard.mkdir(parents=True)
    (dashboard / "manifest.json").write_text(json.dumps({"name": name, "api": "plugin_api.py"}))
    (dashboard / "plugin_api.py").write_text(_PLUGIN_API)
    (dashboard / "index.js").write_text("export {}\n")
    return dashboard / "IMPORTED"


def _plugins(home, *, enabled=(), disabled=(), isolation="in_process"):
    (home / "config.yaml").write_text(json.dumps({"plugins": {
        "enabled": list(enabled), "disabled": list(disabled), "isolation": isolation}}))


def listed(daemon):
    """The dashboard plugins the secondary profile's ticket is offered."""
    return {p["name"] for p in daemon.client.get("/api/dashboard/plugins", headers=daemon.beta_ticket()).json()}


@pytest.fixture
def daemon(_isolate_hermes_home, monkeypatch):
    """The dashboard app as a multiplexing gateway serves it: the launch home plus profile ``beta``."""
    from starlette.testclient import TestClient

    import tui_gateway.launch_profile_policy as launch_policy
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from hermes_cli import profiles
    from hermes_cli.plugin_host import PluginHost
    from gateway.runtime_bootstrap import TicketStore
    from gateway.session_authorities import SessionAuthorities
    from hermes_constants import get_hermes_home

    launch = get_hermes_home().resolve()
    beta = launch / "profiles" / "beta"
    beta.mkdir(parents=True)
    authorities = SessionAuthorities(launch, multiplexed=True)
    for home in (launch, beta):
        authorities.add(home, SimpleNamespace(profile_id=str(home), instance_id="daemon"))
    store = TicketStore("daemon", authorities.profile_ids())
    runner = SimpleNamespace(
        session_authorities=authorities, session_ticket_store=store, _draining=False,
        session_runtime_descriptor={"state": "ready", "instance_id": "daemon",
                                    "served_profiles": authorities.served_profiles()})
    monkeypatch.setattr(web_server.app.state, "gateway_runner", runner, raising=False)
    monkeypatch.setattr(launch_policy, "_snapshot", None)
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: launch)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: launch / "profiles")
    hosted = []

    def asgi_request(host, plugin, dashboard_dir, *_request):
        hosted.append(plugin)
        return {"status": 200, "headers": [], "body": b'{"hosted": true}'}

    monkeypatch.setattr(PluginHost, "asgi_request", asgi_request)
    multiplexed, routes, modules = is_multiplex_active(), list(web_server.app.router.routes), set(sys.modules)
    web_server._dashboard_plugins_cache = None

    async def loopback_peer(scope, receive, send):
        # GatewayRuntimeAPI records the socket peer before any proxy middleware; native tickets need loopback.
        scope["hermes.gateway_socket_peer"] = ("127.0.0.1", 50000)
        await web_server.app(scope, receive, send)

    client = TestClient(loopback_peer)
    try:
        yield SimpleNamespace(
            client=client, launch=launch, beta=beta, hosted=hosted,
            owner={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
            beta_ticket=lambda: {"X-Hermes-Gateway-Ticket": store.mint(
                profile_id=str(beta), subject="desktop", purpose="native-http")})
    finally:
        web_server.app.router.routes[:] = routes
        web_server._dashboard_plugins_cache = None
        for name in set(sys.modules) - modules:
            if name.startswith("hermes_dashboard_plugin_"):
                del sys.modules[name]
        set_multiplex_active(multiplexed)


def test_disabled_plugin_refuses_new_websocket_connections(daemon):
    """PLG-8: disabling a plugin at runtime closes its websocket routes to new connections, and an
    unauthenticated caller still cannot tell a disabled plugin from a refused credential."""
    _install(daemon.launch, "socket-probe")
    _plugins(daemon.launch, enabled=["socket-probe"])
    assert daemon.client.get("/api/plugins/socket-probe/ping", headers=daemon.owner).status_code == 200
    events = "/api/plugins/socket-probe/events"
    with daemon.client.websocket_connect(f"{events}?token={web_server._SESSION_TOKEN}") as ws:
        assert ws.receive_text() == "hello"
    with pytest.raises(WebSocketDisconnect) as no_credential, daemon.client.websocket_connect(events):
        pass

    _plugins(daemon.launch, enabled=["socket-probe"], disabled=["socket-probe"])

    with pytest.raises(WebSocketDisconnect) as refused, \
            daemon.client.websocket_connect(f"{events}?token={web_server._SESSION_TOKEN}"):
        pass
    with pytest.raises(WebSocketDisconnect) as anonymous, daemon.client.websocket_connect(events):
        pass
    assert refused.value.code == anonymous.value.code == no_credential.value.code


def test_secondary_ticket_cannot_reach_a_plugin_the_launch_profile_disabled(daemon):
    """PLG-9: the in-process router and the served files are the launch profile's copy; when the
    launch profile disables the plugin at runtime a secondary profile's ticket loses both too,
    whatever that profile's own lists say, and the plugin list stops offering a tab that cannot load."""
    _install(daemon.launch, "consent-probe")
    _plugins(daemon.launch, enabled=["consent-probe"])
    _plugins(daemon.beta, enabled=["consent-probe"])
    ping, asset = "/api/plugins/consent-probe/ping", "/dashboard-plugins/consent-probe/index.js"
    assert daemon.client.get(ping, headers=daemon.owner).status_code == 200
    assert daemon.client.get(ping, headers=daemon.beta_ticket()).status_code == 200
    assert daemon.client.get(asset, headers=daemon.beta_ticket()).status_code == 200
    assert "consent-probe" in listed(daemon)

    _plugins(daemon.launch, enabled=["consent-probe"], disabled=["consent-probe"])

    assert daemon.client.get(ping, headers=daemon.owner).status_code == 404
    assert daemon.client.get(ping, headers=daemon.beta_ticket()).status_code == 404
    assert daemon.client.get(asset, headers=daemon.beta_ticket()).status_code == 404
    assert "consent-probe" not in listed(daemon)


@pytest.mark.parametrize("isolation", ["in_process", "host"])
def test_secondary_ticket_never_runs_a_plugin_only_it_enabled(daemon, isolation):
    """PLG-9 "loses one it enabled": the only copy this server can find is the launch profile's,
    which the launch profile never enabled, so a secondary profile's consent must neither import it
    here nor hand it to that profile's plugin host."""
    imported = _install(daemon.launch, "beta-only")
    _plugins(daemon.launch)
    _plugins(daemon.beta, enabled=["beta-only"], isolation=isolation)

    assert daemon.client.get("/api/plugins/beta-only/ping", headers=daemon.beta_ticket()).status_code == 404
    assert not imported.exists() and daemon.hosted == []


def test_hot_mount_keeps_the_launch_profiles_isolation(daemon):
    """The launch profile runs its plugins in a plugin host; a secondary profile's first request
    must not import the launch copy into this process because that profile's own config says
    in-process."""
    imported = _install(daemon.launch, "iso-probe")
    _plugins(daemon.launch, enabled=["iso-probe"], isolation="host")
    _plugins(daemon.beta, enabled=["iso-probe"])

    assert daemon.client.get("/api/plugins/iso-probe/ping", headers=daemon.beta_ticket()).json() == {"hosted": True}
    assert not imported.exists() and daemon.hosted == ["iso-probe"]


def test_profile_selector_cannot_run_a_plugin_on_a_profile_that_disabled_it(daemon):
    """``?profile=`` with the session token binds the selected profile's scope for the handler, so
    that profile's opt-out holds there exactly as it does for its own ticket, HTTP and WebSocket."""
    _install(daemon.launch, "q-probe")
    _plugins(daemon.launch, enabled=["q-probe"])
    _plugins(daemon.beta, enabled=["q-probe"], disabled=["q-probe"])
    ping = "/api/plugins/q-probe/ping"
    assert daemon.client.get(ping, headers=daemon.owner).status_code == 200

    assert daemon.client.get(ping, params={"profile": "beta"}, headers=daemon.owner).status_code == 404
    events = f"/api/plugins/q-probe/events?token={web_server._SESSION_TOKEN}"
    with daemon.client.websocket_connect(events) as ws:
        assert ws.receive_text() == "hello"
    with pytest.raises(WebSocketDisconnect), daemon.client.websocket_connect(f"{events}&profile=beta"):
        pass
