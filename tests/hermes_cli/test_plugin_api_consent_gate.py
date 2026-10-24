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
    hosted, calls = [], []

    def asgi_request(host, plugin, dashboard_dir, api_file, method, path, query, headers, body, **_):
        hosted.append(plugin)
        calls.append(SimpleNamespace(home=host._home, dir=dashboard_dir, path=path, query=query,
                                     headers={str(k).lower() for k, _ in headers}))
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
            client=client, launch=launch, beta=beta, hosted=hosted, calls=calls,
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


def test_launch_profile_runtime_disable_closes_its_router_assets_and_tab(daemon):
    """PLG-9 for the launch profile's own copy: disabling it at runtime closes the in-process
    router and the served files, and the plugin list stops offering a tab that cannot load. A
    secondary profile never reaches this router: it runs the root copy in its own host (PLG-3,
    ``test_plugin_secondary_profile_host``)."""
    _install(daemon.launch, "consent-probe")
    _plugins(daemon.launch, enabled=["consent-probe"])
    _plugins(daemon.beta, enabled=["consent-probe"])
    ping, asset = "/api/plugins/consent-probe/ping", "/dashboard-plugins/consent-probe/index.js"
    owner_list = lambda: {p["name"] for p in daemon.client.get("/api/dashboard/plugins", headers=daemon.owner).json()}
    assert daemon.client.get(ping, headers=daemon.owner).status_code == 200
    assert daemon.client.get(asset).status_code == 200
    assert "consent-probe" in owner_list()
    assert daemon.client.get(ping, headers=daemon.beta_ticket()).json() == {"hosted": True}
    assert [c.home for c in daemon.calls] == [daemon.beta]

    _plugins(daemon.launch, enabled=["consent-probe"], disabled=["consent-probe"])

    assert daemon.client.get(ping, headers=daemon.owner).status_code == 404
    assert daemon.client.get(asset).status_code == 404
    assert "consent-probe" not in owner_list()


@pytest.mark.parametrize("isolation", ["in_process", "host"])
def test_secondary_ticket_runs_a_root_plugin_only_it_enabled_in_its_own_host(daemon, isolation):
    """PLG-9 "loses one it enabled", PLG-3: a root plugin (the launch home is the root here) that only
    the secondary profile enabled runs in that profile's plugin host whatever its isolation says;
    it is never imported here."""
    imported = _install(daemon.launch, "beta-only")
    _plugins(daemon.launch)
    _plugins(daemon.beta, enabled=["beta-only"], isolation=isolation)

    assert daemon.client.get("/api/plugins/beta-only/ping", headers=daemon.beta_ticket()).json() == {"hosted": True}
    assert [c.home for c in daemon.calls] == [daemon.beta]
    assert not imported.exists()


def test_hot_mount_keeps_the_launch_profiles_isolation(daemon):
    """The launch profile runs its plugins in a plugin host; a secondary profile's first request for
    a same-named plugin goes to its own host and neither imports nor mounts the launch copy here
    because that profile's own config says in-process."""
    imported = _install(daemon.launch, "iso-probe")
    beta_imported = _install(daemon.beta, "iso-probe")
    _plugins(daemon.launch, enabled=["iso-probe"], isolation="host")
    _plugins(daemon.beta, enabled=["iso-probe"])
    ping = "/api/plugins/iso-probe/ping"

    assert daemon.client.get(ping, headers=daemon.beta_ticket()).json() == {"hosted": True}
    assert daemon.client.get(ping, headers=daemon.owner).json() == {"hosted": True}
    assert [c.home for c in daemon.calls] == [daemon.beta, daemon.launch]
    assert not imported.exists() and not beta_imported.exists()


def test_launch_consent_read_under_a_secondary_grant_keeps_the_launch_secrets(daemon, tmp_path, monkeypatch):
    """A secondary profile's first request for its bundled plugin (a ``?profile=`` selection) binds
    that profile's secrets before the cold mount reads the launch profile's lists. That read must
    expand the launch config's ``${VAR}`` refs with the launch profile's secrets: otherwise the
    launch home's last-good config holds the secondary's values, and the launch profile is served
    them once its config.yaml stops parsing. A root plugin's socket for that profile is refused by
    name (PLG-3): its host serves it."""
    from starlette.testclient import WebSocketDenialResponse

    from hermes_cli import plugins as plugins_mod
    from hermes_cli.config import load_config
    from hermes_cli.web_server_profiles import _config_profile_scope

    monkeypatch.delenv("PLG_SCOPE_PROBE", raising=False)
    _install(tmp_path / "bundled-home", "scope-bundled")
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "bundled-home" / "plugins")
    _install(daemon.launch, "scope-probe")
    config = {"plugins": {"enabled": ["scope-probe"], "disabled": []}, "scope_probe": "${PLG_SCOPE_PROBE}"}
    (daemon.launch / "config.yaml").write_text(json.dumps(config))
    (daemon.launch / ".env").write_text("PLG_SCOPE_PROBE=launch-value\n")
    _plugins(daemon.beta, enabled=["scope-probe"])
    (daemon.beta / ".env").write_text("PLG_SCOPE_PROBE=beta-value\n")

    cold = daemon.client.get("/api/plugins/scope-bundled/ping", params={"profile": "beta"}, headers=daemon.owner)
    assert cold.json() == {"pong": True}
    with daemon.client.websocket_connect("/api/plugins/scope-bundled/events", headers=daemon.beta_ticket()) as ws:
        assert ws.receive_text() == "hello"
    with pytest.raises(WebSocketDenialResponse) as denied, \
            daemon.client.websocket_connect("/api/plugins/scope-probe/events", headers=daemon.beta_ticket()):
        pass
    assert denied.value.status_code == 501 and "plugin host" in denied.value.json()["detail"]
    assert daemon.calls == []

    (daemon.launch / "config.yaml").write_text("plugins: [unclosed\n")

    with _config_profile_scope(None):
        assert load_config()["scope_probe"] == "launch-value"


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
    assert daemon.calls == []
    # The secondary profile does see this root copy (PLG-3): its own opt-out alone refused it, and
    # once it enables the plugin the copy runs in its own host, not in the launch profile's router.
    _plugins(daemon.beta, enabled=["q-probe"])
    assert daemon.client.get(ping, params={"profile": "beta"}, headers=daemon.owner).json() == {"hosted": True}
    assert [c.home for c in daemon.calls] == [daemon.beta]
