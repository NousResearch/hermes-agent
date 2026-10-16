"""PLG-3: a secondary served profile's dashboard plugins run in that profile's own plugin host.

On a gateway serving several profiles, dashboard plugins were discovered from the launch home only
and mounted once in the gateway process: a plugin installed only in a secondary profile answered 404
from that profile's Desktop window, and its ``plugins.isolation`` was ignored. Now a request scoped to
a secondary profile (native grant, ``?profile=``) reaches that profile's own copy through its own
plugin host, whatever its isolation setting says; its assets and plugin list are its own; the launch
profile's consent no longer gates them; and its plugin code is never imported here.

Driven through the real app on the multiplexed daemon stand-in of the consent-gate tests. The host is
a recorder there; the last test runs a real one.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qsl, urljoin

import pytest
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocketDisconnect

from hermes_cli import plugins as plugins_mod
from hermes_cli import web_server
from hermes_cli.plugin_host import PluginHost
from tests.hermes_cli import test_plugin_api_consent_gate as consent_gate

daemon = consent_gate.daemon  # the multiplexed daemon stand-in fixture
_install, _plugins, listed = consent_gate._install, consent_gate._plugins, consent_gate.listed

_REAL_ASGI_REQUEST = PluginHost.asgi_request
NAME = "beta-only"
PING, EVENTS, ASSET = (f"/api/plugins/{NAME}/ping", f"/api/plugins/{NAME}/events",
                       f"/dashboard-plugins/{NAME}/index.js")


def _beta_plugin(daemon, *, isolation="in_process"):
    imported = _install(daemon.beta, NAME)
    _plugins(daemon.beta, enabled=[NAME], isolation=isolation)
    _plugins(daemon.launch)
    return imported


def _never_imported(imported) -> bool:
    return not imported.exists() and f"hermes_dashboard_plugin_{NAME}" not in sys.modules


@pytest.mark.parametrize("isolation", ["in_process", "host"])
def test_secondary_profiles_own_plugin_answers_through_its_own_host(daemon, isolation):
    imported = _beta_plugin(daemon, isolation=isolation)

    response = daemon.client.get(PING, headers=daemon.beta_ticket())

    assert (response.status_code, response.json()) == (200, {"hosted": True})
    (call,) = daemon.calls
    assert call.home == daemon.beta
    assert Path(call.dir) == daemon.beta / "plugins" / NAME / "dashboard"
    assert call.path == "/ping"
    assert _never_imported(imported)


def test_profile_selector_reaches_the_selected_profiles_host(daemon):
    imported = _beta_plugin(daemon)

    response = daemon.client.get(PING, params={"profile": "beta"}, headers=daemon.owner)

    assert response.json() == {"hosted": True}
    assert [c.home for c in daemon.calls] == [daemon.beta]
    assert _never_imported(imported)


def test_launch_profile_cannot_reach_a_secondary_profiles_plugin(daemon):
    imported = _beta_plugin(daemon)

    assert daemon.client.get(PING, headers=daemon.owner).status_code == 404
    assert daemon.client.get(ASSET).status_code == 404
    assert NAME not in {p["name"] for p in daemon.client.get("/api/dashboard/plugins", headers=daemon.owner).json()}
    assert daemon.calls == [] and _never_imported(imported)


def test_assets_and_plugin_list_are_the_secondary_profiles_own(daemon):
    _beta_plugin(daemon)

    asset = daemon.client.get(ASSET, headers=daemon.beta_ticket())
    assert (asset.status_code, asset.text) == (200, "export {}\n")
    assert NAME in listed(daemon)


def test_browser_assets_follow_the_selected_profile(daemon):
    """The browser loads assets with the ``?profile=`` its plugin list used: same-named plugins
    answer with that profile's own files, a secondary-only plugin loads, and that profile's consent,
    the traversal check and the suffix allowlist hold."""
    for home in (daemon.launch, daemon.beta):
        _install(home, "same")
        (home / "plugins" / "same" / "dashboard" / "index.js").write_text(home.name)
    _install(daemon.beta, NAME)
    _plugins(daemon.launch, enabled=["same"])
    _plugins(daemon.beta, enabled=["same", NAME])

    def asset(name, file="index.js", **params):
        return daemon.client.get(f"/dashboard-plugins/{name}/{file}", params=params)

    assert asset("same").text == daemon.launch.name
    assert asset("same", profile="beta").text == "beta"
    assert asset(NAME, profile="beta").status_code == 200
    assert asset(NAME).status_code == 404
    assert asset(NAME, "plugin_api.py", profile="beta").status_code == 404
    assert asset(NAME, "..%2F..%2F..%2Fconfig.yaml", profile="beta").status_code == 403

    _plugins(daemon.beta, enabled=["same", NAME], disabled=["same"])
    assert asset("same", profile="beta").status_code == 404
    assert asset("same").text == daemon.launch.name
    unknown = asset("no-such-plugin", profile="beta")
    assert (asset(NAME, profile="nobody").status_code, asset(NAME, profile="nobody").json()) == (
        unknown.status_code, unknown.json()) == (404, {"detail": "Plugin not found"})


def test_profile_bearing_asset_path_keeps_relative_urls_in_that_profile(daemon):
    """A relative URL in a stylesheet or bundle (``url("./icon.svg")``, a chunk import) drops the
    sheet's query string, so the loader names the profile in the path: what it resolves to is still
    that profile's copy, under the same consent, traversal and suffix checks."""
    for home in (daemon.launch, daemon.beta):
        _install(home, "same")
        (home / "plugins" / "same" / "dashboard" / "icon.svg").write_text(home.name)
    _install(daemon.beta, NAME)
    _plugins(daemon.launch, enabled=["same"])
    _plugins(daemon.beta, enabled=["same", NAME])
    get = daemon.client.get
    icon = urljoin("/dashboard-plugins/@beta/same/style.css", "./icon.svg")

    assert (get(icon).status_code, get(icon).text) == (200, "beta")
    assert get(f"/dashboard-plugins/@beta/{NAME}/index.js").status_code == 200
    assert get("/dashboard-plugins/same/icon.svg").text == daemon.launch.name
    assert get(f"/dashboard-plugins/@beta/{NAME}/plugin_api.py").status_code == 404
    assert get(f"/dashboard-plugins/@beta/{NAME}/..%2F..%2F..%2Fconfig.yaml").status_code == 403
    assert get(f"/dashboard-plugins/@nobody/{NAME}/index.js").json() == {"detail": "Plugin not found"}
    _plugins(daemon.beta, enabled=["same", NAME], disabled=["same"])
    assert get(icon).status_code == 404


def test_launch_profile_consent_no_longer_gates_a_secondary_profiles_copy(daemon):
    launch_imported = _install(daemon.launch, NAME)
    beta_imported = _install(daemon.beta, NAME)
    _plugins(daemon.launch, disabled=[NAME])
    _plugins(daemon.beta, enabled=[NAME])

    assert daemon.client.get(PING, headers=daemon.beta_ticket()).json() == {"hosted": True}
    assert [Path(c.dir) for c in daemon.calls] == [daemon.beta / "plugins" / NAME / "dashboard"]
    assert NAME in listed(daemon)
    assert daemon.client.get(PING, headers=daemon.owner).status_code == 404

    _plugins(daemon.beta, enabled=[NAME], disabled=[NAME])  # the secondary profile's own opt-out holds
    assert daemon.client.get(PING, headers=daemon.beta_ticket()).status_code == 404
    assert NAME not in listed(daemon)
    assert len(daemon.calls) == 1
    assert not launch_imported.exists() and not beta_imported.exists()


def test_gateway_credentials_never_reach_a_secondary_profiles_host(daemon):
    """The owner token and native grants are valid for every served profile: a profile's plugin code
    must not receive them."""
    _beta_plugin(daemon)
    daemon.client.get(PING, headers=daemon.beta_ticket())
    daemon.client.get(PING, params={"profile": "beta", "keep": "1"},
                      headers={**daemon.owner, "Authorization": "Bearer x", "Cookie": "a=b"})

    assert len(daemon.calls) == 2
    forbidden = {"x-hermes-gateway-ticket", web_server._SESSION_HEADER_NAME.lower(), "authorization", "cookie"}
    for call in daemon.calls:
        assert not call.headers & forbidden
    assert "keep=1" in daemon.calls[1].query


@pytest.mark.parametrize("selector", [False, True])
def test_gateway_query_credentials_never_reach_a_secondary_profiles_host(daemon, selector):
    """Every query credential the gateway accepts (``token``, ``ticket``, ``internal``,
    ``display_ticket``, ``attach``) and the profile selector are dropped, repeated and URL-encoded
    keys too; the plugin's own parameters pass through unchanged."""
    _beta_plugin(daemon)
    query = ("token=a&ticket=b&internal=c&display_ticket=d&attach=e&%74oken=f&ticket=g&tic%6Bet=h"
             "&keep=1&keep=2&q=x%20y&tokenish=ok")
    if selector:
        response = daemon.client.get(f"{PING}?{query}&profile=beta", headers=daemon.owner)
    else:
        response = daemon.client.get(f"{PING}?{query}", headers=daemon.beta_ticket())

    assert response.json() == {"hosted": True}
    (call,) = daemon.calls
    assert parse_qsl(call.query, keep_blank_values=True) == [
        ("keep", "1"), ("keep", "2"), ("q", "x y"), ("tokenish", "ok")]


def test_secondary_plugin_socket_gets_a_named_refusal(daemon):
    """The host protocol carries no websockets yet: an admitted socket is refused by name (HTTP 501,
    before accept), never run in-process. An unauthenticated dial gets the plain policy close."""
    imported = _beta_plugin(daemon)

    with pytest.raises(WebSocketDenialResponse) as denied, \
            daemon.client.websocket_connect(EVENTS, headers=daemon.beta_ticket()):
        pass
    assert denied.value.status_code == 501
    assert "plugin host" in denied.value.json()["detail"]

    with pytest.raises(WebSocketDisconnect) as anonymous, \
            daemon.client.websocket_connect(f"{EVENTS}?profile=beta"):
        pass
    assert not isinstance(anonymous.value, WebSocketDenialResponse) and anonymous.value.code == 1008
    assert daemon.calls == [] and _never_imported(imported)


def test_root_plugins_stay_available_to_a_secondary_profile_in_its_own_host(daemon):
    """As a standalone ``hermes --profile beta serve`` did: beta scans its own ``plugins/``, then
    the root's, then the bundled ones; its own copy wins a same-name conflict; its own lists decide;
    the root copy runs in beta's host, never here. (The launch home is the root in this fixture.)"""
    root_imported = _install(daemon.launch, NAME)
    _plugins(daemon.launch)  # the launch profile never enabled it
    _plugins(daemon.beta, enabled=[NAME])

    assert daemon.client.get(PING, headers=daemon.beta_ticket()).json() == {"hosted": True}
    assert NAME in listed(daemon)
    assert daemon.client.get(ASSET, headers=daemon.beta_ticket()).status_code == 200
    assert daemon.client.get(PING, headers=daemon.owner).status_code == 404

    beta_imported = _install(daemon.beta, NAME)
    daemon.client.get("/api/dashboard/plugins/rescan", headers=daemon.beta_ticket())
    assert daemon.client.get(PING, headers=daemon.beta_ticket()).json() == {"hosted": True}
    assert [(c.home, Path(c.dir)) for c in daemon.calls] == [
        (daemon.beta, daemon.launch / "plugins" / NAME / "dashboard"),
        (daemon.beta, daemon.beta / "plugins" / NAME / "dashboard")]
    assert not root_imported.exists() and not beta_imported.exists()


def test_selected_profile_rescan_refreshes_that_profiles_cache_only(daemon):
    """The browser's rescan names its profile with ``?profile=`` beside the owner token, so no native
    grant binds the scope first: it still rescans and counts the selected profile's plugins, and the
    launch profile's cache is left as it was."""
    from hermes_cli.web_server_plugin_profiles import profile_dashboard_plugins
    _install(daemon.beta, NAME)
    _plugins(daemon.launch)
    _plugins(daemon.beta, enabled=[NAME, "later"])
    selected = {"params": {"profile": "beta"}, "headers": daemon.owner}
    names = lambda **kw: {p["name"] for p in daemon.client.get("/api/dashboard/plugins", **kw).json()}
    warmed, launch_listed = names(**selected), names(headers=daemon.owner)
    assert NAME in warmed and NAME not in launch_listed
    launch_cache = web_server._dashboard_plugins_cache
    _install(daemon.beta, "later")
    _install(daemon.launch, "launch-later")

    rescan = daemon.client.get("/api/dashboard/plugins/rescan", **selected)

    assert names(**selected) == warmed | {"later"}
    assert rescan.json() == {"ok": True, "count": len(profile_dashboard_plugins(daemon.beta))}
    assert web_server._dashboard_plugins_cache is launch_cache
    assert names(headers=daemon.owner) == launch_listed


def _refusal(daemon, path, headers):
    """How a socket dial ends; any accept fails the test (refusals happen before accept)."""
    with pytest.raises(WebSocketDisconnect) as closed, daemon.client.websocket_connect(path, headers=headers):
        pytest.fail(f"{path} was accepted")
    if isinstance(closed.value, WebSocketDenialResponse):
        return "denied", closed.value.status_code
    return "closed", closed.value.code


@pytest.mark.parametrize("selector", [False, True])
def test_secondary_consent_and_refusal_matrix(daemon, selector):
    """Enabled, disabled, not enabled and unknown, over HTTP and websockets, for beta's ticket and for
    the owner token with ``?profile=beta``: only an enabled plugin reaches beta's host, every refusal
    comes before accept, and disabled, not enabled and unknown look the same."""
    imported = _install(daemon.beta, NAME)
    _plugins(daemon.launch)
    token = f"token={web_server._SESSION_TOKEN}&profile=beta"

    def outcome(name):
        ping, events = f"/api/plugins/{name}/ping", f"/api/plugins/{name}/events"
        if selector:
            http = daemon.client.get(f"{ping}?profile=beta", headers=daemon.owner)
            ws = _refusal(daemon, f"{events}?{token}", {})
        else:
            http = daemon.client.get(ping, headers=daemon.beta_ticket())
            ws = _refusal(daemon, events, daemon.beta_ticket())
        anonymous = daemon.client.get(f"{ping}?profile=beta"), _refusal(daemon, f"{events}?profile=beta", {})
        return (http.status_code, http.json()), ws, (anonymous[0].status_code, anonymous[1])

    outcomes = {}
    for state, lists in {"enabled": {"enabled": [NAME]}, "disabled": {"enabled": [NAME], "disabled": [NAME]},
                         "not enabled": {}}.items():
        _plugins(daemon.beta, **lists)
        outcomes[state] = outcome(NAME)
    outcomes["unknown"] = outcome("no-such-plugin")

    enabled_http, enabled_ws, enabled_anonymous = outcomes.pop("enabled")
    assert enabled_http == (200, {"hosted": True}) and enabled_ws == ("denied", 501)
    refused = ((404, {"detail": "Plugin not found"}), ("closed", 1008), enabled_anonymous)
    assert outcomes == dict.fromkeys(outcomes, refused)
    assert enabled_anonymous[1] == ("closed", 1008)
    assert [c.home for c in daemon.calls] == [daemon.beta]
    assert _never_imported(imported)


def _bundled(tmp_path, monkeypatch, name):
    """A bundled plugin ``name``; returns its dashboard dir."""
    _install(tmp_path / "bundled-home", name)
    (tmp_path / "bundled-home" / "plugins").rename(tmp_path / "bundled")
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "bundled")
    return tmp_path / "bundled" / name / "dashboard"


def _as_beta(daemon, selector):
    """``(http kwargs, websocket url suffix, websocket headers)`` for beta's ticket or the owner token
    with ``?profile=beta``."""
    if selector:
        return ({"params": {"profile": "beta"}, "headers": daemon.owner},
                f"?token={web_server._SESSION_TOKEN}&profile=beta", {})
    return {"headers": daemon.beta_ticket()}, "", daemon.beta_ticket()


@pytest.mark.parametrize("selector", [False, True], ids=["ticket", "selector"])
@pytest.mark.parametrize("mounted", [True, False], ids=["mounted", "cold"])
def test_launch_user_copy_shadowing_a_bundled_name_never_serves_a_secondary_profile(
        daemon, tmp_path, monkeypatch, mounted, selector):
    """The router this process runs for a bundled name the launch profile overrides is the launch
    profile's user code. A secondary profile whose own entry is the bundled copy gets that copy from
    its own host (HTTP) or a named refusal (websocket), never the launch profile's code, whether that
    router is mounted already or would be mounted by this request."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: tmp_path / "root")  # no root copy
    bundled = _bundled(tmp_path, monkeypatch, "shadowed")
    launch_imported = _install(daemon.launch, "shadowed")
    _plugins(daemon.launch, enabled=["shadowed"])
    _plugins(daemon.beta)
    ping = "/api/plugins/shadowed/ping"
    if mounted:
        assert daemon.client.get(ping, headers=daemon.owner).json() == {"pong": True}
    http, suffix, ws_headers = _as_beta(daemon, selector)

    assert daemon.client.get(ping, **http).json() == {"hosted": True}
    assert _refusal(daemon, f"/api/plugins/shadowed/events{suffix}", ws_headers) == ("denied", 501)
    assert [(c.home, Path(c.dir)) for c in daemon.calls] == [(daemon.beta, bundled)]
    assert launch_imported.exists() is mounted and not (bundled / "IMPORTED").exists()


@pytest.mark.parametrize("selector", [False, True], ids=["ticket", "selector"])
def test_disabled_launch_override_does_not_hide_the_bundled_copy_from_a_secondary_profile(
        daemon, tmp_path, monkeypatch, selector):
    """The launch profile disabled its own override of a bundled name, so this process has no router a
    secondary profile may use for it: the secondary profile's bundled copy runs in its own host."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: tmp_path / "root")
    bundled = _bundled(tmp_path, monkeypatch, "hidden")
    launch_imported = _install(daemon.launch, "hidden")
    _plugins(daemon.launch, enabled=["hidden"], disabled=["hidden"])
    _plugins(daemon.beta)
    http, suffix, ws_headers = _as_beta(daemon, selector)

    assert daemon.client.get("/api/plugins/hidden/ping", **http).json() == {"hosted": True}
    assert _refusal(daemon, f"/api/plugins/hidden/events{suffix}", ws_headers) == ("denied", 501)
    assert [(c.home, Path(c.dir)) for c in daemon.calls] == [(daemon.beta, bundled)]
    assert not launch_imported.exists() and not (bundled / "IMPORTED").exists()


@pytest.mark.parametrize("selector", [False, True], ids=["ticket", "selector"])
def test_bundled_plugin_the_launch_profile_disabled_still_serves_a_secondary_profile(
        daemon, tmp_path, monkeypatch, selector):
    """Bundled plugins are Hermes code and stay in-process; the secondary profile's own lists decide,
    also for the first request that mounts the router, whether its grant or ``?profile=`` names it."""
    _bundled(tmp_path, monkeypatch, "bundled-probe")
    _plugins(daemon.launch, disabled=["bundled-probe"])
    _plugins(daemon.beta)
    ping = "/api/plugins/bundled-probe/ping"
    http, _, _ = _as_beta(daemon, selector)

    assert daemon.client.get(ping, **http).json() == {"pong": True}
    assert daemon.client.get(ping, headers=daemon.owner).status_code == 404
    assert daemon.calls == []


@pytest.mark.parametrize("selector", [False, True], ids=["ticket", "selector"])
def test_secondary_profiles_opt_out_holds_on_a_cold_bundled_mount(daemon, tmp_path, monkeypatch, selector):
    bundled = _bundled(tmp_path, monkeypatch, "bundled-optout")
    _plugins(daemon.launch)
    _plugins(daemon.beta, disabled=["bundled-optout"])
    ping = "/api/plugins/bundled-optout/ping"
    http, _, _ = _as_beta(daemon, selector)

    assert daemon.client.get(ping, **http).status_code == 404
    assert not (bundled / "IMPORTED").exists()
    assert daemon.client.get(ping, headers=daemon.owner).json() == {"pong": True}


_HOME_API = '''
import os
from fastapi import APIRouter

router = APIRouter()


@router.get("/whoami")
async def whoami():
    from hermes_constants import get_hermes_home
    return {"pid": os.getpid(), "home": str(get_hermes_home())}
'''


@pytest.mark.platforms("posix")  # the host is a real child process
def test_secondary_plugin_runs_in_a_real_host_bound_to_its_home(daemon, monkeypatch):
    monkeypatch.setattr(PluginHost, "asgi_request", _REAL_ASGI_REQUEST)
    imported = _beta_plugin(daemon)
    (daemon.beta / "plugins" / NAME / "dashboard" / "plugin_api.py").write_text(_HOME_API)
    try:
        response = daemon.client.get(f"/api/plugins/{NAME}/whoami", headers=daemon.beta_ticket())
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["pid"] != os.getpid()
        assert Path(body["home"]).resolve() == daemon.beta.resolve()
        assert _never_imported(imported)
    finally:
        _shutdown_hosts()


def _shutdown_hosts():
    for manager in list(plugins_mod._plugin_managers_by_home.values()):
        host = getattr(manager, "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()


def _asgi_client():
    """An async client on the real app, so one test can hold requests in flight side by side."""
    import httpx

    async def loopback_peer(scope, receive, send):
        scope["hermes.gateway_socket_peer"] = ("127.0.0.1", 50000)
        await web_server.app(scope, receive, send)

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=loopback_peer), base_url="http://testserver")


def test_slow_hosted_requests_do_not_starve_the_gateways_shared_threads(daemon, monkeypatch):
    """A slow secondary-profile plugin ties up threads of its own host's forwarding only: with every
    hosted request still in flight, a ``to_thread`` route (the plugin list) answers."""
    _beta_plugin(daemon)
    release, selected = threading.Event(), {"params": {"profile": "beta"}, "headers": daemon.owner}

    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(2))  # the gateway's shared pool, kept small
        started, two_started = [], asyncio.Event()

        def slow(host, *args, **kwargs):
            loop.call_soon_threadsafe(lambda: started.append(1) or (len(started) >= 2 and two_started.set()))
            release.wait(30)
            return {"status": 200, "headers": [], "body": b'{"hosted": true}'}

        monkeypatch.setattr(PluginHost, "asgi_request", slow)
        async with _asgi_client() as client:
            hosted = [asyncio.ensure_future(client.get(PING, **selected)) for _ in range(4)]
            try:
                await asyncio.wait_for(two_started.wait(), 30)
                try:
                    listed = await asyncio.wait_for(client.get("/api/dashboard/plugins", **selected), 10)
                except TimeoutError:
                    listed = None
                in_flight = not any(t.done() for t in hosted)
            finally:
                release.set()
            answers = await asyncio.gather(*hosted)
        return listed, in_flight, [a.status_code for a in answers]

    listed, in_flight, answers = asyncio.run(scenario())
    assert listed is not None, "the plugin list waited behind slow hosted plugin requests"
    assert listed.status_code == 200 and NAME in {p["name"] for p in listed.json()}
    assert in_flight and answers == [200] * 4


_HANG_API = _HOME_API + '''

@router.get("/hang")
async def hang():
    import asyncio
    await asyncio.Event().wait()
'''


@pytest.mark.platforms("posix")  # the host is a real child process
def test_hung_hosted_request_answers_504_and_the_host_serves_the_next(daemon, monkeypatch):
    from hermes_cli import web_server_plugin_profiles
    monkeypatch.setattr(PluginHost, "asgi_request", _REAL_ASGI_REQUEST)
    monkeypatch.setattr(web_server_plugin_profiles, "HOSTED_REQUEST_TIMEOUT_SECS", 1.0, raising=False)
    _beta_plugin(daemon)
    (daemon.beta / "plugins" / NAME / "dashboard" / "plugin_api.py").write_text(_HANG_API)
    selected = {"params": {"profile": "beta"}, "headers": daemon.owner}
    whoami, hang = f"/api/plugins/{NAME}/whoami", f"/api/plugins/{NAME}/hang"

    async def scenario():
        async with _asgi_client() as client:
            try:
                before = await client.get(whoami, **selected)
                try:
                    hung = await asyncio.wait_for(client.get(hang, **selected), 20)
                except TimeoutError:
                    hung = None
                after = await client.get(whoami, **selected)
            finally:
                _shutdown_hosts()  # also frees a request still parked on the host
        return before, hung, after

    before, hung, after = asyncio.run(scenario())
    assert hung is not None, "a hung plugin request never answered"
    assert hung.status_code == 504 and NAME in hung.json()["detail"]
    assert (before.status_code, after.status_code) == (200, 200)
    assert after.json()["pid"] == before.json()["pid"]  # the same host, not a restarted one


@pytest.mark.platforms("posix")  # the hosts are real child processes
def test_concurrent_first_requests_start_one_host_and_share_it(daemon, monkeypatch):
    """A secondary profile's first plugin requests, arriving together, start one plugin host and all
    reach it: each racing request used to construct and start a host of its own."""
    monkeypatch.setattr(PluginHost, "asgi_request", _REAL_ASGI_REQUEST)
    _beta_plugin(daemon)
    (daemon.beta / "plugins" / NAME / "dashboard" / "plugin_api.py").write_text(_HOME_API)
    selected = {"params": {"profile": "beta"}, "headers": daemon.owner}
    racers, entered, all_entered, built = 4, [], threading.Event(), []
    lookup, build = plugins_mod.PluginManager._plugin_host, PluginHost._build_base_context

    def counted_lookup(manager):
        entered.append(1)
        if len(entered) >= racers:
            all_entered.set()
        return lookup(manager)

    def held_build(host):  # the first host is built only once every request is looking it up
        built.append(host)
        all_entered.wait(30)
        return build(host)

    monkeypatch.setattr(plugins_mod.PluginManager, "_plugin_host", counted_lookup)
    monkeypatch.setattr(PluginHost, "_build_base_context", held_build)

    async def scenario():
        async with _asgi_client() as client:
            return await asyncio.gather(*(client.get(f"/api/plugins/{NAME}/whoami", **selected)
                                          for _ in range(racers)))

    try:
        answers = asyncio.run(scenario())
    finally:
        for host in built:
            host.shutdown()
    assert [a.status_code for a in answers] == [200] * racers
    assert len(built) == 1, f"{len(built)} plugin hosts were built for one profile"
    assert len({a.json()["pid"] for a in answers}) == 1


def test_a_host_that_failed_to_build_is_built_again_by_the_next_request(daemon, monkeypatch):
    _beta_plugin(daemon)
    build, failures = PluginHost._build_base_context, [RuntimeError("profile secrets unreadable")]

    def flaky(host):
        if failures:
            raise failures.pop()
        return build(host)

    monkeypatch.setattr(PluginHost, "_build_base_context", flaky)
    selected = {"params": {"profile": "beta"}, "headers": daemon.owner}

    assert daemon.client.get(PING, **selected).status_code == 502
    assert daemon.client.get(PING, **selected).json() == {"hosted": True}


def test_launch_profiles_hosted_mount_answers_504_too(daemon, monkeypatch):
    """``plugins.isolation: host`` for the launch profile forwards through the same path."""
    _install(daemon.launch, "launch-hosted")
    _plugins(daemon.launch, enabled=["launch-hosted"], isolation="host")
    calls = []

    def unanswered(host, *args, timeout=None, **kwargs):
        calls.append((host._home, timeout))
        if timeout is None:  # no deadline: the gateway would wait on the host for good
            return {"status": 200, "headers": [], "body": b"{}"}
        raise TimeoutError  # what the channel raises when the deadline passes without a reply

    monkeypatch.setattr(PluginHost, "asgi_request", unanswered)
    response = daemon.client.get("/api/plugins/launch-hosted/ping", headers=daemon.owner)

    assert response.status_code == 504
    ((home, timeout),) = calls
    assert home == daemon.launch and 0 < timeout <= 60


def test_unknown_plugin_names_do_not_rescan_the_profile_on_every_request(daemon, monkeypatch):
    """A 404 probe rescans a secondary profile's plugin dirs once per short window, not on every
    request; an explicit rescan, or the end of the window, still finds a plugin installed meanwhile."""
    from hermes_cli import web_server_dashboard, web_server_plugin_profiles
    _beta_plugin(daemon)
    _plugins(daemon.beta, enabled=[NAME, "later", "latest"])
    selected = {"params": {"profile": "beta"}, "headers": daemon.owner}
    status = lambda name: daemon.client.get(f"/api/plugins/{name}/ping", **selected).status_code
    assert status(NAME) == 200
    scans, discover, clock = [], web_server_dashboard._discover_dashboard_plugins, [1000.0]
    monkeypatch.setattr(web_server_dashboard, "_discover_dashboard_plugins",
                        lambda *a, **k: scans.append(1) or discover(*a, **k))
    monkeypatch.setattr(web_server_plugin_profiles, "time", SimpleNamespace(monotonic=lambda: clock[0]),
                        raising=False)

    assert [status("later") for _ in range(3)] == [404] * 3
    assert len(scans) == 1
    _install(daemon.beta, "later")
    daemon.client.get("/api/dashboard/plugins/rescan", **selected)
    assert status("later") == 200

    assert status("latest") == 404
    _install(daemon.beta, "latest")
    clock[0] += 60
    assert status("latest") == 200


def test_alternating_unknown_names_each_rescan_once_per_window(daemon, monkeypatch):
    """One name's miss rescan keeps another name's cached miss, so probes alternating two absent
    names scan once per name per window; an explicit rescan, or expiry, finds a plugin installed meanwhile."""
    from hermes_cli import web_server_dashboard, web_server_plugin_profiles
    _beta_plugin(daemon)
    _plugins(daemon.beta, enabled=[NAME, "absent-a", "absent-b"])
    selected = {"params": {"profile": "beta"}, "headers": daemon.owner}
    status = lambda name: daemon.client.get(f"/api/plugins/{name}/ping", **selected).status_code
    assert status(NAME) == 200
    scans, discover, clock = [], web_server_dashboard._discover_dashboard_plugins, [1000.0]
    monkeypatch.setattr(web_server_dashboard, "_discover_dashboard_plugins",
                        lambda *a, **k: scans.append(1) or discover(*a, **k))
    monkeypatch.setattr(web_server_plugin_profiles, "time", SimpleNamespace(monotonic=lambda: clock[0]),
                        raising=False)

    assert [status(n) for n in ["absent-a", "absent-b"] * 3] == [404] * 6
    assert len(scans) == 2, "an automatic miss rescan dropped the other name's cached miss"

    _install(daemon.beta, "absent-b")
    daemon.client.get("/api/dashboard/plugins/rescan", **selected)
    assert status("absent-b") == 200

    assert status("absent-a") == 404  # the explicit rescan cleared its miss: this one scans again
    _install(daemon.beta, "absent-a")
    assert status("absent-a") == 404  # cached until the window ends
    clock[0] += 60
    assert status("absent-a") == 200
