"""Remote Webapp access (``hermes_cli.webapp_access``, ``/api/webapp-access/*``).

Contracts: Web Access writes only the ``webapp:`` block (the own-URL mode's URL in
``webapp.public_url``), LAN access registers the LAN callback on a Nous Portal client of Web Access's
own (never the dashboard's, never in ``.env``), and nothing is written or restarted without a Nous
account to sign in with.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    """Native Desktop on this computer, calling the gateway's own listener as its Web Access pane
    does: one native grant per request."""
    from tests.hermes_cli.native_gateway_client import native_desktop_client

    return native_desktop_client(monkeypatch)


@pytest.fixture()
def restarts(monkeypatch):
    """Capture the gateway restart that applies the config instead of spawning it."""
    import hermes_cli.web_server_gateway as gw

    calls: list[str] = []

    def _restart(profile, *, what, label):
        calls.append(what)
        return {"restart_started": True}

    monkeypatch.setattr(gw, "_restart_gateway_after", _restart)
    return calls


@pytest.fixture()
def nous(monkeypatch):
    """A signed-in Nous account; portal registration is captured (no network)."""
    from hermes_cli import webapp_access

    callbacks: list[str] = []
    monkeypatch.setattr(webapp_access, "nous_signed_in", lambda: True)
    monkeypatch.setattr(webapp_access, "_register_nous_sign_in", callbacks.append)
    monkeypatch.setattr(webapp_access, "lan_address", lambda: "192.168.1.50")
    return callbacks


@pytest.fixture(autouse=True)
def builds(monkeypatch):
    """Capture the Webapp bundle builds (``hermes_cli.webapp.prepare_webapp_renderer``) instead of
    compiling the renderer."""
    from hermes_cli import webapp

    roots: list = []

    def prepare(project_root, **_options):
        roots.append(project_root)
        return webapp.webapp_dist_dir(project_root)

    monkeypatch.setattr(webapp, "prepare_webapp_renderer", prepare)
    return roots


def _config() -> dict:
    from hermes_cli.config import load_config

    return load_config()


def _dashboard_setup() -> tuple[object, object]:
    """The standalone dashboard's own public URL, in config.yaml and in .env, as Web Access found it."""
    from hermes_cli.config import load_env, read_raw_config

    return read_raw_config().get("dashboard", {}).get("public_url"), load_env().get("HERMES_DASHBOARD_PUBLIC_URL")


def test_web_access_keeps_its_url_in_the_webapp_block_and_leaves_the_dashboard_alone(client, restarts, nous):
    """Start and stop write only ``webapp.*``: the own-URL mode pins ``webapp.public_url``, LAN mode
    drops it (its callback follows each request), and the standalone dashboard's
    ``dashboard.public_url`` and ``.env`` pin, which a tunnelled dashboard needs after a restart,
    stay exactly as the user set them."""
    from hermes_cli.config import save_config, save_env_value

    save_config({**_config(), "dashboard": {"public_url": "https://dashboard.example.test"}})
    save_env_value("HERMES_DASHBOARD_PUBLIC_URL", "https://dashboard.example.test")
    dashboard = _dashboard_setup()

    response = client.post("/api/webapp-access/start", json={"mode": "public", "public_url": "https://hermes.example.test"})
    assert response.status_code == 200, response.text
    assert nous == ["https://hermes.example.test/auth/callback"]
    assert {key: _config()["webapp"].get(key) for key in ("enabled", "host", "public_url")} == {
        "enabled": True, "host": "127.0.0.1", "public_url": "https://hermes.example.test"}
    assert response.json()["public_url"] == "https://hermes.example.test"
    assert _dashboard_setup() == dashboard

    assert client.post("/api/webapp-access/stop").status_code == 200
    assert _config()["webapp"]["enabled"] is False
    assert _dashboard_setup() == dashboard

    assert client.post("/api/webapp-access/start", json={"mode": "lan"}).status_code == 200
    assert nous[-1] == "http://192.168.1.50:9120/auth/callback"
    assert _config()["webapp"]["host"] == "0.0.0.0"
    assert not _config()["webapp"].get("public_url")
    assert _dashboard_setup() == dashboard
    assert restarts == ["turning on Web Access", "turning off Web Access", "turning on Web Access"]


def test_start_without_a_nous_account_writes_and_restarts_nothing(client, restarts, nous, monkeypatch):
    from hermes_cli import webapp_access

    monkeypatch.setattr(webapp_access, "nous_signed_in", lambda: False)

    assert client.post("/api/webapp-access/start", json={"mode": "lan"}).status_code == 400
    assert nous == []
    assert restarts == []
    assert not _config().get("webapp", {}).get("enabled")


@pytest.mark.parametrize(("webapp", "mode"), [
    ({"enabled": True}, "local"),
    ({"enabled": True, "host": "::1"}, "local"),
    ({"enabled": True, "host": "0.0.0.0"}, "lan"),
    ({"enabled": True, "public_url": "https://hermes.example.test"}, "public"),
    ({"enabled": False, "host": "0.0.0.0", "public_url": "https://hermes.example.test"}, None),
])
def test_status_names_the_mode_the_config_serves(client, webapp, mode):
    """The pane's mode follows the config: the documented loopback setup (``webapp.enabled: true``)
    serves this computer only, a wildcard host serves the network, and ``webapp.public_url`` serves
    the user's own URL. A loopback Webapp is never shown as "My own URL" with no address."""
    from hermes_cli.config import save_config

    save_config({**_config(), "webapp": webapp})

    status = client.get("/api/webapp-access/status")
    assert status.status_code == 200, status.text
    assert status.json()["mode"] == mode


def test_start_keeps_a_custom_port_and_builds_the_addresses_on_it(client, restarts, nous):
    """A ``webapp.port`` the user chose (a second gateway on this machine needs its own) survives
    every start and stop, and the status and the LAN callback use it, even while Web Access is off."""
    from hermes_cli.config import save_config

    save_config({**_config(), "webapp": {"enabled": False, "port": 9333}})
    assert client.get("/api/webapp-access/status").json()["port"] == 9333

    assert client.post("/api/webapp-access/start", json={"mode": "lan"}).status_code == 200
    assert nous == ["http://192.168.1.50:9333/auth/callback"]
    assert _config()["webapp"]["port"] == 9333

    assert client.post("/api/webapp-access/stop").status_code == 200
    response = client.post("/api/webapp-access/start", json={"mode": "public", "public_url": "https://hermes.example.test"})
    assert response.status_code == 200, response.text
    assert _config()["webapp"]["port"] == 9333
    assert response.json()["port"] == 9333


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("public_url", [
    "https://user:secret@hermes.example.test",
    "https://hermes.example.test/?next=x",
    "https://hermes.example.test:99999",
    "hermes.example.test",
])
def test_status_never_shows_a_public_url_the_gateway_refuses(client, public_url, enabled):
    """The pane shows the ``webapp:`` block as the gateway reads it, with the gateway's own reader: a
    ``webapp.public_url`` the gateway refuses shows as none, whether the Webapp is on or off, and
    the port the user chose still shows."""
    from hermes_cli.config import save_config

    save_config({**_config(), "webapp": {"enabled": enabled, "port": 9333, "public_url": public_url}})

    status = client.get("/api/webapp-access/status").json()
    assert (status["enabled"], status["mode"], status["public_url"], status["port"]) == (False, None, "", 9333)


@pytest.mark.parametrize("public_url", [
    "https://hermes.example.test:99999",
    "https://hermes.example.test:0",
    "https://hermes.example.test/a@b",
    "https://hermes.example.test/a b",
])
def test_start_refuses_a_public_url_the_gateway_would_refuse(client, restarts, nous, public_url):
    """Start checks the URL with the gateway's own ``webapp.public_url`` reader (plus https on a host
    other devices reach), so it never registers, saves and restarts onto a Webapp the gateway then
    keeps off."""
    response = client.post("/api/webapp-access/start", json={"mode": "public", "public_url": public_url})
    assert response.status_code == 400, response.text
    assert nous == []
    assert restarts == []
    assert not _config().get("webapp", {}).get("enabled")


def test_start_takes_the_public_url_as_typed(client, restarts, nous):
    """Spaces around the URL and a trailing slash are not part of the address."""
    response = client.post("/api/webapp-access/start", json={"mode": "public", "public_url": "  https://hermes.example.test/ "})
    assert response.status_code == 200, response.text
    assert nous == ["https://hermes.example.test/auth/callback"]
    assert _config()["webapp"]["public_url"] == "https://hermes.example.test"


@pytest.fixture()
def lan_browser(monkeypatch):
    """The gateway's Webapp on all interfaces behind password sign-in, and a browser on the LAN
    signed in with a cookie session."""
    import plugins.dashboard_auth.basic as basic_plugin
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from hermes_cli.dashboard_auth.routes import _reset_password_rate_limit

    monkeypatch.delenv("HERMES_DASHBOARD_PUBLIC_URL", raising=False)
    for key, value in (("bound_host", "0.0.0.0"), ("bound_port", 9120), ("auth_required", True),
                       ("trusted_public_hosts", frozenset()), ("ui_surface", "webapp"),
                       ("listener_public_url", "")):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    clear_providers()
    register_provider(basic_plugin.BasicAuthProvider(
        username="admin", password_hash=basic_plugin.hash_password("hunter2"), secret=b"s" * 32))
    _reset_password_rate_limit()
    browser = TestClient(web_server.app, base_url="http://192.168.1.50:9120")
    login = browser.post("/auth/password-login", json={"provider": "basic", "username": "admin", "password": "hunter2"})
    assert login.status_code == 200, login.text
    yield browser
    clear_providers()


def test_a_browser_signed_in_over_the_network_cannot_use_web_access(lan_browser, restarts, nous):
    """Web Access decides what this machine exposes, so only its owner here may use it. A browser
    signed in behind the sign-in gate may be on another device: every Web Access route answers it
    403 and changes nothing, while the same session still reaches an ordinary route."""
    page = {"Origin": "http://192.168.1.50:9120", "Sec-Fetch-Site": "same-origin"}

    assert lan_browser.get("/api/config").status_code == 200
    assert lan_browser.get("/api/webapp-access/status").status_code == 403
    assert lan_browser.post("/api/webapp-access/start", json={"mode": "lan"}, headers=page).status_code == 403
    assert lan_browser.post("/api/webapp-access/stop?profile=default", headers=page).status_code == 403
    assert nous == []
    assert restarts == []
    assert not _config()["webapp"]["enabled"]


@pytest.fixture()
def webapp_tab(monkeypatch):
    """A tab of the default Webapp (loopback, no sign-in gate), which may be a browser on another
    machine through ``ssh -L 9120:127.0.0.1:9120``: it holds the browser credential, which on this
    listener is the loopback session token."""
    from hermes_cli import web_server

    for key, value in (("ui_surface", "webapp"), ("auth_required", False), ("bound_host", "127.0.0.1"),
                       ("trusted_public_hosts", frozenset())):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    tab = TestClient(web_server.app, base_url="http://127.0.0.1:9120", raise_server_exceptions=False)
    tab.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    yield tab
    tab.close()


def test_a_webapp_tab_cannot_use_web_access(webapp_tab, restarts, nous):
    """Web Access decides what this machine exposes, so only native Desktop on this computer may use
    it: each of its requests carries a one-use grant from the gateway's private control socket, which
    answers only this OS user, and no Origin, and a browser tab can present neither. The browser
    credential is not that proof, even where it is the loopback session token: every Web Access
    route answers a Webapp tab 403 and changes nothing (the pane being hidden in the browser is UI
    only)."""
    assert webapp_tab.get("/api/webapp-access/status").status_code == 403
    assert webapp_tab.post("/api/webapp-access/start", json={"mode": "lan"}).status_code == 403
    assert webapp_tab.post("/api/webapp-access/stop").status_code == 403
    assert webapp_tab.post("/api/webapp-access/restart").status_code == 403
    assert nous == []
    assert restarts == []
    assert not _config().get("webapp", {}).get("enabled")


def test_native_desktop_uses_web_access_on_the_webapps_own_listener(client, monkeypatch):
    """With the Webapp on, native Desktop dials the gateway's listener that serves it, and its Web
    Access pane keeps working there."""
    from hermes_cli import web_server

    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)

    assert client.get("/api/webapp-access/status").status_code == 200


def test_start_builds_the_webapp_bundle_before_turning_it_on(client, restarts, nous, builds):
    """Start builds the bundle the gateway serves (``apps/desktop/dist-webapp``) with the function
    ``hermes webapp`` uses, reusing it when current, so it never reports a Webapp with nothing to
    serve as running."""
    from hermes_cli.config import get_project_root

    assert client.post("/api/webapp-access/start", json={"mode": "lan"}).status_code == 200
    assert builds == [get_project_root()]
    assert restarts == ["turning on Web Access"]


def test_a_failed_build_turns_nothing_on_and_says_how_to_build(client, restarts, nous, monkeypatch):
    """When the bundle cannot be built, Start registers nothing with the Portal, writes nothing and
    restarts nothing, and its error names the command that builds the bundle."""
    from hermes_cli import webapp

    def fail(_project_root, **_options):
        raise webapp.WebappBuildError("Browser-hosted Desktop build failed: npm exited 1")

    monkeypatch.setattr(webapp, "prepare_webapp_renderer", fail)

    response = client.post("/api/webapp-access/start", json={"mode": "lan"})
    assert response.status_code == 400
    assert "npm exited 1" in response.json()["detail"]
    assert "hermes webapp --build-only" in response.json()["detail"]
    assert nous == []
    assert restarts == []
    assert not _config()["webapp"]["enabled"]


@pytest.mark.parametrize("step", ["build", "registration"])
def test_other_requests_answer_while_start_builds_and_registers(client, restarts, nous, monkeypatch, step):
    """Start's slow steps hold no lock that ordinary requests take: while the bundle builds (it may
    compile for minutes or wait up to 30 minutes on another build) and while the Portal registers
    the callback (a network round trip), a config request answers at once."""
    import threading

    from hermes_cli import webapp, webapp_access

    answered_during_step: list[list[int]] = []

    def ask_for_config(answers: list[int]) -> None:
        other = TestClient(client.app, raise_server_exceptions=False)
        answers.append(other.get("/api/config").status_code)

    def slow_step(*_args, **_options):
        answers: list[int] = []
        request = threading.Thread(target=ask_for_config, args=(answers,), daemon=True)
        request.start()
        request.join(timeout=10)
        answered_during_step.append(list(answers))

    if step == "build":
        monkeypatch.setattr(webapp, "prepare_webapp_renderer", slow_step)
    else:
        monkeypatch.setattr(webapp_access, "_register_nous_sign_in", slow_step)

    response = client.post("/api/webapp-access/start", json={"mode": "lan"})
    assert response.status_code == 200, response.text
    assert answered_during_step == [[200]]
    assert restarts == ["turning on Web Access"]


def test_restart_applies_the_saved_config_and_does_nothing_else(client, restarts, nous, builds):
    """The pane's "Restart gateway" (a saved change the gateway has not applied yet) restarts the
    gateway serving the profile and nothing else: no Portal registration, no build, no config
    write."""
    from hermes_cli.config import read_raw_config, save_config

    save_config({**_config(), "webapp": {"enabled": True, "host": "0.0.0.0"}})
    saved = read_raw_config()

    response = client.post("/api/webapp-access/restart")
    assert response.status_code == 200, response.text
    assert response.json()["restart_started"] is True
    assert response.json()["mode"] == "lan"
    assert restarts == ["applying Web Access"]
    assert nous == []
    assert builds == []
    assert read_raw_config() == saved


def test_a_restart_refused_by_a_running_chat_is_reported_as_busy(client, nous, monkeypatch):
    """When a running chat refuses the gateway restart, the change stays saved and the answer says
    so in a field the pane can word plainly (``restart_busy``), beside the CLI's own refusal text."""
    import hermes_cli.web_server_gateway as gw

    refusal = ("Refusing to restart the gateway: 1 live session(s) would be interrupted.\n"
               "    POST /api/gateway/restart?interrupt=true   # interrupts 1 live session(s)")
    monkeypatch.setattr(gw, "live_session_refusal", lambda _profile, _verb: refusal)

    response = client.post("/api/webapp-access/start", json={"mode": "lan"})
    assert response.status_code == 200, response.text
    assert response.json()["restart_started"] is False
    assert response.json()["restart_busy"] is True
    assert response.json()["enabled"] is True


DASHBOARD_CLIENT = "agent:dashboard"
DASHBOARD_CALLBACK = "https://dashboard.example.test/auth/callback"
LAN_CALLBACK = "http://192.168.1.50:9120/auth/callback"


@pytest.fixture()
def portal(monkeypatch):
    """A signed-in Nous account and a fake Nous Portal (no network). Its self-hosted-client endpoint
    mints ``agent:client-N`` for a request without a known client id and, for one it knows, REPLACES
    that client's redirect URI: the worst case the real Portal may do. It already holds the
    standalone dashboard's client (``agent:dashboard``)."""
    from types import SimpleNamespace

    from hermes_cli import auth, dashboard_register, webapp_access

    clients = {DASHBOARD_CLIENT: DASHBOARD_CALLBACK}
    requests: list[dict] = []

    def register(*, access_token, portal_base_url, name, custom_redirect_uri, existing_client_id=None, timeout=15.0):
        requests.append({"portal": portal_base_url, "name": name, "callback": custom_redirect_uri,
                         "client_id": existing_client_id})
        client_id = existing_client_id if existing_client_id in clients else f"agent:client-{len(clients)}"
        clients[client_id] = custom_redirect_uri
        return {"client_id": client_id, "name": name or "kept"}

    monkeypatch.delenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.setenv("HERMES_DASHBOARD_PORTAL_URL", "https://portal.example.test")
    monkeypatch.setattr(dashboard_register, "_register_self_hosted_client", register)
    monkeypatch.setattr(auth, "resolve_nous_access_token", lambda: "throwaway-access-token")
    monkeypatch.setattr(webapp_access, "nous_signed_in", lambda: True)
    monkeypatch.setattr(webapp_access, "lan_address", lambda: "192.168.1.50")
    return SimpleNamespace(clients=clients, requests=requests)


def _env_bytes() -> bytes:
    from hermes_cli.config import get_env_path

    path = get_env_path()
    return path.read_bytes() if path.exists() else b""


def test_start_registers_a_portal_client_of_its_own_and_never_the_dashboards(client, restarts, portal, monkeypatch):
    """Web Access signs the Webapp in with a Portal client of its own, so turning it on can never
    change the redirect URI of the standalone dashboard's client, whatever the Portal does on an
    update. The first Start registers a new client, never ``HERMES_DASHBOARD_OAUTH_CLIENT_ID``, and
    writes nothing to ``.env``; a second Start with the same callback asks the Portal nothing; a moved
    callback updates Web Access's own client. The dashboard's client keeps its redirect URI and
    ``.env`` stays exactly as it was."""
    from hermes_cli import webapp_access
    from hermes_cli.config import save_env_value

    save_env_value("HERMES_DASHBOARD_OAUTH_CLIENT_ID", DASHBOARD_CLIENT)
    env = _env_bytes()

    assert client.post("/api/webapp-access/start", json={"mode": "lan"}).status_code == 200
    assert [(r["callback"], r["client_id"]) for r in portal.requests] == [(LAN_CALLBACK, None)]
    assert portal.requests[0]["name"]
    own = next(cid for cid, callback in portal.clients.items() if cid != DASHBOARD_CLIENT)
    assert portal.clients == {DASHBOARD_CLIENT: DASHBOARD_CALLBACK, own: LAN_CALLBACK}
    assert _env_bytes() == env

    assert client.post("/api/webapp-access/start", json={"mode": "lan"}).status_code == 200
    assert len(portal.requests) == 1

    monkeypatch.setattr(webapp_access, "lan_address", lambda: "192.168.1.51")
    assert client.post("/api/webapp-access/start", json={"mode": "lan"}).status_code == 200
    moved = "http://192.168.1.51:9120/auth/callback"
    assert (portal.requests[-1]["callback"], portal.requests[-1]["client_id"]) == (moved, own)
    assert portal.clients == {DASHBOARD_CLIENT: DASHBOARD_CALLBACK, own: moved}
    assert _env_bytes() == env


def test_web_access_keeps_its_client_id_under_the_gateway_home_not_in_env(portal):
    """The id of Web Access's client lives in ``webapp/sign-in-registration.json`` under the gateway
    home, beside the Portal it was registered on and the callback it carries; ``.env`` gets no
    ``HERMES_DASHBOARD_OAUTH_CLIENT_ID``, so a standalone dashboard on this home finds no client it
    did not register itself."""
    import json

    from hermes_cli import webapp_access
    from hermes_cli.config import load_env
    from hermes_constants import get_hermes_home

    webapp_access.register(LAN_CALLBACK)

    record = json.loads((get_hermes_home() / "webapp" / "sign-in-registration.json").read_text(encoding="utf-8"))
    assert {key: record.get(key) for key in ("portal", "client_id", "callback")} == {
        "portal": "https://portal.example.test", "client_id": "agent:client-1", "callback": LAN_CALLBACK}
    assert "HERMES_DASHBOARD_OAUTH_CLIENT_ID" not in load_env()
    assert webapp_access.sign_in_client(get_hermes_home()) == ("agent:client-1", "https://portal.example.test")


@pytest.mark.platforms("posix")  # POSIX mode bits; a Windows home keeps its own ACL
def test_web_access_keeps_its_client_record_user_only(portal):
    import stat

    from hermes_cli import webapp_access
    from hermes_constants import get_hermes_home

    webapp_access.register(LAN_CALLBACK)
    record = get_hermes_home() / "webapp" / "sign-in-registration.json"
    assert stat.S_IMODE(record.stat().st_mode) == 0o600
    assert stat.S_IMODE(record.parent.stat().st_mode) == 0o700


def test_a_home_registered_on_the_dashboards_client_gets_its_own_at_the_next_start(portal):
    """Earlier builds of Web Access registered the Webapp's callback on the dashboard's client and
    recorded that client's id. The next Start registers a client of its own, even for the same
    callback, and leaves the dashboard's client and ``.env`` alone; until then the record names no
    client of Web Access's own, so the Webapp keeps signing in with the configured one."""
    import json

    from hermes_cli import webapp_access
    from hermes_cli.config import save_env_value
    from hermes_constants import get_hermes_home

    save_env_value("HERMES_DASHBOARD_OAUTH_CLIENT_ID", DASHBOARD_CLIENT)
    record = get_hermes_home() / "webapp" / "sign-in-registration.json"
    record.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    record.write_text(json.dumps({"portal": "https://portal.example.test", "client_id": DASHBOARD_CLIENT,
                                  "callback": LAN_CALLBACK}) + "\n", encoding="utf-8")
    env = _env_bytes()
    assert webapp_access.sign_in_client(get_hermes_home()) is None

    webapp_access.register(LAN_CALLBACK)

    assert [(r["callback"], r["client_id"]) for r in portal.requests] == [(LAN_CALLBACK, None)]
    assert portal.clients[DASHBOARD_CLIENT] == DASHBOARD_CALLBACK
    assert webapp_access.sign_in_client(get_hermes_home()) == ("agent:client-1", "https://portal.example.test")
    assert _env_bytes() == env
    webapp_access.register(LAN_CALLBACK)
    assert len(portal.requests) == 1


def test_dashboard_register_leaves_web_access_and_its_client_alone(portal):
    """``hermes dashboard register`` updates the dashboard's own client only: Web Access's client,
    its record and its next Start (which asks the Portal nothing for an unchanged callback) are
    untouched."""
    import argparse

    from hermes_cli import dashboard_register, webapp_access
    from hermes_cli.config import save_env_value
    from hermes_constants import get_hermes_home

    save_env_value("HERMES_DASHBOARD_OAUTH_CLIENT_ID", DASHBOARD_CLIENT)
    webapp_access.register(LAN_CALLBACK)
    record = (get_hermes_home() / "webapp" / "sign-in-registration.json").read_bytes()

    dashboard_register.cmd_dashboard_register(argparse.Namespace(
        name=None, redirect_uri="https://dashboard2.example.test/auth/callback", portal_url=None))

    assert portal.requests[-1]["client_id"] == DASHBOARD_CLIENT
    assert portal.clients["agent:client-1"] == LAN_CALLBACK
    assert (get_hermes_home() / "webapp" / "sign-in-registration.json").read_bytes() == record
    webapp_access.register(LAN_CALLBACK)
    assert len(portal.requests) == 2


async def _login_client_id(mount, origin: str) -> str:
    """The Portal client id the listener behind *mount* sends a browser starting Nous sign-in with."""
    from urllib.parse import parse_qs, urlsplit

    import httpx

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mount), base_url=origin) as browser:
        started = await browser.get("/auth/login", params={"provider": "nous"})
    assert started.status_code in {302, 307}, started.text
    location = urlsplit(started.headers["location"])
    assert f"{location.scheme}://{location.netloc}" == "https://portal.example.test"
    return parse_qs(location.query)["client_id"][0]


def _load_nous_plugin_as_discovery_does():
    """The bundled Nous sign-in plugin imported as plugin discovery imports it: from its folder under
    a module name of the loader's own (``hermes_plugins.<key>``), not as ``plugins.dashboard_auth.nous``,
    so its module-level state is not that of any other import of it."""
    import importlib.util
    from pathlib import Path

    import plugins.dashboard_auth.nous as source

    folder = Path(source.__file__).parent
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins_test_f5.dashboard_auth__nous", folder / "__init__.py", submodule_search_locations=[str(folder)])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def gateway_listeners(tmp_path, monkeypatch):
    """The gateway's dashboard mount over the process's dashboard app, as the gateway builds it at
    start: ``listener(webapp=True)`` is the Webapp's listener (every interface, behind sign-in), else
    the private one. ``discover()`` runs the Nous plugin's registration as plugin discovery does at
    gateway start (and on a plugin reload), on the plugin imported the way discovery imports it. The
    auth registry and the listener's sign-in client are reset before and after."""
    import types

    from gateway.run_api import GatewayDashboardMount
    from gateway.webapp_listener import WebappMount
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import clear_providers, listener_client
    from hermes_cli.dashboard_auth.registry import register_global_provider
    from hermes_constants import get_hermes_home

    for key in ("bound_host", "bound_port", "auth_required", "trusted_public_hosts"):
        monkeypatch.setattr(web_server.app.state, key, getattr(web_server.app.state, key, None), raising=False)
    for name in ("_current", "_swap"):
        monkeypatch.setattr(listener_client, name, None)
    dist = tmp_path / "dist-webapp"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><html><body>WEBAPP</body></html>", encoding="utf-8")
    discovery = types.SimpleNamespace(register_dashboard_auth_provider=register_global_provider)
    nous_plugin = _load_nous_plugin_as_discovery_does()

    def listener(*, webapp: bool):
        state = types.SimpleNamespace(bound_host="0.0.0.0", bound_port=9120,
                                      **({"webapp": WebappMount(dist, get_hermes_home())} if webapp else {}))
        return GatewayDashboardMount(types.SimpleNamespace(session_authority=None), state)

    clear_providers()
    yield types.SimpleNamespace(listener=listener, discover=lambda: nous_plugin.register(discovery))
    clear_providers()


@pytest.mark.parametrize("dashboard_client", [DASHBOARD_CLIENT, None])
def test_the_webapps_nous_sign_in_uses_its_own_client_and_the_dashboards_keeps_its_own(
        portal, gateway_listeners, monkeypatch, dashboard_client):
    """The gateway's Webapp listener signs browsers in to Nous with the client Web Access registered
    for it, also on a home with no dashboard client at all. The gateway's private listener mounted
    after it signs in with ``HERMES_DASHBOARD_OAUTH_CLIENT_ID`` as on main (the standalone dashboard
    reads the same setting), or offers no Nous sign-in when that is unset."""
    import asyncio

    from hermes_cli import webapp_access
    from hermes_cli.dashboard_auth import get_provider

    webapp_access.register(LAN_CALLBACK)
    if dashboard_client:
        monkeypatch.setenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", dashboard_client)
    gateway_listeners.discover()

    async def scenario():
        webapp = gateway_listeners.listener(webapp=True)
        await webapp.ensure()
        try:
            assert await _login_client_id(webapp, "http://192.168.1.50:9120") == "agent:client-1"
        finally:
            await webapp.close()
        if dashboard_client is None:
            assert get_provider("nous") is None
            return
        private = gateway_listeners.listener(webapp=False)
        await private.ensure()
        try:
            assert await _login_client_id(private, "http://192.168.1.50:9120") == DASHBOARD_CLIENT
        finally:
            await private.close()

    asyncio.run(scenario())


def test_a_plugin_reload_while_the_webapp_serves_keeps_its_own_client(portal, gateway_listeners, monkeypatch):
    """Reloading plugins re-registers the Nous provider; while the gateway serves the Webapp it
    is rebuilt on Web Access's client, not on ``HERMES_DASHBOARD_OAUTH_CLIENT_ID``."""
    import asyncio

    from hermes_cli import webapp_access

    webapp_access.register(LAN_CALLBACK)
    monkeypatch.setenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", DASHBOARD_CLIENT)
    gateway_listeners.discover()

    async def scenario():
        webapp = gateway_listeners.listener(webapp=True)
        await webapp.ensure()
        try:
            gateway_listeners.discover()
            assert await _login_client_id(webapp, "http://192.168.1.50:9120") == "agent:client-1"
        finally:
            await webapp.close()

    asyncio.run(scenario())


def test_a_disabled_nous_plugin_keeps_nous_sign_in_off_the_webapp(portal, gateway_listeners, tmp_path):
    """Web Access's client signs the Webapp in only through the Nous sign-in plugin: where
    ``plugins.disabled`` keeps that plugin from loading (discovery never registered it), the Webapp
    gets no Nous sign-in from Web Access either."""
    from gateway.webapp_listener import WebappMount, use_own_sign_in
    from hermes_cli import webapp_access
    from hermes_cli.dashboard_auth import get_provider
    from hermes_constants import get_hermes_home

    webapp_access.register(LAN_CALLBACK)

    assert use_own_sign_in(WebappMount(tmp_path, get_hermes_home())) is None
    assert get_provider("nous") is None


def test_a_webapp_without_a_web_access_client_signs_in_with_the_configured_one(portal, gateway_listeners, monkeypatch):
    """A Webapp whose Nous sign-in was set up by hand (``hermes dashboard register`` for its address,
    no Web Access Start) keeps signing in with ``HERMES_DASHBOARD_OAUTH_CLIENT_ID``."""
    import asyncio

    monkeypatch.setenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", DASHBOARD_CLIENT)
    gateway_listeners.discover()

    async def scenario():
        webapp = gateway_listeners.listener(webapp=True)
        await webapp.ensure()
        try:
            assert await _login_client_id(webapp, "http://192.168.1.50:9120") == DASHBOARD_CLIENT
        finally:
            await webapp.close()

    asyncio.run(scenario())


def _lock_is_free(lock) -> bool:
    """Whether another thread can take *lock* now (the caller's own hold of an RLock does not count)."""
    import threading

    free: list[bool] = []

    def probe() -> None:
        taken = lock.acquire(timeout=1)
        if taken:
            lock.release()
        free.append(taken)

    other = threading.Thread(target=probe, daemon=True)
    other.start()
    other.join(5)
    return free == [True]


@pytest.mark.parametrize("action", ["start", "stop", "restart"])
def test_the_gateway_restart_and_status_run_outside_the_config_lock(client, nous, monkeypatch, action):
    """Start and Stop write the config under the config lock every config, skills and tools request
    takes, then restart the gateway and read its status outside that lock: the restart spawn and the
    status query (a control-socket round trip of up to 5 seconds) hold no other request up."""
    import hermes_cli.web_server_gateway as gw
    from hermes_cli import web_server, webapp_access

    lock = web_server._CONFIG_MUTATION_LOCK
    free_during: dict[str, bool] = {}

    def restart(_profile, *, what, label):
        free_during["restart"] = _lock_is_free(lock)
        return {"restart_started": True}

    real_state = webapp_access._gateway_state

    def gateway_state(home):
        free_during["status"] = _lock_is_free(lock)
        return real_state(home)

    monkeypatch.setattr(gw, "_restart_gateway_after", restart)
    monkeypatch.setattr(webapp_access, "_gateway_state", gateway_state)
    response = {"start": lambda: client.post("/api/webapp-access/start", json={"mode": "lan"}),
                "stop": lambda: client.post("/api/webapp-access/stop"),
                "restart": lambda: client.post("/api/webapp-access/restart")}[action]()
    assert response.status_code == 200, response.text
    assert free_during == {"restart": True, "status": True}


@pytest.mark.parametrize(("held", "second"), [("build", "stop"), ("registration", "public")])
def test_web_access_changes_run_one_at_a_time(client, restarts, nous, monkeypatch, held, second):
    """Start, Stop and Restart each build, register, write and restart as one change, and a second
    change waits for the one in flight. A Stop sent while a Start builds therefore ends with the
    Webapp off, and two windows starting different modes end with the Portal's callback and the
    served mode both the last one's: never Nous sign-in returning to an address the Webapp does not
    serve, nor the Webapp on the network after the user stopped it."""
    import threading

    from hermes_cli import webapp, webapp_access
    from hermes_cli.web_routers import webapp_access as routes

    class WatchedLock:
        """The routes' change lock, noting when a change found it held."""

        def __init__(self):
            self.lock, self.found_held = threading.Lock(), threading.Event()

        def acquire(self, blocking=True, timeout=-1):
            taken = self.lock.acquire(blocking, timeout)
            if not taken:
                self.found_held.set()
            return taken

        def release(self):
            self.lock.release()

    watched = WatchedLock()
    monkeypatch.setattr(routes, "_CHANGE_LOCK", watched)
    entered, release = threading.Event(), threading.Event()

    def hold(*_args, **_options):
        if not entered.is_set():
            entered.set()
            release.wait(20)

    if held == "build":
        monkeypatch.setattr(webapp, "prepare_webapp_renderer", lambda root, **options: hold())
    else:
        monkeypatch.setattr(webapp_access, "_register_nous_sign_in",
                            lambda callback: (hold(), nous.append(callback)))
    public_url = "https://hermes.example.test"
    requests = {"lan": ("/api/webapp-access/start", {"mode": "lan"}),
                "stop": ("/api/webapp-access/stop", None),
                "public": ("/api/webapp-access/start", {"mode": "public", "public_url": public_url})}
    answers: dict[str, int] = {}

    def send(name: str) -> None:
        # Each thread needs its own client on the same native-Desktop app, which mints a grant per request.
        other = TestClient(client.app, raise_server_exceptions=False)
        path, body = requests[name]
        answers[name] = other.post(path, json=body).status_code

    first = threading.Thread(target=send, args=("lan",), daemon=True)
    first.start()
    assert entered.wait(20)
    later = threading.Thread(target=send, args=(second,), daemon=True)
    later.start()
    assert watched.found_held.wait(20)
    assert restarts == [] and answers == {}  # the second change waits for the first
    release.set()
    first.join(30)
    later.join(30)
    assert answers == {"lan": 200, second: 200}
    block = _config()["webapp"]
    if second == "stop":
        assert restarts == ["turning on Web Access", "turning off Web Access"]
        assert block["enabled"] is False
    else:
        assert nous == ["http://192.168.1.50:9120/auth/callback", f"{public_url}/auth/callback"]
        assert (block["enabled"], block.get("public_url")) == (True, public_url)


def test_a_gateway_older_than_the_webapp_reads_as_serving_none_until_it_restarts(monkeypatch, tmp_path):
    """A gateway that predates the Webapp answers its status with "unknown verb". It serves no Webapp,
    so Web Access reports it ``off`` with a reason that names the restart (the pane then offers
    Restart gateway), never ``unavailable``, which the pane's restart wait takes as the final answer."""
    from pathlib import Path

    from hermes_cli import webapp_access, webapp_launcher

    answer = {"ok": False, "error": "unknown verb: 'webapp-status'", "protocol": 1,
              "supported_verbs": ["identify", "session-ticket", "status"]}
    monkeypatch.setattr(webapp_launcher, "_control", lambda _home, _verb, params=None: answer)
    state = webapp_access._gateway_state(Path(tmp_path))
    assert state["state"] == "off", state
    assert webapp_launcher.RESTART_COMMAND in state["reason"]
