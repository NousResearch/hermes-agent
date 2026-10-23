"""Remote Webapp access (``hermes_cli.webapp_access``, ``/api/webapp-access/*``).

Contracts: Web Access writes only the ``webapp:`` block (the own-URL mode's URL in
``webapp.public_url``), LAN access registers the LAN callback, and nothing is written or restarted
without a Nous account to sign in with.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from hermes_cli import web_server

    c = TestClient(web_server.app, raise_server_exceptions=False)
    c.headers["Authorization"] = f"Bearer {web_server._SESSION_TOKEN}"
    return c


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


@pytest.fixture()
def lan_browser(monkeypatch):
    """This listener on all interfaces behind password sign-in, and a browser on the LAN signed in
    with a cookie session."""
    import plugins.dashboard_auth.basic as basic_plugin
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from hermes_cli.dashboard_auth.routes import _reset_password_rate_limit

    monkeypatch.delenv("HERMES_DASHBOARD_PUBLIC_URL", raising=False)
    for key, value in (("bound_host", "0.0.0.0"), ("bound_port", 9120), ("auth_required", True),
                       ("trusted_public_hosts", frozenset())):
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


def test_web_access_answers_the_native_app_and_the_loopback_token_only():
    """The owner on this machine is the native app (its same-user gateway ticket, gated or not) or,
    on a listener without sign-in, the holder of the loopback session token, which the auth
    middleware has already checked. Behind the sign-in gate anything else is a remote login."""
    from types import SimpleNamespace

    from fastapi import HTTPException

    from hermes_cli.web_routers.webapp_access import owner_on_this_machine

    def request(*, gated: bool, native: bool):
        return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(auth_required=gated)),
                               state=SimpleNamespace(native_http_principal={"profile_id": "x"} if native else None))

    owner_on_this_machine(request(gated=True, native=True))
    owner_on_this_machine(request(gated=False, native=False))
    with pytest.raises(HTTPException) as refused:
        owner_on_this_machine(request(gated=True, native=False))
    assert refused.value.status_code == 403


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

    from hermes_cli import web_server, webapp, webapp_access

    answered_during_step: list[list[int]] = []

    def ask_for_config(answers: list[int]) -> None:
        other = TestClient(web_server.app, raise_server_exceptions=False)
        other.headers.update(client.headers)
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


def test_the_portal_is_asked_again_only_when_the_callback_or_client_changed(monkeypatch):
    """Registering posts the callback under the dashboard's shared OAuth client id, and the Portal
    may replace that client's redirect URI, so Web Access does not post the same callback for the
    same client and Portal twice. A new LAN address, a new client id, or a registration made by
    ``hermes dashboard register`` in between makes the next start register again."""
    import argparse

    from hermes_cli import auth, dashboard_register, webapp_access
    from hermes_cli.config import save_env_value

    posts: list[tuple] = []

    def portal(*, access_token, portal_base_url, name, custom_redirect_uri, existing_client_id, timeout=15.0):
        posts.append((custom_redirect_uri, existing_client_id))
        return {"client_id": existing_client_id or "client-1", "name": name or "kept"}

    monkeypatch.delenv("HERMES_DASHBOARD_PORTAL_URL", raising=False)
    monkeypatch.setattr(dashboard_register, "_register_self_hosted_client", portal)
    monkeypatch.setattr(dashboard_register, "_resolve_portal_base_url", lambda _override=None: "https://portal.example.test")
    monkeypatch.setattr(auth, "resolve_nous_access_token", lambda: "token")
    lan = "http://192.168.1.50:9120/auth/callback"

    webapp_access.register(lan)
    webapp_access.register(lan)
    assert posts == [(lan, None)]

    moved = "http://192.168.1.51:9120/auth/callback"
    webapp_access.register(moved)
    assert posts[-1] == (moved, "client-1")

    save_env_value("HERMES_DASHBOARD_OAUTH_CLIENT_ID", "client-2")
    webapp_access.register(moved)
    assert posts[-1] == (moved, "client-2")

    dashboard_register.cmd_dashboard_register(argparse.Namespace(
        name=None, redirect_uri="https://dashboard.example.test/auth/callback", portal_url=None))
    webapp_access.register(moved)
    assert posts[-1] == (moved, "client-2")
    assert len(posts) == 5
