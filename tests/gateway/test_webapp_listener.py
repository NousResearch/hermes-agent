"""The browser-hosted Desktop on the gateway's own listener: config, bind address and browser credential.

``webapp.enabled`` in the gateway home's ``config.yaml`` turns it on; the gateway then binds its one HTTP
listener to ``webapp.port`` on a loopback or wildcard ``webapp.host``, and native clients keep dialling a
loopback address.
"""
import asyncio
import errno
import json
import os
import re
import stat
from types import SimpleNamespace

import pytest

from gateway import webapp_listener as webapp


def _home(tmp_path, config=None):
    home = tmp_path / "home"
    home.mkdir(parents=True)
    if config is not None:
        (home / "config.yaml").write_text(config, encoding="utf-8")
    return home


def test_the_webapp_stays_off_until_config_turns_it_on(tmp_path):
    off = webapp.webapp_settings(_home(tmp_path))
    assert (off.enabled, off.bind) == (False, ("127.0.0.1", 0))


def test_an_enabled_webapp_binds_its_configured_address(tmp_path):
    on = webapp.webapp_settings(_home(tmp_path, "webapp:\n  enabled: true\n  host: 0.0.0.0\n  port: 9555\n"))
    assert (on.enabled, on.bind) == (True, ("0.0.0.0", 9555))
    default = webapp.webapp_settings(_home(tmp_path / "d", "webapp:\n  enabled: true\n"))
    assert default.bind == ("127.0.0.1", webapp.DEFAULT_PORT)


@pytest.mark.parametrize("block", [
    "webapp: yes\n",
    "webapp:\n  enabled: true\n  port: 70000\n",
    "webapp:\n  enabled: true\n  port: http\n",
    "webapp:\n  enabled: true\n  host: ''\n",
    "webapp:\n  enabled: 'true'\n",
    "webapp:\n  enabled: true\n  host: 192.168.1.20\n",
    "webapp:\n  enabled: true\n  host: hermes.example.net\n",
])
def test_a_malformed_webapp_block_keeps_the_listener_private(tmp_path, block):
    settings = webapp.webapp_settings(_home(tmp_path, block))
    assert settings.bind == ("127.0.0.1", 0)


@pytest.mark.parametrize(("bind", "dial"), [
    ("127.0.0.1", "127.0.0.1"),
    ("::1", "::1"),
    ("0.0.0.0", "127.0.0.1"),
    ("::", "::1"),
])
def test_native_clients_dial_loopback_whatever_the_browser_bind(bind, dial):
    assert webapp.dial_host(bind) == dial


@pytest.mark.parametrize(("bind", "origin"), [
    ("127.0.0.1", "http://127.0.0.1:9120"),
    ("0.0.0.0", "http://127.0.0.1:9120"),
    ("::", "http://[::1]:9120"),
    ("::1", "http://[::1]:9120"),
])
def test_the_advertised_api_origin_is_the_loopback_dial_address(bind, origin):
    assert webapp.dial_origin(bind, 9120) == origin


def test_a_gateway_restart_keeps_the_browser_credential(tmp_path):
    home = _home(tmp_path)
    first = webapp.browser_session_token(home)
    assert len(first) >= 43
    assert webapp.browser_session_token(home) == first
    if os.name != "nt":
        assert stat.S_IMODE((home / webapp.TOKEN_PATH).stat().st_mode) == 0o600


def test_signing_out_replaces_the_browser_credential(tmp_path):
    home = _home(tmp_path)
    first = webapp.browser_session_token(home)
    second = webapp.browser_session_token(home, rotate=True)
    assert second != first
    assert webapp.browser_session_token(home) == second


@pytest.mark.parametrize("stored", ["short", "x" * 60, "not/url+safe" + "a" * 31])
def test_a_credential_the_browser_would_refuse_is_replaced(tmp_path, stored):
    home = _home(tmp_path)
    (home / webapp.TOKEN_PATH).parent.mkdir(parents=True)
    (home / webapp.TOKEN_PATH).write_text(stored + "\n", encoding="utf-8")
    token = webapp.browser_session_token(home)
    assert token != stored
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", token)


def _start_runtime_api(monkeypatch, tmp_path, config, *, busy=False, refuse=None, while_starting=None):
    """Start the runtime's listener for *config* with the listener itself stood in; *while_starting*
    (a list) records what each Webapp verb answers while the Webapp's listener starts."""
    from gateway import run_api, run_runtime
    home = _home(tmp_path, config)
    monkeypatch.setenv("HERMES_HOME", str(home))
    calls = []

    async def start(runner, *, host="127.0.0.1", port=0, webapp=None):
        calls.append((host, port, webapp))
        if while_starting is not None and webapp is not None:
            while_starting.append({verb: _ask(runner, verb) for verb in WEBAPP_VERBS})
        if busy and port:
            raise OSError(errno.EADDRINUSE, "Address already in use")
        if refuse is not None and webapp is not None:
            raise refuse
        return SimpleNamespace(api_origin=f"http://127.0.0.1:{port or 50001}")

    monkeypatch.setattr(run_api, "start_gateway_api", start)
    from gateway.control_socket import GatewayControlServer
    runner = SimpleNamespace(session_runtime_descriptor={}, session_authorities=[],
                             session_control_server=GatewayControlServer(home))
    asyncio.run(run_runtime.start_gateway_runtime_api(runner))
    return runner, calls, home


WEBAPP_VERBS = ("webapp-status", "webapp-launch", "webapp-sign-out")


def _ask(runner, verb, peer="same-user-peer", params=None, server=None):
    raw = json.dumps({"protocol": 1, "verb": verb, "id": 1, "params": params or {}}).encode()
    return json.loads((server or runner.session_control_server).handle_request_line(raw, peer))


def test_the_webapp_verbs_answer_while_the_webapps_listener_starts(monkeypatch, tmp_path):
    """While the Webapp's listener starts (the dashboard import and mount take seconds), its verbs
    answer: status says ``starting``, which a client polling a restarting gateway keeps waiting on,
    never an unknown verb it would show as a failure. Launch and sign-out refuse as still starting,
    and sign-out leaves the stored credential alone (the starting listener may already hold it)."""
    seen = []
    runner, _, home = _start_runtime_api(
        monkeypatch, tmp_path, "webapp:\n  enabled: true\n  port: 9555\n", while_starting=seen)
    [answers] = seen
    assert answers["webapp-status"]["result"] == {"state": "starting", "url": None, "reason": None}
    for verb in ("webapp-launch", "webapp-sign-out"):
        assert answers[verb]["ok"] is False and "still starting" in answers[verb]["error"], answers[verb]
    assert not (home / webapp.TOKEN_PATH).exists()  # sign-out wrote no credential
    assert runner.webapp_status["state"] == "serving"


def test_the_gateway_registers_the_webapp_verbs_before_its_control_socket_answers(monkeypatch, tmp_path):
    """The gateway's control socket answers from the moment it starts; the Webapp's verbs are on it
    from that moment too (``starting`` until the listener settles), not only once the listener is up."""
    from gateway import control_socket, run_bootstrap
    home = _home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    seen = {}

    async def start(server):
        seen["status"] = _ask(None, "webapp-status", server=server)
        return False  # nothing binds in this test

    monkeypatch.setattr(control_socket.GatewayControlServer, "start", start)
    runner = SimpleNamespace(session_runtime_descriptor={}, session_ticket_store=None)
    assert asyncio.run(run_bootstrap._start_gateway_start_control_socket(runner)) is None
    assert seen["status"]["ok"] is True, seen
    assert seen["status"]["result"]["state"] == "starting"


def test_the_gateway_listens_on_the_webapp_port_when_enabled(monkeypatch, tmp_path):
    runner, calls, home = _start_runtime_api(monkeypatch, tmp_path, "webapp:\n  enabled: true\n  port: 9555\n")
    [(host, port, mount)] = calls
    assert (host, port) == ("127.0.0.1", 9555)
    assert (mount.home, mount.dist.name) == (home, "dist-webapp")
    assert runner.webapp_status == {"state": "serving", "url": "http://127.0.0.1:9555", "reason": None}


def test_a_busy_webapp_port_still_leaves_native_clients_a_listener(monkeypatch, tmp_path):
    runner, calls, _ = _start_runtime_api(
        monkeypatch, tmp_path, "webapp:\n  enabled: true\n  port: 9555\n", busy=True)
    assert [(host, port) for host, port, _ in calls] == [("127.0.0.1", 9555), ("127.0.0.1", 0)]
    assert calls[1][2] is None
    assert runner.session_runtime_descriptor["api_origin"] == "http://127.0.0.1:50001"
    assert runner.webapp_status["state"] == "unavailable"
    assert "127.0.0.1:9555" in runner.webapp_status["reason"]


def test_a_gateway_without_the_webapp_starts_mains_private_listener(monkeypatch, tmp_path):
    runner, calls, _ = _start_runtime_api(monkeypatch, tmp_path, None)
    assert calls == [("127.0.0.1", 0, None)]
    assert runner.webapp_status == {"state": "off", "url": None, "reason": None}


def test_the_launcher_reads_the_webapp_state_from_its_gateway(monkeypatch, tmp_path):
    off, _, _ = _start_runtime_api(monkeypatch, tmp_path / "off", None)
    assert _ask(off, "webapp-status")["result"] == {"state": "off", "url": None, "reason": None}
    busy, _, _ = _start_runtime_api(
        monkeypatch, tmp_path / "busy", "webapp:\n  enabled: true\n  port: 9555\n", busy=True)
    status = _ask(busy, "webapp-status")["result"]
    assert status["state"] == "unavailable" and "127.0.0.1:9555" in status["reason"]


def test_a_gateway_not_serving_the_webapp_mints_no_launch_link(monkeypatch, tmp_path):
    off, _, _ = _start_runtime_api(monkeypatch, tmp_path / "off", None)
    refused = _ask(off, "webapp-launch")
    assert refused["ok"] is False and "webapp.enabled" in refused["error"]
    busy, _, _ = _start_runtime_api(
        monkeypatch, tmp_path / "busy", "webapp:\n  enabled: true\n  port: 9555\n", busy=True)
    refused = _ask(busy, "webapp-launch")
    assert refused["ok"] is False and "127.0.0.1:9555" in refused["error"]


@pytest.mark.parametrize("verb", ["webapp-status", "webapp-launch", "webapp-sign-out"])
def test_webapp_verbs_answer_only_an_authenticated_same_user_peer(monkeypatch, tmp_path, verb):
    runner, _, _ = _start_runtime_api(monkeypatch, tmp_path, "webapp:\n  enabled: true\n  port: 9555\n")
    refused = _ask(runner, verb, peer=None)
    assert refused["ok"] is False and "PermissionError" in refused["error"]


def test_where_one_use_links_cannot_be_checked_the_launcher_gets_one_to_open_by_hand(monkeypatch, tmp_path):
    """Where the server cannot tell who redeems a launch link (macOS, the BSDs, WSL), the launcher still gets
    a one-use link that lasts 60 seconds, never the long-lived browser credential: it spends without a
    peer check, as a window ticket does, and the launcher prints it rather than opening it."""
    from urllib.parse import parse_qs

    runner, _, home = _start_runtime_api(monkeypatch, tmp_path, "webapp:\n  enabled: true\n  port: 9555\n")
    webapp.register_control_verbs(runner, home, links_checked=False)  # as on macOS, the BSDs and WSL
    # The gateway mounts the dashboard app as the Webapp surface on a private (loopback) listener.
    web = SimpleNamespace(_SESSION_TOKEN="s" * 43, app=SimpleNamespace(state=SimpleNamespace(ui_surface="webapp")))
    runner.session_api.app = SimpleNamespace(web=web)
    result = _ask(runner, "webapp-launch")["result"]
    page, fragment = result["url"].split("#", 1)
    assert page == "http://127.0.0.1:9555/webapp/launch"
    assert (result["private"], result["expires_in_seconds"]) == (True, 60)
    assert "s" * 43 not in result["url"] and "hermes-session" not in result["url"]
    [ticket] = parse_qs(fragment)["ticket"]
    _expires, _token, peer_checked = web.app.state.webapp_window_tickets[ticket]
    assert peer_checked is False


def test_behind_the_dashboard_sign_in_the_launcher_gets_the_webapp_address(monkeypatch, tmp_path):
    # The page asks for the sign-in itself; a launch link only works on a private (loopback) Webapp.
    runner, _, _ = _start_runtime_api(
        monkeypatch, tmp_path, "webapp:\n  enabled: true\n  host: 0.0.0.0\n  port: 9555\n")
    state = SimpleNamespace(ui_surface="webapp", auth_required=True)
    runner.session_api.app = SimpleNamespace(web=SimpleNamespace(_SESSION_TOKEN="s" * 43, app=SimpleNamespace(state=state)))
    result = _ask(runner, "webapp-launch", params={"profile": "work"})["result"]
    assert result == {"url": "http://127.0.0.1:9555/?profile=work", "private": False,
                      "expires_in_seconds": None, "sign_in": True}
    assert not getattr(state, "webapp_window_tickets", None)


LAN = "192.168.1.20"


def _lan_webapp(monkeypatch, tmp_path, *, registered=None, address=LAN, public_url=""):
    """A gateway serving the Webapp on every interface (``webapp.host: 0.0.0.0``) behind the sign-in
    gate, this machine's LAN address being *address*, after Web Access Start registered Nous sign-in
    for the callback *registered* (None: never)."""
    from hermes_cli import webapp_access

    config = "webapp:\n  enabled: true\n  host: 0.0.0.0\n  port: 9555\n"
    runner, _, home = _start_runtime_api(monkeypatch, tmp_path, config)
    if registered is not None:
        record = home / "webapp" / "sign-in-registration.json"
        record.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        record.write_text(json.dumps({"portal": "https://portal.example", "client_id": "agent:x",
                                      "callback": registered}) + "\n", encoding="utf-8")
        record.chmod(0o600)
    monkeypatch.setattr(webapp_access, "lan_address", lambda: address)
    state = SimpleNamespace(ui_surface="webapp", auth_required=True, **(
        {"listener_public_url": public_url} if public_url else {}))
    runner.session_api.app = SimpleNamespace(web=SimpleNamespace(_SESSION_TOKEN="s" * 43, app=SimpleNamespace(state=state)))
    return runner


def test_on_the_lan_the_launcher_opens_the_address_web_access_registered_for_sign_in(monkeypatch, tmp_path):
    """Nous sign-in returns to the callback Web Access Start registered with the Portal, the LAN
    address's. Opened at the loopback address, sign-in would send a callback the Portal never saw,
    so the launcher opens the registered address."""
    runner = _lan_webapp(monkeypatch, tmp_path, registered=f"http://{LAN}:9555/auth/callback")
    result = _ask(runner, "webapp-launch", params={"profile": "work"})["result"]
    assert result == {"url": f"http://{LAN}:9555/?profile=work", "private": False,
                      "expires_in_seconds": None, "sign_in": True}


@pytest.mark.parametrize(("registered", "address"), [
    (None, LAN),  # set up by hand: a password provider, or a client registered for this address
    (f"http://{LAN}:9120/auth/callback", LAN),  # registered before webapp.port changed
    (f"http://{LAN}:9555/auth/callback", "192.168.1.77"),  # this machine's LAN address changed since
    ("https://webapp.example.net/auth/callback", LAN),  # registered for a public URL no longer set
])
def test_on_the_lan_without_a_current_registration_the_launcher_opens_the_loopback_address(
        monkeypatch, tmp_path, registered, address):
    runner = _lan_webapp(monkeypatch, tmp_path, registered=registered, address=address)
    assert _ask(runner, "webapp-launch")["result"]["url"] == "http://127.0.0.1:9555/"


def test_the_webapps_own_public_url_wins_over_a_lan_registration(monkeypatch, tmp_path):
    runner = _lan_webapp(monkeypatch, tmp_path, registered=f"http://{LAN}:9555/auth/callback",
                         public_url="https://webapp.example.net")
    assert _ask(runner, "webapp-launch")["result"]["url"] == "https://webapp.example.net/"


@pytest.mark.parametrize(("public_url", "params", "opened"), [
    ("https://webapp.example.net", {}, "https://webapp.example.net/"),
    ("https://example.net:8443/hermes", {"profile": "work"}, "https://example.net:8443/hermes/?profile=work"),
])
def test_behind_its_own_public_url_the_launcher_opens_that_url_to_sign_in(monkeypatch, tmp_path, public_url,
                                                                          params, opened):
    # Sign-in keeps its state cookie on the address the browser opened and returns to a callback on
    # webapp.public_url, so the launcher opens that URL, path prefix kept; a sign-in started on the
    # loopback address could never finish. Native clients keep the loopback address.
    runner, _, _ = _start_runtime_api(
        monkeypatch, tmp_path, f"webapp:\n  enabled: true\n  port: 9555\n  public_url: {public_url}\n")
    # The gateway's dashboard mount records webapp.public_url as the listener's own public URL.
    state = SimpleNamespace(ui_surface="webapp", auth_required=True, listener_public_url=public_url)
    runner.session_api.app = SimpleNamespace(web=SimpleNamespace(_SESSION_TOKEN="s" * 43, app=SimpleNamespace(state=state)))
    result = _ask(runner, "webapp-launch", params=params)["result"]
    assert result == {"url": opened, "private": False, "expires_in_seconds": None, "sign_in": True}
    assert runner.webapp_status["url"] == "http://127.0.0.1:9555"


def _serving_runner(monkeypatch, tmp_path):
    runner, _, _ = _start_runtime_api(monkeypatch, tmp_path, "webapp:\n  enabled: true\n  port: 9555\n")
    # The gateway mounts the dashboard app as the Webapp surface on a private (loopback) listener.
    web = SimpleNamespace(_SESSION_TOKEN="s" * 43, app=SimpleNamespace(state=SimpleNamespace(ui_surface="webapp")))
    runner.session_api.app = SimpleNamespace(web=web)
    return runner


def test_a_launch_link_for_a_named_profile_opens_that_profile(monkeypatch, tmp_path):
    from urllib.parse import parse_qs
    runner = _serving_runner(monkeypatch, tmp_path)
    for checked in (True, False):  # Linux and Windows; macOS, the BSDs and WSL
        webapp.register_control_verbs(runner, tmp_path / "home", links_checked=checked)
        url = _ask(runner, "webapp-launch", params={"profile": "work"})["result"]["url"]
        page, fragment = url.split("#", 1)
        assert page == "http://127.0.0.1:9555/webapp/launch"
        assert parse_qs(fragment)["query"] == ["?profile=work"]


@pytest.mark.parametrize("profile", ["../etc", "a b", "", "x" * 65, 7])
def test_a_launch_link_refuses_a_malformed_profile_name(monkeypatch, tmp_path, profile):
    runner = _serving_runner(monkeypatch, tmp_path)
    refused = _ask(runner, "webapp-launch", params={"profile": profile})
    assert refused["ok"] is False and "profile" in refused["error"]


@pytest.mark.parametrize("refusal", [
    # The dashboard's auth gate refuses a gated bind with no sign-in provider by exiting.
    SystemExit("Refusing to bind dashboard to 0.0.0.0 — the auth gate engages on non-loopback binds "
               "(0.0.0.0), but no auth providers are registered."),
    RuntimeError("gateway API lifespan startup failed"),
])
def test_a_webapp_listener_that_cannot_start_still_leaves_native_clients_a_listener(monkeypatch, tmp_path, refusal):
    runner, calls, _ = _start_runtime_api(
        monkeypatch, tmp_path, "webapp:\n  enabled: true\n  host: 0.0.0.0\n", refuse=refusal)
    assert [(host, port) for host, port, _ in calls] == [("0.0.0.0", webapp.DEFAULT_PORT), ("127.0.0.1", 0)]
    assert runner.webapp_status["state"] == "unavailable"
    assert str(refusal) in runner.webapp_status["reason"]
    assert runner.session_runtime_descriptor["api_origin"] == "http://127.0.0.1:50001"


def test_cancelling_the_gateway_while_the_webapp_listener_starts_cancels_it(monkeypatch, tmp_path):
    with pytest.raises(asyncio.CancelledError):
        _start_runtime_api(monkeypatch, tmp_path, "webapp:\n  enabled: true\n", refuse=asyncio.CancelledError())


def _reuse_address(port: int) -> bool:
    import socket

    from gateway import run_api
    with run_api._listener_socket(socket.AF_INET, socket.SOCK_STREAM, 0, port) as listener:
        return bool(listener.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR))


@pytest.mark.platforms("posix")
def test_on_posix_the_webapps_fixed_port_binds_again_at_once():
    # A restarted gateway must take the Webapp's port back while the old connections sit in TIME_WAIT.
    assert _reuse_address(webapp.DEFAULT_PORT) is True
    assert _reuse_address(0) is False  # main's private listener


@pytest.mark.platforms("windows")
def test_on_windows_the_webapps_fixed_port_never_lets_another_process_bind_it():
    # On Windows SO_REUSEADDR would let another process bind the port, so it stays off.
    assert _reuse_address(webapp.DEFAULT_PORT) is False
    assert _reuse_address(0) is False  # main's private listener


def test_a_dual_stack_listener_takes_ipv4_too():
    """The ``::`` wildcard means every address, IPv4 included, on every platform: Windows otherwise
    binds IPv6 only, so IPv4 clients (a LAN browser at http://192.168.x.y:<port>) never reach the
    Webapp, and its exclusive hold on the port leaves 0.0.0.0 free for another program."""
    import socket

    from gateway import run_api
    with run_api._listener_socket(socket.AF_INET6, socket.SOCK_STREAM, 0, webapp.DEFAULT_PORT,
                                  dual_stack=True) as wildcard:
        assert wildcard.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 0


@pytest.mark.parametrize(("host", "dual_stack"), [("::", True), ("::1", False), ("127.0.0.1", False)])
def test_the_webapp_listener_is_dual_stack_only_on_the_ipv6_wildcard(monkeypatch, host, dual_stack):
    """Only a ``::`` bind takes IPv4 as well; ``::1`` stays the IPv6 loopback address it names."""
    from gateway import run_api

    asked = []

    class Stop(Exception):
        pass

    def listener_socket(family, kind, proto, port, **options):
        asked.append(options.get("dual_stack", False))
        raise Stop

    monkeypatch.setattr(run_api, "_listener_socket", listener_socket)
    with pytest.raises(Stop):
        asyncio.run(run_api.start_gateway_api(SimpleNamespace(), host=host, port=webapp.DEFAULT_PORT,
                                              webapp=webapp.WebappMount(dist=None, home=None)))
    assert asked == [dual_stack]



def test_a_fixed_port_binds_again_while_the_last_gateways_connections_linger():
    # The server closing first leaves its side of the connection in TIME_WAIT on the fixed port.
    import socket

    from gateway import run_api
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    first = run_api._listener_socket(socket.AF_INET, socket.SOCK_STREAM, 0, port)
    first.bind(("127.0.0.1", port))
    first.listen()
    client = socket.create_connection(("127.0.0.1", port))
    served, _ = first.accept()
    served.close()
    client.recv(1)
    client.close()
    first.close()
    with run_api._listener_socket(socket.AF_INET, socket.SOCK_STREAM, 0, port) as second:
        second.bind(("127.0.0.1", port))
