"""``webapp.host`` may be one literal IP address of this machine (a LAN or Tailscale address), as
``hermes dashboard --host`` took it, so a gateway that takes over an old backend keeps its address.

Such a bind is gated by sign-in exactly like a wildcard bind, never treated as loopback. The listener
also binds loopback on the same port, behind the same sign-in, so native clients (Desktop accepts only
``127.0.0.1``/``[::1]`` origins) keep attaching. Hostnames stay refused, and an address no interface
holds leaves the Webapp unavailable with a reason instead of taking the gateway down.
"""
import ipaddress
from argparse import Namespace
from urllib.parse import urlsplit

import pytest

from gateway import webapp_listener as webapp
from tests.gateway.fixtures.webapp_fallback_peer import free_port, run


def _home(tmp_path, config):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(config, encoding="utf-8")
    return home


@pytest.mark.parametrize(("host", "bind_host", "companion"), [
    ("192.168.1.20", "192.168.1.20", "127.0.0.1"),
    ("100.101.102.103", "100.101.102.103", "127.0.0.1"),
    ("169.254.10.20", "169.254.10.20", "127.0.0.1"),  # IPv4 link-local is an interface address too
    ("'fd7a:115c:a1e0::0005'", "fd7a:115c:a1e0::5", "::1"),
    ("'[fd7a:115c:a1e0::5]'", "fd7a:115c:a1e0::5", "::1"),
])
def test_one_address_of_this_machine_is_a_gated_bind_with_a_loopback_dial(tmp_path, host, bind_host, companion):
    from hermes_cli.web_server_boundary import should_require_auth

    settings = webapp.webapp_settings(_home(tmp_path, f"webapp:\n  enabled: true\n  host: {host}\n  port: 9119\n"))
    assert settings.enabled and settings.bind == (bind_host, 9119), settings.reason
    assert should_require_auth(settings.host), "a specific address is never treated as loopback"
    assert webapp.loopback_companion(settings.host) == companion
    assert webapp.dial_host(settings.host) == companion


@pytest.mark.parametrize("host", [
    "hermes.example.net", "my-laptop", "127.0.0.2", "'::ffff:192.168.1.20'", "224.0.0.1", "'fe80::1%eth0'",
    "255.255.255.255", "240.0.0.1", "'ff02::1'", "'fe80::1'", "0.0.0.1",
])
def test_names_and_other_odd_addresses_keep_the_webapp_off_and_say_what_works(tmp_path, host):
    settings = webapp.webapp_settings(_home(tmp_path, f"webapp:\n  enabled: true\n  host: {host}\n"))
    assert settings.bind == ("127.0.0.1", 0)
    assert f"webapp.host must be {webapp.HOST_CHOICES}" in settings.reason


@pytest.mark.parametrize(("bind", "companion"), [
    ("127.0.0.1", None), ("::1", None), ("localhost", None), ("0.0.0.0", None), ("::", None),
])
def test_loopback_and_wildcard_binds_get_no_second_socket(bind, companion):
    """Control: every bind main accepts keeps its one socket and its dial address."""
    assert webapp.loopback_companion(bind) is companion
    assert webapp.bind_origin(bind, 9119) is None


def _this_machines_address():
    from hermes_cli.webapp_access import lan_address
    return lan_address()


def test_a_gateway_bound_to_its_own_address_serves_it_behind_sign_in_and_native_clients_over_loopback(tmp_path):
    from hermes_cli.gateway_runtime import _endpoint

    address = _this_machines_address()
    if address is None:
        pytest.skip("this machine has no non-loopback IPv4 address")
    port = free_port()
    report = run(tmp_path, {"webapp": {"enabled": True, "host": address, "port": port}, "basic_auth": True,
                            "probe_bind_host": True, "probe_host": "rebound.example.net"})
    assert report["start_error"] is None, report.get("start_traceback")
    assert report["status"] == {"state": "serving", "url": f"http://127.0.0.1:{port}", "reason": None,
                                "address": f"http://{address}:{port}"}
    assert report["state"]["auth_required"] is True
    # The address answers, and its API needs sign-in like a wildcard bind's; so does the loopback socket.
    assert report["bind_host_status_code"] == 200
    assert report["bind_host_sessions_code"] == 401 and report["loopback_sessions_code"] == 401
    assert report["probe_host_status_code"] == 400, "the DNS-rebinding Host check still refuses other names"
    # Native clients get a loopback origin both Python discovery and Desktop accept, and attach over it.
    origin = report["api_origin"]
    assert origin == f"http://127.0.0.1:{port}" and ipaddress.ip_address(urlsplit(origin).hostname).is_loopback
    assert report["native_session_list"] is True
    home = tmp_path / "state"
    payload = {"runtime_protocol": 1, "served_profiles": [{"home": str(home), "profile_id": "default"}],
               "state": "ready", "capabilities": ["session-authority-v1"], "authority_epoch": 1,
               "instance_id": "webapp-fallback", "api_origin": origin, "supervisor": "none"}
    assert _endpoint(payload, home).state == "ready"
    assert report["launch"]["url"] == f"http://{address}:{port}/" and report["launch"]["sign_in"] is True


def test_an_address_no_interface_holds_leaves_the_webapp_unavailable_and_native_clients_a_listener(tmp_path):
    port = free_port()
    report = run(tmp_path, {"webapp": {"enabled": True, "host": "192.0.2.77", "port": port}, "basic_auth": True,
                            "report_listening": ["127.0.0.1"]})
    assert report["start_error"] is None, report.get("start_traceback")
    assert report["status"]["state"] == "unavailable"
    assert report["status"]["reason"] == (f"could not listen on 192.0.2.77:{port} "
                                          "(the address is not assigned to any network interface on this machine)")
    assert report["listening"] == {"127.0.0.1": False}, "no loopback socket is left on the Webapp's port"
    assert report["native_session_list"] is True


@pytest.mark.platforms("linux")
def test_an_interface_broadcast_is_not_served_as_a_machine_address(tmp_path):
    import socket
    import psutil

    rows = [row for group in psutil.net_if_addrs().values() for row in group if row.family == socket.AF_INET]
    assigned = {row.address for row in rows}
    broadcast = next((row.broadcast for row in rows if row.broadcast and row.broadcast not in assigned), None)
    if broadcast is None:
        pytest.skip("no IPv4 interface with a directed broadcast")
    port = free_port()
    report = run(tmp_path, {"webapp": {"enabled": True, "host": broadcast, "port": port}, "basic_auth": True})
    assert report["status"]["state"] == "unavailable"
    assert "not assigned to any network interface" in report["status"]["reason"]
    assert report["native_session_list"] is True


def test_a_busy_loopback_port_refuses_the_address_too_and_leaves_nothing_listening_there(tmp_path):
    address = _this_machines_address()
    if address is None:
        pytest.skip("this machine has no non-loopback IPv4 address")
    port = free_port()
    report = run(tmp_path, {"webapp": {"enabled": True, "host": address, "port": port}, "basic_auth": True,
                            "hold_port": True, "report_listening": [address]})
    assert report["start_error"] is None, report.get("start_traceback")
    assert (report["status"]["state"], report["status"]["url"]) == ("unavailable", None)
    assert report["status"]["reason"].startswith(f"could not listen on 127.0.0.1:{port} ("), "names the loopback socket"
    assert report["listening"] == {address: False}, "the address is released when its loopback socket cannot bind"
    assert report["native_session_list"] is True


def test_without_a_sign_in_provider_the_address_is_refused_like_a_wildcard_and_both_sockets_closed(tmp_path):
    address = _this_machines_address()
    if address is None:
        pytest.skip("this machine has no non-loopback IPv4 address")
    port = free_port()
    report = run(tmp_path, {"webapp": {"enabled": True, "host": address, "port": port},
                            "report_listening": [address, "127.0.0.1"]})
    assert report["start_error"] is None, report.get("start_traceback")
    assert report["status"]["state"] == "unavailable"
    assert report["status"]["reason"].startswith(f"Refusing to bind dashboard to {address}")
    assert report["listening"] == {address: False, "127.0.0.1": False}
    assert report["native_session_list"] is True


SPECIFIC_SERVING = {"state": "serving", "url": "http://127.0.0.1:9119", "reason": None,
                    "address": "http://100.101.102.103:9119"}


def test_status_names_the_address_the_running_gateway_is_bound_to(monkeypatch, tmp_path, capsys):
    from hermes_cli import webapp_launcher as launcher

    _status(monkeypatch, tmp_path, "webapp:\n  enabled: true\n  host: 100.101.102.103\n  port: 9119\n",
            SPECIFIC_SERVING)
    out = capsys.readouterr().out
    assert out.splitlines()[0] == ("Serving at http://100.101.102.103:9119 "
                                   "(this machine's own clients use http://127.0.0.1:9119)")
    assert launcher.PLAIN_HTTP_NOTE in out and "after a restart" not in out


@pytest.mark.parametrize(("config", "live", "pending"), [
    # The config names another address than the one the gateway bound when it started.
    ("webapp:\n  enabled: true\n  host: 100.64.1.5\n  port: 9119\n", SPECIFIC_SERVING, "100.64.1.5"),
    # The gateway still serves the address; the config moved back to a wildcard.
    ("webapp:\n  enabled: true\n  host: 0.0.0.0\n  port: 9119\n", SPECIFIC_SERVING, "0.0.0.0"),
    # The gateway serves a wildcard bind; the config now names one address.
    ("webapp:\n  enabled: true\n  host: 100.64.1.5\n  port: 9119\n",
     {"state": "serving", "url": "http://127.0.0.1:9119", "reason": None}, "100.64.1.5"),
])
def test_status_reports_the_live_bind_and_names_a_config_change_pending_a_restart(
        monkeypatch, tmp_path, capsys, config, live, pending):
    home = _status(monkeypatch, tmp_path, config, live)
    lines = capsys.readouterr().out.splitlines()
    served = live.get("address")
    assert lines[0] == (f"Serving at {served} (this machine's own clients use {live['url']})" if served
                        else f"Serving at {live['url']}")
    assert lines[-1] == (f"{home / 'config.yaml'} now sets webapp.host: {pending}: the gateway serves it after a "
                         "restart (hermes gateway restart).")


def _status(monkeypatch, tmp_path, config, status):
    from hermes_cli import webapp_launcher as launcher

    home = _home(tmp_path, config)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(launcher, "_control", lambda _home, verb, params=None: {"ok": True, "result": status})
    args = Namespace(no_open=False, status=True, sign_out=False, build_only=False, force_build=False, skip_build=False)
    assert launcher.cmd_webapp(args) == 0
    return home


BOUND = "100.64.1.5"


def _specific_webapp(monkeypatch, tmp_path, *, registered=None, public_url=""):
    """A gateway serving the Webapp on *BOUND* (and loopback) behind sign-in, after Web Access Start
    registered Nous sign-in for the callback *registered* (None: never); the listener stood in."""
    import asyncio
    import json
    from types import SimpleNamespace

    from gateway import run_api, run_runtime
    from gateway.control_socket import GatewayControlServer
    from hermes_cli import webapp_access

    home = _home(tmp_path, f"webapp:\n  enabled: true\n  host: {BOUND}\n  port: 9555\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    async def start(runner, *, host="127.0.0.1", port=0, webapp=None):
        return SimpleNamespace(api_origin=f"http://127.0.0.1:{port or 50001}")

    monkeypatch.setattr(run_api, "start_gateway_api", start)
    runner = SimpleNamespace(session_runtime_descriptor={}, session_authorities=[],
                             session_control_server=GatewayControlServer(home))
    asyncio.run(run_runtime.start_gateway_runtime_api(runner))
    if registered is not None:
        record = home / "webapp" / "sign-in-registration.json"
        record.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        record.write_text(json.dumps({"portal": "https://portal.example", "client_id": "agent:x",
                                      "callback": registered}) + "\n", encoding="utf-8")
        record.chmod(0o600)
    monkeypatch.setattr(webapp_access, "lan_address", lambda: "192.168.1.20")  # the machine's primary LAN address
    state = SimpleNamespace(ui_surface="webapp", auth_required=True, bound_host=BOUND,
                            **({"listener_public_url": public_url} if public_url else {}))
    runner.session_api.app = SimpleNamespace(web=SimpleNamespace(_SESSION_TOKEN="s" * 43, app=SimpleNamespace(state=state)))
    return runner


def _launch(runner):
    import json
    raw = json.dumps({"protocol": 1, "verb": "webapp-launch", "id": 1, "params": {}}).encode()
    return json.loads(runner.session_control_server.handle_request_line(raw, "same-user-peer"))["result"]


def test_the_gateway_reports_the_specific_address_it_serves(monkeypatch, tmp_path):
    runner = _specific_webapp(monkeypatch, tmp_path)
    assert runner.webapp_status == {"state": "serving", "url": "http://127.0.0.1:9555", "reason": None,
                                    "address": f"http://{BOUND}:9555"}


@pytest.mark.parametrize("registered", [None, f"http://{BOUND}:9555/auth/callback"])
def test_a_specific_bind_opens_its_own_address_to_sign_in(monkeypatch, tmp_path, registered):
    runner = _specific_webapp(monkeypatch, tmp_path, registered=registered)
    assert _launch(runner) == {"url": f"http://{BOUND}:9555/", "private": False, "expires_in_seconds": None,
                               "sign_in": True}


@pytest.mark.parametrize(("registered", "changed"), [
    # Web Access registered the LAN address; the gateway is bound to another address on the same port.
    ("http://192.168.1.20:9555/auth/callback", "webapp.host"),
    (f"http://{BOUND}:9120/auth/callback", "webapp.port"),  # registered before webapp.port changed
    ("http://192.168.1.20:9120/auth/callback", "webapp.host and webapp.port"),  # both differ
])
def test_a_registration_for_another_address_never_sends_the_browser_there_and_says_why(
        monkeypatch, tmp_path, registered, changed):
    runner = _specific_webapp(monkeypatch, tmp_path, registered=registered)
    link = _launch(runner)
    registered_origin = registered.removesuffix("/auth/callback")
    assert link["url"] == f"http://{BOUND}:9555/"
    assert link["note"] == (
        f"Nous sign-in that Settings → Web Access registered returns to {registered_origin}, but the Webapp now "
        f"listens on http://{BOUND}:9555, so Nous sign-in cannot finish there. Sign in with a password provider, "
        f"or set {changed} back to that address and restart the gateway.")


@pytest.mark.parametrize(("registered", "note"), [
    (f"http://{BOUND}/auth/callback", None),  # the same address: no port is port 80
    ("http://192.168.1.20/auth/callback",
     "Nous sign-in that Settings → Web Access registered returns to http://192.168.1.20:80, but the Webapp now "
     f"listens on http://{BOUND}:80, so Nous sign-in cannot finish there. Sign in with a password provider, "
     "or set webapp.host back to that address and restart the gateway."),
])
def test_a_registered_callback_with_no_port_is_port_80(tmp_path, registered, note):
    import json
    from hermes_cli.webapp_access import sign_in_address

    home = _home(tmp_path, f"webapp:\n  enabled: true\n  host: {BOUND}\n  port: 80\n")
    record = home / "webapp" / "sign-in-registration.json"
    record.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    record.write_text(json.dumps({"portal": "https://portal.example", "client_id": "agent:x", "callback": registered}))
    assert sign_in_address(home, BOUND, 80, "") == (f"http://{BOUND}:80", note)


def test_a_specific_binds_own_public_url_still_wins(monkeypatch, tmp_path):
    runner = _specific_webapp(monkeypatch, tmp_path, registered="http://192.168.1.20:9555/auth/callback",
                              public_url="https://webapp.example.net")
    assert _launch(runner) == {"url": "https://webapp.example.net/", "private": False, "expires_in_seconds": None,
                               "sign_in": True}


def test_specific_bind_reports_an_own_client_still_registered_for_https(monkeypatch, tmp_path):
    import json
    from hermes_cli.webapp_access import sign_in_client

    public = "https://proxy.example/work"
    runner = _specific_webapp(monkeypatch, tmp_path, registered=public + "/auth/callback")
    home = tmp_path / "home"
    record = home / "webapp" / "sign-in-registration.json"
    record.write_text(json.dumps(dict(json.loads(record.read_text()), own_client=True)))
    assert sign_in_client(home) is not None
    link = _launch(runner)
    assert link["url"] == f"http://{BOUND}:9555/"
    assert "webapp.public_url" in link.get("note", "")


def test_hermes_webapp_prints_the_note_before_opening_the_address(monkeypatch, tmp_path, capsys):
    from hermes_cli import webapp_launcher as launcher

    link = {"url": f"http://{BOUND}:9555/", "private": False, "expires_in_seconds": None, "sign_in": True,
            "note": "Nous sign-in that Settings → Web Access registered returns to elsewhere."}
    home = _home(tmp_path, f"webapp:\n  enabled: true\n  host: {BOUND}\n  port: 9555\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(launcher, "_ensure", lambda _target: None)
    monkeypatch.setattr(launcher, "_build", lambda _args: None)
    monkeypatch.setattr(launcher, "_control", lambda _home, verb, params=None: {"ok": True, "result": link})
    monkeypatch.setattr(launcher, "_open", lambda url: False)
    args = Namespace(no_open=True, status=False, sign_out=False, build_only=False, force_build=False, skip_build=True)
    assert launcher.cmd_webapp(args) == 0
    out = capsys.readouterr().out
    assert link["note"] in out
    assert out.index(link["note"]) < out.index(f"sign in with your dashboard account: {link['url']}")


def test_web_access_counts_a_specific_address_as_the_lan(tmp_path):
    from hermes_cli.webapp_access import _mode
    for host, mode in (("100.101.102.103", "lan"), ("0.0.0.0", "lan"), ("127.0.0.1", "local")):
        settings = webapp.webapp_settings(_home(tmp_path / host, f"webapp:\n  enabled: true\n  host: {host}\n"))
        assert _mode(settings) == mode


def test_only_a_listener_with_a_loopback_socket_accepts_a_loopback_host_beside_its_address():
    """Control: ``hermes dashboard --host <address>`` (no loopback socket) still refuses a loopback Host."""
    from hermes_cli.web_server_boundary import _is_accepted_host
    assert not _is_accepted_host("127.0.0.1:9119", "100.101.102.103")
    assert _is_accepted_host("127.0.0.1:9119", "100.101.102.103", frozenset(), True)
    assert _is_accepted_host("100.101.102.103:9119", "100.101.102.103", frozenset(), True)
    assert not _is_accepted_host("rebound.example.net", "100.101.102.103", frozenset(), True)
