"""Every ``webapp.host`` that works on main behaves as it does there: what the gateway serves, what it
reports, the address native clients dial, the Host header it accepts and what ``hermes webapp --status``
and Web Access print. Pinned as literals, so the same file passes against main.

Covered: ``127.0.0.1``, ``::1``, ``localhost``, ``0.0.0.0``, ``::``, a ``webapp:`` block that is off
(also one naming a specific address) and no block at all. A specific address is
``test_webapp_specific_address.py``'s.
"""
import asyncio
import errno
import socket
from argparse import Namespace
from types import SimpleNamespace

import pytest

from gateway import webapp_listener as webapp
from tests.gateway.fixtures.webapp_fallback_peer import free_port, run

HOSTS = ("127.0.0.1", "::1", "localhost", "0.0.0.0", "::")

# What the running gateway reports for an enabled Webapp on each host (``{port}`` is the Webapp's).
SERVED = {
    "127.0.0.1": {
        "api_origin": "http://127.0.0.1:{port}", "auth_required": False,
        "launch_url": "http://127.0.0.1:{port}/webapp/launch", "launch_sign_in": None,
        "status": {"state": "serving", "url": "http://127.0.0.1:{port}", "reason": None},
        "loopback_status_code": 200, "index_status": 200, "native_session_list": True},
    "::1": {
        "api_origin": "http://[::1]:{port}", "auth_required": False,
        "launch_url": "http://[::1]:{port}/webapp/launch", "launch_sign_in": None,
        "status": {"state": "serving", "url": "http://[::1]:{port}", "reason": None},
        "loopback_status_code": 200, "index_status": 200, "native_session_list": True},
    "localhost": {
        "api_origin": "<the address localhost resolves to>", "auth_required": False,
        "launch_url": "http://localhost:{port}/webapp/launch", "launch_sign_in": None,
        "status": {"state": "serving", "url": "http://localhost:{port}", "reason": None},
        "loopback_status_code": 200, "index_status": 200, "native_session_list": True},
    "0.0.0.0": {
        "api_origin": "http://127.0.0.1:{port}", "auth_required": True,
        "launch_url": "http://127.0.0.1:{port}/", "launch_sign_in": True,
        "status": {"state": "serving", "url": "http://127.0.0.1:{port}", "reason": None},
        "loopback_status_code": 200, "index_status": 302, "native_session_list": True},
    "::": {
        "api_origin": "http://[::1]:{port}", "auth_required": True,
        "launch_url": "http://[::1]:{port}/", "launch_sign_in": True,
        "status": {"state": "serving", "url": "http://[::1]:{port}", "reason": None},
        "loopback_status_code": 200, "index_status": 302, "native_session_list": True},
}


def _ipv6_loopback() -> bool:
    try:
        with socket.socket(socket.AF_INET6) as probe:
            probe.bind(("::1", 0))
    except OSError:
        return False
    return True


def _served(tmp_path, webapp_block) -> dict:
    port = webapp_block.get("port", 0) if webapp_block else 0
    report = run(tmp_path, {"webapp": webapp_block, "basic_auth": True})
    assert report["start_error"] is None, report.get("start_traceback")
    launch = report["launch"]
    observed = {
        "api_origin": report["api_origin"], "auth_required": report["state"]["auth_required"],
        "launch_url": launch.get("url", "").split("?")[0].split("#")[0] if "url" in launch else launch,
        "launch_sign_in": launch.get("sign_in"), "status": report["status"],
        "loopback_status_code": report["loopback_status_code"], "index_status": report["index"]["status"],
        "native_session_list": report["native_session_list"]}
    return {key: value.replace(str(port), "{port}") if isinstance(value, str) and port else value
            for key, value in observed.items()} | {
        "status": {key: value.replace(str(port), "{port}") if isinstance(value, str) and port else value
                   for key, value in report["status"].items()}}


def _localhost_origin() -> str:
    """The API origin a ``localhost`` bind advertises: the address the socket bound, in the family
    this machine's resolver gives ``localhost`` first."""
    address = socket.getaddrinfo("localhost", None, type=socket.SOCK_STREAM)[0][4][0]
    return f"http://[{address}]:{{port}}" if ":" in address else f"http://{address}:{{port}}"


@pytest.mark.parametrize("host", HOSTS)
def test_an_enabled_webapp_on_a_host_main_accepts_serves_as_on_main(tmp_path, host):
    if ":" in host and not _ipv6_loopback():
        pytest.skip("this machine has no IPv6 loopback")
    expected = SERVED[host] | ({"api_origin": _localhost_origin()} if host == "localhost" else {})
    assert _served(tmp_path, {"enabled": True, "host": host, "port": free_port()}) == expected


@pytest.mark.parametrize("block", [None, {"enabled": False, "host": "0.0.0.0", "port": 9555},
                                   {"enabled": False, "host": "192.168.1.20", "port": 9555}],
                         ids=["no-block", "off-wildcard", "off-specific-address"])
def test_a_webapp_that_is_off_leaves_mains_private_listener(tmp_path, block):
    report = run(tmp_path, {"webapp": block, "basic_auth": True})
    assert report["start_error"] is None, report.get("start_traceback")
    assert report["status"] == {"state": "off", "url": None, "reason": None}
    assert report["launch"] == {"refused": report["launch"]["refused"]} and report["launch"]["refused"]
    assert report["api_origin"].startswith("http://127.0.0.1:") and not report["api_origin"].endswith(":9555")
    assert report["state"]["auth_required"] is False
    assert report["native_session_list"] is True


def _home(tmp_path, config):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(config, encoding="utf-8")
    return home


SETTINGS = {  # (enabled, host, port, public_url, bind, reason) for "webapp: {enabled: true, host: <host>}"
    "127.0.0.1": (True, "127.0.0.1", 9120, "", ("127.0.0.1", 9120), None),
    "::1": (True, "::1", 9120, "", ("::1", 9120), None),
    "localhost": (True, "localhost", 9120, "", ("localhost", 9120), None),
    "0.0.0.0": (True, "0.0.0.0", 9120, "", ("0.0.0.0", 9120), None),
    "::": (True, "::", 9120, "", ("::", 9120), None),
}


@pytest.mark.parametrize("host", HOSTS)
def test_the_settings_dial_address_and_host_check_of_a_host_main_accepts_are_mains(tmp_path, host):
    from hermes_cli.web_server_boundary import _is_accepted_host, should_require_auth
    from hermes_cli.webapp_access import _mode

    settings = webapp.webapp_settings(_home(tmp_path, f"webapp:\n  enabled: true\n  host: '{host}'\n"))
    assert (settings.enabled, settings.host, settings.port, settings.public_url, settings.bind,
            settings.reason) == SETTINGS[host]
    wildcard = host in {"0.0.0.0", "::"}
    assert _mode(settings) == ("lan" if wildcard else "local")
    assert should_require_auth(settings.host) is wildcard
    assert webapp.dial_host(host) == {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    accepted = {header: _is_accepted_host(header, host) for header in (
        "127.0.0.1:9120", "localhost:9120", "[::1]:9120", "192.168.1.20:9120", "rebound.example.net")}
    assert accepted == (dict.fromkeys(accepted, True) if wildcard else {
        "127.0.0.1:9120": True, "localhost:9120": True, "[::1]:9120": True,
        "192.168.1.20:9120": False, "rebound.example.net": False})


STATUS_OUTPUT = {  # ``hermes webapp --status`` against a gateway serving the Webapp on each host (all of it)
    "127.0.0.1": "Serving at http://127.0.0.1:9120\n",
    "::1": "Serving at http://[::1]:9120\n",
    "localhost": "Serving at http://localhost:9120\n",
    "0.0.0.0": "Serving at http://127.0.0.1:9120\n",
    "::": "Serving at http://[::1]:9120\n",
}


@pytest.mark.parametrize("host", HOSTS)
def test_hermes_webapp_status_prints_what_main_prints(monkeypatch, tmp_path, capsys, host):
    from hermes_cli import webapp_launcher as launcher

    home = _home(tmp_path, f"webapp:\n  enabled: true\n  host: '{host}'\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    url = webapp.dial_origin(host, 9120) if host != "localhost" else "http://localhost:9120"
    monkeypatch.setattr(launcher, "_control", lambda _home, verb, params=None: {
        "ok": True, "result": {"state": "serving", "url": url, "reason": None}})
    args = Namespace(no_open=False, status=True, sign_out=False, build_only=False, force_build=False, skip_build=False)
    assert launcher.cmd_webapp(args) == 0
    assert capsys.readouterr().out == STATUS_OUTPUT[host]


# macOS lets a wildcard bind under SO_REUSEADDR succeed beside the 127.0.0.1 rival (as on main), so the
# wildcard case runs where that bind is refused.
@pytest.mark.parametrize("host", ["127.0.0.1", pytest.param("0.0.0.0", marks=pytest.mark.platforms("linux", "windows"))])
def test_a_busy_webapp_port_leaves_the_webapp_unavailable_and_native_clients_a_listener(tmp_path, host):
    """Real sockets; the reason's wording is the OS's (pinned with a fixed error below)."""
    port = free_port()
    report = run(tmp_path, {"webapp": {"enabled": True, "host": host, "port": port}, "basic_auth": True,
                            "hold_port": True})
    assert (report["status"]["state"], report["status"]["url"]) == ("unavailable", None)
    assert report["status"]["reason"].startswith(f"could not listen on {host}:{port} (")
    assert report["native_session_list"] is True


@pytest.mark.parametrize("error", [
    OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address"),  # ::1 on a machine with IPv6 off
    OSError(errno.EADDRINUSE, "Address already in use"),
], ids=["not-available", "in-use"])
@pytest.mark.parametrize("host", HOSTS)
def test_a_bind_error_gives_mains_reason(monkeypatch, tmp_path, host, error):
    """The reason carries the OS's own words for the error, as on main (a fixed error here)."""
    from gateway import run_api, run_runtime
    from gateway.control_socket import GatewayControlServer

    home = _home(tmp_path, f"webapp:\n  enabled: true\n  host: '{host}'\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    async def start(runner, *, host="127.0.0.1", port=0, webapp=None):
        if webapp is not None:
            raise error
        return SimpleNamespace(api_origin="http://127.0.0.1:50001")

    monkeypatch.setattr(run_api, "start_gateway_api", start)
    runner = SimpleNamespace(session_runtime_descriptor={}, session_authorities=[],
                             session_control_server=GatewayControlServer(home))
    asyncio.run(run_runtime.start_gateway_runtime_api(runner))
    assert runner.webapp_status == {"state": "unavailable", "url": None,
                                    "reason": f"could not listen on {host}:9120 ({error.strerror})"}
