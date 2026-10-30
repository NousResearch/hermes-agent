"""A ``webapp:`` block that asks for the Webapp but cannot be used says why.

A malformed block keeps the Webapp off (the listener stays private), and the reason reaches the user:
``webapp_settings`` returns it, the gateway reports the Webapp unavailable with it, and ``hermes webapp``
prints it instead of telling the user to set ``webapp.enabled: true`` when it is already set.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway import webapp_listener as webapp

MALFORMED = [
    ("webapp: yes\n", "webapp in"),
    ("webapp:\n  enabled: 'true'\n", "webapp.enabled"),
    ("webapp:\n  enabled: true\n  host: 192.168.1.20\n", "webapp.host"),
    ("webapp:\n  enabled: true\n  host: ''\n", "webapp.host"),
    ("webapp:\n  enabled: true\n  port: 70000\n", "webapp.port"),
    ("webapp:\n  enabled: true\n  port: http\n", "webapp.port"),
    ("webapp:\n  enabled: true\n  public_url: hermes.example.net\n", "webapp.public_url"),
]


def _home(tmp_path, config):
    home = tmp_path / "home"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(config, encoding="utf-8")
    return home


@pytest.mark.parametrize(("block", "key"), MALFORMED)
def test_a_malformed_webapp_block_says_why_it_keeps_the_webapp_off(tmp_path, block, key):
    home = _home(tmp_path, block)
    settings = webapp.webapp_settings(home)
    assert (settings.enabled, settings.bind) == (False, ("127.0.0.1", 0))
    assert key in settings.reason and str(home / "config.yaml") in settings.reason


@pytest.mark.parametrize("block", ["", "webapp:\n  enabled: false\n  port: 70000\n", "webapp:\n  enabled: true\n"])
def test_a_webapp_block_that_is_off_or_valid_has_no_reason(tmp_path, block):
    assert webapp.webapp_settings(_home(tmp_path, block)).reason is None


def test_a_webapp_block_that_is_off_still_reads_its_address(tmp_path):
    """Web Access shows and registers the address the gateway will bind, so a block that is off, or
    asks for the Webapp but cannot be used, still reads its valid host, port and public URL (an
    invalid one reads as the default). The listener stays main's private one either way."""
    off = webapp.webapp_settings(_home(tmp_path / "off", (
        "webapp:\n  enabled: false\n  host: 0.0.0.0\n  port: 9333\n  public_url: https://webapp.example.net/\n")))
    assert (off.enabled, off.host, off.port, off.public_url, off.reason) == (
        False, "0.0.0.0", 9333, "https://webapp.example.net", None)
    assert off.bind == ("127.0.0.1", 0)
    unusable = webapp.webapp_settings(_home(tmp_path / "unusable", (
        "webapp:\n  enabled: true\n  host: 192.168.1.20\n  port: 9333\n  public_url: https://u:p@x.example\n")))
    assert (unusable.enabled, unusable.host, unusable.port, unusable.public_url) == (False, "127.0.0.1", 9333, "")
    assert unusable.bind == ("127.0.0.1", 0)
    assert "webapp.host" in unusable.reason


@pytest.mark.parametrize(("host", "bind_host"), [("'[::1]'", "::1"), ("'[::]'", "::"), ("' ::1 '", "::1")])
def test_a_bracketed_ipv6_host_is_read_as_the_address(tmp_path, host, bind_host):
    # Brackets belong to URLs; the bind address and the address native clients dial carry none.
    settings = webapp.webapp_settings(_home(tmp_path, f"webapp:\n  enabled: true\n  host: {host}\n"))
    assert settings.bind == (bind_host, webapp.DEFAULT_PORT)
    assert webapp.dial_origin(settings.host, 9120) == "http://[::1]:9120"


@pytest.mark.parametrize("host", ["'[127.0.0.1]'", "'[localhost]'", "'[::1'", "'::1]'", "'[[::1]]'"])
def test_brackets_around_anything_but_an_ipv6_address_keep_the_webapp_off(tmp_path, host):
    settings = webapp.webapp_settings(_home(tmp_path, f"webapp:\n  enabled: true\n  host: {host}\n"))
    assert settings.enabled is False and "webapp.host" in settings.reason


@pytest.mark.parametrize(("host", "bind_host"), [
    ("127.0.0.1", "127.0.0.1"),
    ("'::1'", "::1"),
    ("localhost", "localhost"),
    ("LocalHost", "localhost"),
    ("'0:0:0:0:0:0:0:1'", "::1"),
    ("'0000::0001'", "::1"),
    ("0.0.0.0", "0.0.0.0"),
    ("'0:0:0:0:0:0:0:0'", "::"),
    ("127.0.0.2", None),
    ("127.255.255.254", None),
    ("'::ffff:127.0.0.1'", None),
])
def test_the_webapp_and_its_sign_in_gate_agree_on_which_hosts_stay_on_this_machine(tmp_path, host, bind_host):
    """A webapp.host the settings accept as local-only is one the listener's sign-in gate leaves
    open, and only a wildcard bind needs sign-in. Another spelling of 127.0.0.1, ::1 or localhost
    binds that address; any other loopback address, which the gate would treat as public, keeps the
    Webapp off and says which hosts work."""
    from hermes_cli.web_server_boundary import should_require_auth

    settings = webapp.webapp_settings(_home(tmp_path, f"webapp:\n  enabled: true\n  host: {host}\n"))
    if bind_host is None:
        assert settings.bind == ("127.0.0.1", 0)
        assert "webapp.host must be 127.0.0.1, ::1, localhost, 0.0.0.0 or ::" in settings.reason
        return
    assert settings.bind == (bind_host, webapp.DEFAULT_PORT)
    assert should_require_auth(settings.host) is webapp._is_wildcard(settings.host)


def _gateway(monkeypatch, tmp_path, config):
    """The runtime's listener start for *config*, with the listener itself replaced by a stand-in."""
    from gateway import run_api, run_runtime
    from gateway.control_socket import GatewayControlServer

    home = _home(tmp_path, config)
    monkeypatch.setenv("HERMES_HOME", str(home))

    async def start(runner, *, host="127.0.0.1", port=0, webapp=None):
        return SimpleNamespace(api_origin=f"http://127.0.0.1:{port or 50001}")

    monkeypatch.setattr(run_api, "start_gateway_api", start)
    runner = SimpleNamespace(session_runtime_descriptor={}, session_authorities=[],
                             session_control_server=GatewayControlServer(home))
    asyncio.run(run_runtime.start_gateway_runtime_api(runner))
    return runner


def _ask(runner, verb):
    raw = json.dumps({"protocol": 1, "verb": verb, "id": 1, "params": {}}).encode()
    return json.loads(runner.session_control_server.handle_request_line(raw, "same-user-peer"))


def test_the_gateway_reports_a_malformed_webapp_block_as_unavailable_with_its_reason(monkeypatch, tmp_path):
    runner = _gateway(monkeypatch, tmp_path, "webapp:\n  enabled: true\n  port: 70000\n")
    status = _ask(runner, "webapp-status")["result"]
    assert (status["state"], status["url"]) == ("unavailable", None)
    assert "webapp.port" in status["reason"]
    refused = _ask(runner, "webapp-launch")
    assert refused["ok"] is False and "webapp.port" in refused["error"]
    assert "webapp.enabled: true" not in refused["error"]


def test_a_gateway_without_a_webapp_block_still_reports_it_off(monkeypatch, tmp_path):
    runner = _gateway(monkeypatch, tmp_path, "")
    assert _ask(runner, "webapp-status")["result"] == {"state": "off", "url": None, "reason": None}
