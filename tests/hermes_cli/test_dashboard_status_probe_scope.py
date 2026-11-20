"""``hermes dashboard --status`` probes the address it names only when this machine itself can be listening
there without a name lookup: ``localhost``, a loopback or wildcard IP literal, or an address one of its
interfaces holds. Any other ``--host`` (a DNS name, another machine's address, something that does not
parse) behaves as on main: no resolver call, no connection attempt, no exception, exit 0.

The process scan is replaced with an empty one so the output is exactly main's for a host with no
backends; the probe itself and the resolver are the real ones unless a test guards them.
"""
import socket

import pytest

from hermes_cli import dashboard_procs, web_server_port_holder
from hermes_cli.main_dashboard import _report_dashboard_status

NONE_RUNNING = "No hermes dashboard or serve processes running.\n"


@pytest.fixture(autouse=True)
def no_backends(monkeypatch):
    monkeypatch.setattr(dashboard_procs, "_scan_dashboard_processes", lambda **_: [])
    monkeypatch.setattr(dashboard_procs, "_ledger_serve_binds", dict)

    def never_asked(*args, **kwargs):
        pytest.fail(f"this home's gateway was asked: {args}")
    monkeypatch.setattr(web_server_port_holder, "gateway_webapp_summary", never_asked)


@pytest.mark.parametrize("host", ["slow.invalid", "hermes.example.net", "my-laptop", "::ffff:zz", "127.0.0.1.invalid"])
def test_a_name_or_unparsable_host_is_never_resolved(monkeypatch, capsys, host):
    def resolver(*args, **kwargs):
        pytest.fail(f"resolver called: {args}")
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    assert _report_dashboard_status(host, 9119) == 0
    assert capsys.readouterr().out == NONE_RUNNING


def test_a_label_that_cannot_be_idna_encoded_prints_mains_output(capsys):
    assert _report_dashboard_status("a" * 64 + ".invalid", 9119) == 0
    assert capsys.readouterr().out == NONE_RUNNING


@pytest.mark.parametrize("host", ["192.0.2.1", "198.51.100.7", "2001:db8::1", "[2001:db8::1]"])
def test_another_machines_address_gets_no_connection_attempt(monkeypatch, capsys, host):
    attempts = []
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: attempts.append(a) or pytest.fail(str(a)))
    assert _report_dashboard_status(host, 9119) == 0
    assert capsys.readouterr().out == NONE_RUNNING
    assert attempts == []


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "[::1]", "0.0.0.0", "::", "localhost"])
def test_loopback_wildcard_and_localhost_are_probed(host):
    from hermes_cli.main_dashboard import _status_host_is_local
    assert _status_host_is_local(host)


def test_an_address_this_machine_holds_is_probed():
    import psutil

    from hermes_cli.main_dashboard import _status_host_is_local
    held = [row.address.partition("%")[0] for rows in psutil.net_if_addrs().values() for row in rows
            if row.family == socket.AF_INET and not row.address.startswith("127.")]
    if not held:
        pytest.skip("no non-loopback IPv4 interface address")
    assert _status_host_is_local(held[0])
