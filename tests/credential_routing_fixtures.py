"""Optional offline transport accounting for credential-routing acceptance suites.

Load with ``-p tests.credential_routing_fixtures --routing-network-ledger PATH``.
The audit hook is installed before conftest collection and records even caught denials.
Quota observations are stubbed only in tests requesting ``codex_quota_not_restored``.
"""

import json
import os
from pathlib import Path
import sys
import traceback

import pytest

_denials = []
_active_test = None


def install_network_guard(ledger, *, allow_loopback=False):
    path = Path(ledger)

    def audit(event, args):
        if event in {
            "socket.connect", "socket.connect_ex", "socket.sendto", "socket.sendmsg",
            "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
        }:
            record = {"pid": os.getpid(), "event": event, "args": repr(args), "test": _active_test,
                      "trace": traceback.format_stack(limit=20)}
            host = args[1][0] if event in {"socket.connect", "socket.connect_ex"} else args[0]
            allowed = allow_loopback and host in ("127.0.0.1", "::1", "localhost")
            record["allowed_loopback"] = allowed
            with path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if not allowed:
                _denials.append(record)
                raise RuntimeError("Credential-routing tests forbid network access")

    sys.addaudithook(audit)
    with path.open("a") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "event": "guard_installed"}) + "\n")


def pytest_addoption(parser):
    parser.addoption("--routing-network-ledger", default=None)
    parser.addoption("--routing-allow-loopback", action="store_true", default=False)


def pytest_load_initial_conftests(early_config, parser, args):
    ledger = early_config.known_args_namespace.routing_network_ledger
    if ledger:
        install_network_guard(ledger, allow_loopback=early_config.known_args_namespace.routing_allow_loopback)


def pytest_sessionfinish(session, exitstatus):
    if _denials:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter:
            reporter.write_sep("!", f"Network gate failed: {len(_denials)} denied attempts (including caught exceptions)")


def pytest_runtest_setup(item):
    global _active_test
    _active_test = item.nodeid


@pytest.fixture
def codex_quota_not_restored(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth._probe_codex_quota_restored", lambda *a, **kw: False)
    monkeypatch.setattr("hermes_cli.auth_codex._probe_codex_quota_restored", lambda *a, **kw: False)
