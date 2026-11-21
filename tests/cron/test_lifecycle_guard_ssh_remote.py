"""A lifecycle command sent over ssh to ANOTHER host cannot SIGTERM this gateway, so it is allowed;
anything that could still land on this host stays blocked."""

import socket

import pytest

from cron import lifecycle_guard_ssh as ssh_guard
from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script as guard

RESTART = "systemctl --user restart hermes-gateway"


@pytest.fixture
def remote_host(monkeypatch):
    config = {"hostname": "192.0.2.10", "proxycommand": "none", "permitlocalcommand": "no"}
    monkeypatch.setattr(ssh_guard, "_ssh_config", lambda options, dest: dict(config))
    return config


def test_remote_ssh_restart_is_allowed(remote_host):
    assert guard(f"ssh skuser01@chiap04 '{RESTART}'") is False
    assert guard(f"timeout 90 ssh -o BatchMode=yes -p 2222 chiap04 'cp a b && {RESTART}; sleep 5'") is False


@pytest.mark.parametrize("command", [
    f"ssh chiap04 'true'; {RESTART}",                 # second segment runs locally
    f"ssh chiap04 'true' && {RESTART}",
    f"ssh chiap04 echo $({RESTART})",                 # substitution runs before ssh
    f"ssh chiap04 'ssh noroc2027 {RESTART}'",          # nested hop may come back here
])
def test_local_parts_stay_blocked(remote_host, command):
    assert guard(command) is True


def test_proxycommand_stays_blocked(remote_host):
    remote_host["proxycommand"] = f"sh -c '{RESTART}'"
    assert guard(f"ssh chiap04 '{RESTART}'") is True


def test_local_destination_stays_blocked(monkeypatch):
    monkeypatch.setattr(ssh_guard, "_ssh_config", lambda o, d: {"hostname": "127.0.0.1"})
    assert guard(f"ssh me@127.0.0.1 '{RESTART}'") is True


def test_unresolvable_config_stays_blocked(monkeypatch):
    monkeypatch.setattr(ssh_guard, "_ssh_config", lambda o, d: None)
    assert guard(f"ssh chiap04 '{RESTART}'") is True


def test_local_host_detection():
    assert ssh_guard._is_local_host("localhost") is True
    assert ssh_guard._is_local_host("127.0.0.1") is True
    assert ssh_guard._is_local_host(socket.gethostname()) is True
    assert ssh_guard._is_local_host("192.0.2.10") is False  # TEST-NET-1, never bound locally
