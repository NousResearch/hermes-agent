"""Least-privilege system scope must read limits without sudo on both attempts."""

import subprocess

import pytest

from hermes_cli import update_cmd_fleet as fleet


@pytest.mark.parametrize("retry", [False, True])
def test_unit_restart_reads_limits_without_management_privileges(monkeypatch, retry):
    scope_cmd = ["systemctl", "--no-ask-password"]
    manage_cmd = ["sudo", "-n", "systemctl", "--no-ask-password"]
    name = "hermes-serve-test"
    calls = []

    def systemctl(cmd, *, timeout):
        calls.append((cmd, timeout))
        if "show" in cmd:
            if cmd[:len(manage_cmd)] == manage_cmd:
                return subprocess.CompletedProcess(cmd, 1, "", "sudo: show not allowed")
            return subprocess.CompletedProcess(cmd, 0, "TimeoutStopUSec=30min\nTimeoutStartUSec=90s", "")
        if "restart" in cmd and timeout < 300:
            raise subprocess.TimeoutExpired(cmd, timeout)
        return subprocess.CompletedProcess(cmd, 0, "active", "")

    monkeypatch.setattr(fleet, "_systemctl", systemctl)
    waits = iter([False, True] if retry else [True])
    monkeypatch.setattr(fleet, "_wait_for_service_active", lambda *a, **kw: next(waits))
    restarted, failed = [], []
    fleet._restart_one_systemd_gateway_unit(
        name, scope="system", scope_cmd=scope_cmd, drain_budget=45,
        _manage_cmd_cache={"system": manage_cmd},
        restarted_services=restarted, failed_or_stale_units=failed,
    )
    assert restarted == [name]
    assert failed == []
    restart_calls = [(cmd, timeout) for cmd, timeout in calls if "restart" in cmd]
    assert len(restart_calls) == (2 if retry else 1)
    assert all(cmd == manage_cmd + ["restart", name] and timeout == 1905 for cmd, timeout in restart_calls)
    assert all(cmd[:len(scope_cmd)] == scope_cmd for cmd, _ in calls if "show" in cmd)
