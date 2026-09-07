"""A systemctl client must outwait a legitimate unit stop/start transaction."""

import math
import subprocess

import pytest

from hermes_cli import update_cmd_fleet as fleet


@pytest.mark.parametrize("scope_cmd", [["systemctl", "--user"], ["sudo", "-n", "systemctl"]])
@pytest.mark.parametrize("retry", [False, True])
def test_blunt_restart_and_retry_cover_unit_transaction(monkeypatch, scope_cmd, retry):
    restarts = []

    def systemctl(cmd, *, timeout):
        if "show" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "TimeoutStopUSec=1min 10s\nTimeoutStartUSec=1min 30s\n", "")
        if "restart" in cmd:
            # The manager may legitimately spend the sum of both unit budgets.
            if timeout <= 160:
                raise subprocess.TimeoutExpired(cmd, timeout)
            restarts.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "active\n", "")

    monkeypatch.setattr(fleet, "_systemctl", systemctl)
    monkeypatch.setattr(fleet, "_resolve_manage_cmd", lambda *args: scope_cmd)
    health = iter([False, True] if retry else [True])
    monkeypatch.setattr(fleet, "_wait_for_service_active", lambda *args, **kwargs: next(health))
    restarted, failed = [], []
    # serve takes the blunt path without signalling a real gateway.
    fleet._restart_one_systemd_gateway_unit(
        "hermes-serve-test", scope="user", scope_cmd=scope_cmd,
        drain_budget=45, _manage_cmd_cache={},
        restarted_services=restarted, failed_or_stale_units=failed,
    )
    assert restarted == ["hermes-serve-test"] and not failed
    assert restarts == [scope_cmd + ["restart", "hermes-serve-test"]] * (2 if retry else 1)


@pytest.mark.parametrize("probe", [
    "", "TimeoutStopUSec=infinity\nTimeoutStartUSec=infinity\n",
    "TimeoutStopUSec=garbage\nTimeoutStartUSec=NaN\n",
    "TimeoutStopUSec=2s\n", "timeout", "failure",
])
@pytest.mark.parametrize("outcome", ["success", "error", "timeout"])
def test_unavailable_budgets_are_finite_without_hiding_restart_failure(monkeypatch, probe, outcome):
    def systemctl(cmd, *, timeout):
        if "show" in cmd:
            if probe == "timeout":
                raise subprocess.TimeoutExpired(cmd, timeout)
            return subprocess.CompletedProcess(cmd, 1 if probe == "failure" else 0, probe, "")
        if "restart" in cmd:
            assert math.isfinite(timeout) and timeout > 90
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(cmd, timeout)
            return subprocess.CompletedProcess(cmd, 1 if outcome == "error" else 0, "", "denied")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(fleet, "_systemctl", systemctl)
    if outcome == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            fleet._systemctl_reset_and_restart(["systemctl", "--user"], "hermes-serve-test")
    else:
        result = fleet._systemctl_reset_and_restart(["systemctl", "--user"], "hermes-serve-test")
        assert result.returncode == (1 if outcome == "error" else 0)
