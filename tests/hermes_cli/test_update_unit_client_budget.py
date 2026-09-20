"""Update clients cover unit transactions without hiding manager failures."""
import subprocess

import pytest

from hermes_cli import update_cmd_fleet as fleet


@pytest.mark.parametrize("graceful,retry", [(False, False), (False, True), (True, False), ("catchup", False)])
def test_unit_transaction_budget_preserves_scope_and_health(monkeypatch, graceful, retry):
    catchup = graceful == "catchup"
    scope = ["systemctl", "--user"] if catchup else ["systemctl", "--no-ask-password"]
    manage = scope if catchup else ["sudo", "-n", *scope]
    calls = []

    def systemctl(cmd, *, timeout):
        calls.append((cmd, timeout))
        if "list-units" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "hermes-serve-test.service loaded active running", "")
        if "show" in cmd:
            assert cmd[:len(scope)] == scope
            output = "42" if "--property=MainPID" in cmd else "TimeoutStopUSec=70s\nTimeoutStartUSec=90s"
            return subprocess.CompletedProcess(cmd, 0, output, "")
        if "restart" in cmd or "start" in cmd:
            assert cmd[:len(manage)] == manage
            assert timeout > (90 if graceful else 160)
        return subprocess.CompletedProcess(cmd, 0, "active", "")

    monkeypatch.setattr(fleet, "_systemctl", systemctl)
    if catchup:
        monkeypatch.setattr(fleet, "_SYSTEMD_SCOPES", (("user", scope),))
        failed = []
        fleet._restart_systemd_gateway_units_best_effort(failed, list(fleet._systemd_gateway_unit_listings()))
        assert not failed
        assert sum("restart" in cmd for cmd, _ in calls) == 1
        return
    monkeypatch.setattr(fleet, "_drain_or_signal_gateway_for_update", lambda *a: True)
    health = iter([False, True] if retry else [True])
    monkeypatch.setattr(fleet, "_wait_for_service_active", lambda *a, **kw: next(health))
    name = "hermes-gateway-test" if graceful else "hermes-serve-test"
    restarted, failed = [], []
    fleet._restart_one_systemd_gateway_unit(
        name, scope="system", scope_cmd=scope, drain_budget=45,
        _manage_cmd_cache={"system": manage}, restarted_services=restarted,
        failed_or_stale_units=failed,
    )
    assert restarted == [name] and not failed
    assert sum("restart" in cmd or "start" in cmd for cmd, _ in calls) == (2 if retry else 1)


@pytest.mark.parametrize("limits", ["", "TimeoutStopUSec=infinity\nTimeoutStartUSec=invalid", "TimeoutStopUSec=70000000\nTimeoutStartUSec=90s"])
@pytest.mark.parametrize("outcome", [0, 7, "timeout"])
def test_budget_fallback_keeps_real_errors(monkeypatch, limits, outcome):
    def systemctl(cmd, *, timeout):
        if "show" in cmd:
            return subprocess.CompletedProcess(cmd, 0, limits, "")
        if "restart" in cmd:
            assert 160 < timeout < 2**31 / 1000
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(cmd, timeout)
            return subprocess.CompletedProcess(cmd, outcome, "", "manager diagnostic")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(fleet, "_systemctl", systemctl)
    if outcome == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            fleet._systemctl_reset_and_restart(["systemctl"], "hermes-serve-test")
    else:
        result = fleet._systemctl_reset_and_restart(["systemctl"], "hermes-serve-test")
        assert result.returncode == outcome
        assert result.stderr == "manager diagnostic"
