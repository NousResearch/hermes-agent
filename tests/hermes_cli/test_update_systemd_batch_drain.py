"""Signal independent systemd gateways before waiting (narrow salvage of #61412)."""

import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway
from hermes_cli import update_cmd_fleet as fleet
from hermes_cli import update_cmd_posix_pause as pause
from gateway import status


@pytest.fixture
def systemd_fleet(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway, "_ensure_user_systemd_env", lambda: None)
    monkeypatch.setattr(gateway, "_is_pid_ancestor_of_current_process", lambda pid: False)
    monkeypatch.setattr(gateway, "probe_gateway_loop_liveness", lambda pid: "alive")
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: 1000 + pid)
    monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(fleet, "_gateway_home_for_pid", lambda pid: tmp_path / str(pid))

    monkeypatch.setattr(fleet, "_repair_unit_without_fatal_exit_park", lambda *args: None)
    monkeypatch.setattr(fleet, "_needs_sudo", lambda scope: False)
    monkeypatch.setattr(pause, "already_restarted", lambda: {"units": set()})
    units = {"hermes-gateway-a": 101, "hermes-gateway-b": 202}
    events = []
    state = SimpleNamespace(inactive=set(), foreign=set(), denied=set(), waits={}, health={}, calls=[])
    monkeypatch.setattr(fleet, "_systemd_unit_owned_by_update", lambda cmd, name: name not in state.foreign)
    scope = ["systemctl", "--user"]
    monkeypatch.setattr(fleet, "_SYSTEMD_SCOPES", (("user", scope),))

    def systemctl(cmd, *, timeout):
        state.calls.append(cmd)
        if "list-units" in cmd:
            output = "\n".join(f"{name}.service loaded active running" for name in units)
        elif "--property=MainPID" in cmd:
            output = str(units[cmd[cmd.index("show") + 1]])
        elif "is-active" in cmd:
            output = "inactive" if cmd[-1] in state.inactive else "active"
        else:
            output = ""
        return subprocess.CompletedProcess(cmd, 0, output, "")

    def kill(pid, sig):
        assert sig == signal.SIGUSR1
        events.append(("signal", pid))
        if pid in state.denied:
            raise PermissionError("signal denied")

    def wait(pid, timeout, **kwargs):
        events.append(("wait", pid))
        result = state.waits.get(pid, True)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(fleet, "_systemctl", systemctl)
    monkeypatch.setattr(gateway.os, "kill", kill)
    monkeypatch.setattr(gateway, "_wait_for_pid_exit", wait)
    monkeypatch.setattr(fleet, "_wait_for_service_active", lambda cmd, name, **kwargs: state.health.get(name, True))
    return units, events, state


@pytest.mark.platforms("linux")
def test_independent_gateways_receive_restart_before_first_drain_wait(systemd_fleet):
    units, events, _ = systemd_fleet
    restarted, failed, scoped = [], [], set()
    fleet._restart_systemd_gateway_units(restarted, failed, scoped, drain_budget=45)

    assert events == [("signal", 101), ("signal", 202), ("wait", 101), ("wait", 202)]
    assert restarted == list(units)
    assert failed == []
    assert scoped == {f"user/{name}" for name in units}


@pytest.mark.platforms("linux")
def test_shared_pid_is_signalled_once_across_scopes(systemd_fleet, monkeypatch):
    units, events, _ = systemd_fleet
    units["hermes-gateway-b"] = 101
    monkeypatch.setattr(fleet, "_SYSTEMD_SCOPES", (
        ("user", ["systemctl", "--user"]), ("system", ["systemctl"])))
    restarted, failed, scoped = [], [], set()
    fleet._restart_systemd_gateway_units(restarted, failed, scoped, 45)
    assert events == [("signal", 101), ("wait", 101)]
    assert restarted == ["hermes-gateway-a"]
    assert scoped == {"user/hermes-gateway-a"}
    assert not failed


@pytest.mark.platforms("linux")
def test_inactive_foreign_and_already_resumed_units_are_untouched(systemd_fleet, monkeypatch):
    units, events, state = systemd_fleet
    units.update({"hermes-gateway-c": 303, "hermes-gateway-d": 404})
    state.inactive.add("hermes-gateway-a")
    state.foreign.add("hermes-gateway-b")
    monkeypatch.setattr(pause, "already_restarted", lambda: {"units": {"user/hermes-gateway-c"}})
    restarted, failed, scoped = [], [], set()
    fleet._restart_systemd_gateway_units(restarted, failed, scoped, 45)
    assert events == [("signal", 404), ("wait", 404)]
    assert restarted == ["hermes-gateway-d"]
    assert scoped == {"user/hermes-gateway-c", "user/hermes-gateway-d"}
    assert not failed


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("failure", [False, subprocess.TimeoutExpired("mock wait", 45)])
def test_failed_drain_or_wait_timeout_does_not_lose_later_success(systemd_fleet, failure):
    _, events, state = systemd_fleet
    state.waits[101] = failure
    state.health["hermes-gateway-a"] = False
    restarted, failed, scoped = [], [], set()
    fleet._restart_systemd_gateway_units(restarted, failed, scoped, 45)
    assert events[:2] == [("signal", 101), ("signal", 202)]
    assert restarted == ["hermes-gateway-b"]
    assert failed == ["hermes-gateway-a"]
    assert scoped == {"user/hermes-gateway-b"}


@pytest.mark.platforms("linux")
def test_signal_denied_keeps_forced_restart_and_readiness_checks(systemd_fleet):
    _, events, state = systemd_fleet
    state.denied.add(101)
    restarted, failed, scoped = [], [], set()
    fleet._restart_systemd_gateway_units(restarted, failed, scoped, 45)
    assert events == [("signal", 101), ("signal", 202), ("wait", 202)]
    assert any("restart" in cmd and cmd[-1] == "hermes-gateway-a" for cmd in state.calls)
    assert restarted == ["hermes-gateway-a", "hermes-gateway-b"]
    assert not failed


@pytest.mark.platforms("linux")
def test_no_sigusr1_and_serve_units_use_existing_restart_fallback(systemd_fleet, monkeypatch):
    units, events, state = systemd_fleet
    units["hermes-serve-test"] = 303
    monkeypatch.delattr(signal, "SIGUSR1")
    restarted, failed, scoped = [], [], set()
    fleet._restart_systemd_gateway_units(restarted, failed, scoped, 45)
    assert events == []
    assert restarted == list(units)
    assert not failed
    assert [cmd[-1] for cmd in state.calls if "restart" in cmd] == list(units)


@pytest.mark.platforms("linux")
def test_unprivileged_signal_failure_reports_manual_recovery(systemd_fleet, monkeypatch):
    _, events, state = systemd_fleet
    state.denied.add(101)
    monkeypatch.setattr(fleet, "_resolve_manage_cmd", lambda *args: None)
    restarted, failed, scoped = [], [], set()
    fleet._restart_systemd_gateway_units(restarted, failed, scoped, 45)
    assert events[:2] == [("signal", 101), ("signal", 202)]
    assert failed == ["hermes-gateway-a"]
    assert restarted == ["hermes-gateway-b"]
    assert not any("restart" in cmd or "start" in cmd for cmd in state.calls)


@pytest.mark.platforms("linux")
def test_ancestor_self_restart_stays_pending_without_drain_wait(systemd_fleet, monkeypatch):
    _, events, _ = systemd_fleet
    monkeypatch.setattr(gateway, "_is_pid_ancestor_of_current_process", lambda pid: pid == 101)
    pending = set()
    restarted, failed, scoped = [], [], set()
    fleet._restart_systemd_gateway_units(restarted, failed, scoped, 45, pending)
    assert pending == {101}
    assert events == [("signal", 101), ("signal", 202), ("wait", 202)]
    assert not failed


@pytest.mark.platforms("linux")
def test_later_wait_uses_only_remaining_signal_budget(systemd_fleet, monkeypatch):
    from hermes_cli import update_cmd_fleet_batch as batch

    _, events, _ = systemd_fleet
    clock = [100.0]
    budgets = []
    monkeypatch.setattr(batch.time, "monotonic", lambda: clock[0])

    def wait(pid, timeout, **kwargs):
        events.append(("wait", pid))
        budgets.append(timeout)
        clock[0] = 145.0
        return True

    monkeypatch.setattr(gateway, "_wait_for_pid_exit", wait)
    fleet._restart_systemd_gateway_units([], [], set(), 45)
    assert budgets == [45.0, 0.0]


@pytest.mark.platforms("linux")
def test_wedged_gateway_escalates_only_after_healthy_siblings_are_signalled(systemd_fleet, monkeypatch):
    _, events, _ = systemd_fleet
    monkeypatch.setattr(gateway, "probe_gateway_loop_liveness", lambda pid: (
        gateway.GATEWAY_LOOP_WEDGED if pid == 101 else "alive"))
    monkeypatch.setattr(gateway, "_escalate_wedged_gateway", lambda pid: events.append(("escalate", pid)))
    restarted, failed, scoped = [], [], set()
    fleet._restart_systemd_gateway_units(restarted, failed, scoped, 45)
    assert events == [("signal", 202), ("escalate", 101), ("wait", 202)]
    assert restarted == ["hermes-gateway-a", "hermes-gateway-b"]
    assert not failed


@pytest.mark.platforms("linux")
def test_reconciliation_abort_keeps_previously_settled_scoped_units(systemd_fleet):
    _, events, state = systemd_fleet
    state.waits[202] = RuntimeError("reconciliation aborted")
    restarted, failed, scoped = [], [], set()
    with pytest.raises(RuntimeError, match="reconciliation aborted"):
        fleet._restart_systemd_gateway_units(restarted, failed, scoped, 45)
    assert events[:2] == [("signal", 101), ("signal", 202)]
    assert restarted == ["hermes-gateway-a"]
    assert scoped == {"user/hermes-gateway-a"}


@pytest.mark.platforms("linux")
def test_deferred_wedged_pid_reused_during_sibling_drain_is_not_signalled(systemd_fleet, monkeypatch):
    units, events, state = systemd_fleet
    births = {101: 1101, 202: 1202}
    monkeypatch.setattr(status, "get_process_start_time", births.get)
    monkeypatch.setattr(gateway, "probe_gateway_loop_liveness", lambda pid: (
        gateway.GATEWAY_LOOP_WEDGED if pid == 202 and births[202] == 1202 else "unknown"))
    monkeypatch.setattr(gateway, "_escalate_wedged_gateway", lambda pid: events.append(("escalate", pid)))

    def wait(pid, timeout, **kwargs):
        events.append(("wait", pid))
        if pid == 101:
            births[202] = 9202
            units["hermes-gateway-b"] = 303
        return True

    monkeypatch.setattr(gateway, "_wait_for_pid_exit", wait)
    restarted, failed, scoped = [], [], set()
    fleet._restart_systemd_gateway_units(restarted, failed, scoped, 45)
    assert events == [("signal", 101), ("wait", 101)]
    assert not any("restart" in cmd for cmd in state.calls)
    assert restarted == list(units)
    assert failed == []
    assert scoped == {f"user/{name}" for name in units}


@pytest.mark.platforms("linux")
def test_healthy_deferred_pid_reuse_does_not_restart_successor(systemd_fleet, monkeypatch):
    units, events, state = systemd_fleet
    births = {101: 1101, 202: 1202}
    monkeypatch.setattr(status, "get_process_start_time", births.get)

    def wait(pid, timeout, **kwargs):
        events.append(("wait", pid))
        if pid == 101:
            births[202] = 9202
            units["hermes-gateway-b"] = 303
        return pid == 101  # A reused, still-existing B PID must not cause a timeout.

    monkeypatch.setattr(gateway, "_wait_for_pid_exit", wait)
    restarted, failed, scoped = [], [], set()
    fleet._restart_systemd_gateway_units(restarted, failed, scoped, 45)
    assert events == [("signal", 101), ("signal", 202), ("wait", 101)]
    assert not any("restart" in cmd for cmd in state.calls)
    assert restarted == list(units)
    assert failed == []
    assert scoped == {f"user/{name}" for name in units}


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("unreadable_at", ["snapshot", "deferred"])
def test_unreadable_wedged_identity_uses_scoped_restart_only(systemd_fleet, monkeypatch, unreadable_at):
    units, events, state = systemd_fleet
    births = {101: 1101, 202: None if unreadable_at == "snapshot" else 1202}
    monkeypatch.setattr(status, "get_process_start_time", births.get)
    monkeypatch.setattr(gateway, "probe_gateway_loop_liveness", lambda pid: (
        gateway.GATEWAY_LOOP_WEDGED if pid == 202 else "alive"))
    monkeypatch.setattr(gateway, "_escalate_wedged_gateway", lambda pid: pytest.fail("unproven escalation"))

    def wait(pid, timeout, **kwargs):
        events.append(("wait", pid))
        births[202] = None
        return True

    monkeypatch.setattr(gateway, "_wait_for_pid_exit", wait)
    restarted, failed, scoped = [], [], set()
    fleet._restart_systemd_gateway_units(restarted, failed, scoped, 45)
    assert events == [("signal", 101), ("wait", 101)]
    assert [cmd[-1] for cmd in state.calls if "restart" in cmd] == ["hermes-gateway-b"]
    assert restarted == list(units)
    assert failed == []
    assert scoped == {f"user/{name}" for name in units}


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("replacement_birth,forced", [(9202, False), (None, True)])
def test_identity_is_rechecked_during_active_drain_wait(systemd_fleet, monkeypatch, replacement_birth, forced):
    units, events, state = systemd_fleet
    births = {101: 1101, 202: 1202}
    monkeypatch.setattr(status, "get_process_start_time", births.get)

    def wait(pid, timeout, *, on_progress):
        events.append(("wait", pid))
        if pid == 202:
            births[202] = replacement_birth
            on_progress()
            pytest.fail("must stop waiting on an unproven or replaced numeric owner")
        return True

    monkeypatch.setattr(gateway, "_wait_for_pid_exit", wait)
    restarted, failed, scoped = [], [], set()
    fleet._restart_systemd_gateway_units(restarted, failed, scoped, 45)
    assert [cmd[-1] for cmd in state.calls if "restart" in cmd] == (["hermes-gateway-b"] if forced else [])
    assert events == [("signal", 101), ("signal", 202), ("wait", 101), ("wait", 202)]
    assert restarted == list(units)
    assert failed == []
    assert scoped == {f"user/{name}" for name in units}
