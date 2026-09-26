"""Launchd lifecycle contracts with fake time and no live process operations."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway import status
from hermes_cli import gateway as gateway_cli
from hermes_cli.service_manager import LaunchdServiceManager, SystemdServiceManager


@pytest.fixture(autouse=True)
def isolated_launchd(monkeypatch, tmp_path):
    """Trap unintended execution even when a regression is present."""
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected process operation")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(gateway_cli, "terminate_pid", forbidden)
    monkeypatch.setattr(status, "get_running_pid", lambda: None)
    monkeypatch.setattr(status, "get_process_start_time", forbidden)
    monkeypatch.setattr(gateway_cli, "_mark_planned_stop", lambda: None)
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: tmp_path / "gateway.plist")
    monkeypatch.setattr(gateway_cli, "_append_launchd_reload_log", lambda message: None)


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(gateway_cli.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(gateway_cli.time, "sleep", sleep)
    return clock


@pytest.mark.parametrize("error", [PermissionError, OSError, ProcessLookupError])
@pytest.mark.parametrize("exited", [False, True])
def test_force_error_requires_fresh_absence(monkeypatch, clock, error, exited):
    live: list[int | None] = [4242]
    monkeypatch.setattr(status, "get_running_pid", lambda: live[0])
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: 100)

    def terminate(pid, **kwargs):
        assert (pid, kwargs) == (4242, {"force": True, "expected_start_time": 100})
        if exited:
            live[0] = None
        raise error("cannot signal")

    monkeypatch.setattr(gateway_cli, "terminate_pid", terminate)
    assert gateway_cli._wait_for_gateway_exit(timeout=1, force_after=0) is exited


@pytest.mark.parametrize("replacement_pid,replacement_start", [(5252, 100), (4242, 200), (4242, None)])
def test_wait_refuses_replacement_identity(monkeypatch, clock, replacement_pid, replacement_start):
    monkeypatch.setattr(status, "get_running_pid", lambda: 4242 if clock.now == 0 else replacement_pid)
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: 100 if clock.now == 0 else replacement_start)
    terminate = Mock()
    monkeypatch.setattr(gateway_cli, "terminate_pid", terminate)

    assert gateway_cli._wait_for_gateway_exit(timeout=1, force_after=0.2) is False
    terminate.assert_not_called()


def test_wait_refuses_missing_initial_identity(monkeypatch, clock):
    monkeypatch.setattr(status, "get_running_pid", lambda: 4242)
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: None)
    terminate = Mock()
    monkeypatch.setattr(gateway_cli, "terminate_pid", terminate)

    assert gateway_cli._wait_for_gateway_exit(timeout=1, force_after=0) is False
    terminate.assert_not_called()


def test_wait_force_keeps_original_identity(monkeypatch, clock):
    live: list[int | None] = [4242]
    monkeypatch.setattr(status, "get_running_pid", lambda: live[0])
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: 100)

    def terminate(pid, **kwargs):
        assert (pid, kwargs) == (4242, {"force": True, "expected_start_time": 100})
        live[0] = None

    terminate_mock = Mock(side_effect=terminate)
    monkeypatch.setattr(gateway_cli, "terminate_pid", terminate_mock)
    assert gateway_cli._wait_for_gateway_exit(timeout=1, force_after=0.2) is True
    terminate_mock.assert_called_once()


def test_wait_already_absent(monkeypatch, clock):
    assert gateway_cli._wait_for_gateway_exit(timeout=1) is True


@pytest.mark.parametrize("stopped", [False, True])
def test_stop_reports_observed_outcome(monkeypatch, capsys, stopped):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(subprocess, "run", run)
    wait = Mock(return_value=stopped)
    monkeypatch.setattr(gateway_cli, "_wait_for_gateway_exit", wait)

    assert gateway_cli.launchd_stop() is stopped
    assert ("✓ Service stopped" in capsys.readouterr().out) is stopped
    wait.assert_called_once_with(timeout=10.0, force_after=5.0)
    assert run.call_args.kwargs["capture_output"] is True


def test_uninstall_preserves_plist_when_stop_fails(monkeypatch, tmp_path, capsys):
    plist = tmp_path / "gateway.plist"
    plist.write_text("existing definition", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0)))
    monkeypatch.setattr(gateway_cli, "_wait_for_gateway_exit", lambda **kwargs: False)

    with pytest.raises(RuntimeError) as caught:
        gateway_cli.launchd_uninstall()
    assert type(caught.value).__name__ == "LaunchdStopError"
    assert plist.read_text(encoding="utf-8") == "existing definition"
    assert "✓ Service uninstalled" not in capsys.readouterr().out


@pytest.mark.parametrize("entry", ["facade", "installed", "manager", "cli"])
def test_failed_stop_propagates_through_public_callers(monkeypatch, entry, capsys):
    monkeypatch.setattr(subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0)))
    monkeypatch.setattr(gateway_cli, "_wait_for_gateway_exit", lambda **kwargs: False)
    monkeypatch.setattr(gateway_cli, "_installed_service_kind", lambda: "launchd")

    def invoke():
        if entry == "facade":
            return gateway_cli._service_call("launchd", "stop")
        if entry == "installed":
            return gateway_cli._stop_installed_service(False)
        if entry == "manager":
            return LaunchdServiceManager().stop("ignored")
        monkeypatch.setattr(gateway_cli, "_gateway_command_inner", lambda args: gateway_cli._stop_installed_service(False))
        return gateway_cli.gateway_command(SimpleNamespace(gateway_command="stop"))

    with pytest.raises(SystemExit if entry == "cli" else RuntimeError) as caught:
        invoke()
    if entry == "cli":
        assert isinstance(caught.value, SystemExit)
        assert caught.value.code == 1
    else:
        assert type(caught.value).__name__ == "LaunchdStopError"
    assert "✓ Service stopped" not in capsys.readouterr().out


def test_stop_success_through_launchd_manager(monkeypatch):
    monkeypatch.setattr(gateway_cli, "launchd_stop", lambda: True)
    assert LaunchdServiceManager().stop("ignored") is None


def test_systemd_false_return_keeps_native_semantics(monkeypatch):
    monkeypatch.setattr(gateway_cli, "systemd_stop", lambda **kwargs: False)
    assert gateway_cli._service_call("systemd", "stop") is False
    assert SystemdServiceManager().stop("ignored") is None


@pytest.mark.parametrize("deadline,spent,expected", [
    (None, (27, 2), [30, 3, 1]),
    (5, (4, 0.5), [5, 1, 0.5]),
])
def test_bootstrap_eio_recovery_shares_one_budget(monkeypatch, clock, deadline, spent, expected):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd[1], kwargs))
        if len(calls) == 1:
            clock.now += spent[0]
            raise subprocess.CalledProcessError(5, cmd)
        if cmd[1] == "bootout":
            clock.now += spent[1]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    options = {} if deadline is None else {"deadline": deadline}
    gateway_cli._launchctl_bootstrap("gui/501", "gateway.plist", "ai.hermes.gateway", **options)
    assert [verb for verb, kwargs in calls] == ["bootstrap", "bootout", "bootstrap"]
    assert [kwargs["timeout"] for verb, kwargs in calls] == expected
    assert calls[1][1]["capture_output"] is True


@pytest.mark.parametrize("entry", ["bootstrap", "retry"])
def test_expired_deadline_does_no_work(monkeypatch, clock, entry):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout='"PID" = 4242;', stderr=""))
    monkeypatch.setattr(subprocess, "run", run)
    if entry == "bootstrap":
        with pytest.raises(subprocess.TimeoutExpired):
            gateway_cli._launchctl_bootstrap("gui/501", "gateway.plist", "ai.hermes.gateway", deadline=0)
    else:
        assert gateway_cli._retry_launchctl_bootstrap_until_registered(
            "gui/501", "gateway.plist", "ai.hermes.gateway", deadline=0,
        ) is False
    run.assert_not_called()
    assert clock.sleeps == []


@pytest.mark.parametrize("spent", [9, 10])
def test_retry_bounds_supervision_probe_by_remaining_time(monkeypatch, clock, spent):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd[1], kwargs["timeout"]))
        if cmd[1] == "bootstrap":
            clock.now += spent
        return SimpleNamespace(returncode=0, stdout='"PID" = 4242;', stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert gateway_cli._retry_launchctl_bootstrap_until_registered(
        "gui/501", "gateway.plist", "ai.hermes.gateway", deadline=10,
    ) is (spent < 10)
    assert calls == ([("bootstrap", 10), ("list", 1)] if spent == 9 else [("bootstrap", 10)])
    assert clock.sleeps == []


def test_retry_forwards_deadline_and_caps_final_sleep(monkeypatch, clock):
    bootstrap = Mock(side_effect=subprocess.CalledProcessError(1, ["launchctl", "bootstrap"]))
    monkeypatch.setattr(gateway_cli, "_launchctl_bootstrap", bootstrap)

    assert gateway_cli._retry_launchctl_bootstrap_until_registered(
        "gui/501", "gateway.plist", "ai.hermes.gateway", deadline=0.5,
    ) is False
    assert clock.sleeps == [0.5]
    bootstrap.assert_called_once_with(
        "gui/501", "gateway.plist", "ai.hermes.gateway", timeout=30, deadline=0.5,
    )


@pytest.mark.parametrize("code", [1, 5, 125])
def test_bootstrap_preserves_terminal_error_for_native_callers(monkeypatch, clock, code):
    calls = []
    failure = subprocess.CalledProcessError(code, ["launchctl", "bootstrap"], stderr="diagnostic")

    def run(cmd, **kwargs):
        calls.append(cmd[1])
        if cmd[1] == "bootstrap":
            raise failure
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        gateway_cli._launchctl_bootstrap("gui/501", "gateway.plist", "ai.hermes.gateway")
    assert caught.value is failure
    assert caught.value.stderr == "diagnostic"
    assert calls == (["bootstrap", "bootout", "bootstrap"] if code == 5 else ["bootstrap"])


def test_bootstrap_stops_recovery_when_bootout_uses_remaining_budget(monkeypatch, clock):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd[1], kwargs["timeout"]))
        if cmd[1] == "bootstrap" and len(calls) == 1:
            clock.now += 29
            raise subprocess.CalledProcessError(5, cmd)
        clock.now += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        gateway_cli._launchctl_bootstrap("gui/501", "gateway.plist", "ai.hermes.gateway")
    assert calls == [("bootstrap", 30), ("bootout", 1)]
