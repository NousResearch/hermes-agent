"""Systemd operations use a fake subprocess boundary, never a live user manager."""

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import update_auto_schedule as scheduler
from hermes_cli import update_auto_schedule_common as common

pytestmark = pytest.mark.platforms("linux")


class FakeSystemd:
    def __init__(self, info):
        self.paths = {info[key].name: info[key] for key in ("service_path", "path")}
        self.active = set()
        self.failed = set()
        self.enabled = set()
        self.calls = []
        self.fail_once = None
        self.ignore_enable = False
        self.foreign_path = None
        self.show_output = None
        self.activate_on_disable = False

    def run(self, command, **kwargs):
        assert command[:2] == ["/fake/systemctl", "--user"]
        assert kwargs["timeout"] == 30
        assert not kwargs.get("shell")
        args = command[2:]
        self.calls.append(args)
        operation = args[0]
        if self.fail_once == operation:
            self.fail_once = None
            return subprocess.CompletedProcess(command, 1, "", "injected manager failure")
        output = self._operate(args)
        return subprocess.CompletedProcess(command, 0, output, "")

    def _operate(self, args):
        operation = args[0]
        if operation == "show":
            return self._show(args[1])
        if operation in {"disable", "stop"}:
            name = args[-1]
            if operation == "stop" or "--now" in args:
                self.active.discard(name)
            if operation == "disable":
                self.enabled.discard(name)
                if self.activate_on_disable and name.endswith(".timer"):
                    self.active.add(name.removesuffix(".timer") + ".service")
                    self.activate_on_disable = False
        if operation == "enable" and not self.ignore_enable:
            self.enabled.add(args[-1])
            if "--now" in args:
                self.active.add(args[-1])
        if operation == "start":
            self.active.add(args[-1])
        if operation == "reset-failed":
            self.failed.discard(args[-1])
        return ""

    def _show(self, name):
        if self.show_output is not None:
            return self.show_output
        path = self.paths[name]
        exists = path.exists()
        load = "loaded" if exists else "not-found"
        unit = "static" if name.endswith(".service") else "disabled"
        unit = "enabled" if name in self.enabled else unit
        unit = unit if exists else ""
        active = "active" if name in self.active else "inactive"
        if name in self.failed:
            active = "failed"
            load = "loaded"
        fragment = str(path) if exists else ""
        fragment = self.foreign_path or fragment
        return f"LoadState={load}\nUnitFileState={unit}\nActiveState={active}\nFragmentPath={fragment}\n"


@pytest.fixture
def environment(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    spec = scheduler.SchedulerSpec("v1-" + "a" * 24, ["/opt/hermes", "update", "auto", "run-scheduled"], home, "04:00", ["21:00"])
    info = scheduler.paths(spec)
    manager = FakeSystemd(info)
    monkeypatch.setattr(common, "locate_command", lambda name: SimpleNamespace(command=("/fake/" + name,)))
    monkeypatch.setattr(common.subprocess, "run", manager.run)
    return spec, info, manager


def _seed(info, manager):
    for key, data, mode in (("service_path", b"original service\n", 0o640), ("path", b"original timer\n", 0o644)):
        info[key].parent.mkdir(parents=True, exist_ok=True)
        info[key].write_bytes(data)
        info[key].chmod(mode)
    manager.active.add(info["path"].name)
    manager.enabled.add(info["path"].name)


def test_enable_verifies_user_timer_and_rolls_back_exact_previous_state(environment):
    spec, info, manager = environment
    _seed(info, manager)
    originals = {key: common.snapshot_file(info[key]) for key in ("service_path", "path")}
    handle = scheduler.enable(spec)
    assert handle.scheduler_type == "systemd-user"
    assert handle.path == info["path"]
    assert info["path"].name in manager.active & manager.enabled
    assert common.snapshot_file(info["path"]) != originals["path"]
    receipt = handle.rollback()
    assert receipt["ok"], receipt
    assert {key: common.snapshot_file(info[key]) for key in originals} == originals
    assert info["path"].name in manager.active & manager.enabled


def test_new_enable_can_be_rolled_back_without_leaving_files(environment):
    spec, info, manager = environment
    handle = scheduler.enable(spec)
    assert handle.rollback()["ok"]
    assert not info["path"].exists()
    assert not info["service_path"].exists()
    assert not manager.enabled
    assert not manager.active


def test_disable_removes_units_and_can_restore_them(environment):
    spec, info, manager = environment
    _seed(info, manager)
    handle = scheduler.disable(spec)
    assert handle.removed
    assert not manager.active
    assert not info["path"].exists()
    assert not info["service_path"].exists()
    assert all(call[:2] != ["stop", info["service_path"].name] for call in manager.calls)
    assert handle.rollback()["ok"]
    assert info["path"].name in manager.active


def test_manager_failure_restores_files_and_mode(environment):
    spec, info, manager = environment
    _seed(info, manager)
    originals = {key: common.snapshot_file(info[key]) for key in ("service_path", "path")}
    manager.fail_once = "enable"
    with pytest.raises(RuntimeError, match="injected manager failure"):
        scheduler.enable(spec)
    assert {key: common.snapshot_file(info[key]) for key in originals} == originals
    assert info["path"].name in manager.active & manager.enabled


def test_success_exit_without_timer_adoption_is_failure(environment):
    spec, info, manager = environment
    manager.ignore_enable = True
    with pytest.raises(RuntimeError, match="not persistently enabled"):
        scheduler.enable(spec)
    assert not info["path"].exists()
    assert not info["service_path"].exists()


def test_recovery_failure_returns_durable_receipt(environment):
    spec, info, manager = environment
    _seed(info, manager)
    manager.ignore_enable = True
    with pytest.raises(scheduler.SchedulerRecoveryError) as raised:
        scheduler.enable(spec)
    assert not raised.value.receipt["ok"]
    assert raised.value.receipt["files"][0]["ok"]
    assert any("unit_file_state" in error for error in raised.value.receipt["errors"])


def test_symlink_artifact_refused_before_mutating_manager(environment, tmp_path):
    spec, info, manager = environment
    foreign = tmp_path / "foreign"
    foreign.write_text("keep")
    info["path"].parent.mkdir(parents=True)
    info["path"].symlink_to(foreign)
    with pytest.raises(RuntimeError, match="symlink"):
        scheduler.enable(spec)
    assert manager.calls == []
    assert foreign.read_text() == "keep"


def test_foreign_manager_source_is_never_mutated(environment):
    spec, info, manager = environment
    _seed(info, manager)
    manager.foreign_path = "/foreign/other.timer"
    with pytest.raises(RuntimeError, match="another path"):
        scheduler.disable(spec)
    assert all(call[0] == "show" for call in manager.calls)
    assert info["path"].exists()


def test_unknown_manager_state_is_not_treated_as_absent(environment):
    spec, info, manager = environment
    manager.show_output = ""
    with pytest.raises(RuntimeError, match="complete scheduler state"):
        scheduler.enable(spec)
    assert not info["path"].exists()
    assert all(call[0] == "show" for call in manager.calls)


def test_disable_already_absent_is_verified(environment):
    spec, info, manager = environment
    handle = scheduler.disable(spec)
    assert not handle.removed
    assert handle.rollback()["ok"]
    assert not info["path"].exists()


def test_identity_never_touches_other_installation(environment):
    spec, info, manager = environment
    foreign = info["path"].with_name("hermes-auto-update-" + "b" * 24 + ".timer")
    foreign.parent.mkdir(parents=True)
    foreign.write_text("other installation")
    scheduler.enable(spec)
    scheduler.disable(spec)
    assert foreign.read_text() == "other installation"
    assert all(foreign.name not in args for args in manager.calls)


@pytest.mark.parametrize("operation", [scheduler.enable, scheduler.disable])
def test_active_updater_is_never_stopped_or_reconfigured(environment, operation):
    spec, info, manager = environment
    _seed(info, manager)
    manager.active.add(info["service_path"].name)
    before = common.snapshot_file(info["service_path"])
    with pytest.raises(RuntimeError, match="updater is active"):
        operation(spec)
    assert all(call[0] == "show" for call in manager.calls)
    assert info["service_path"].name in manager.active
    assert common.snapshot_file(info["service_path"]) == before


def test_rollback_leaves_unexpected_active_updater_running(environment):
    spec, info, manager = environment
    handle = scheduler.enable(spec)
    before = common.snapshot_file(info["service_path"])
    manager.active.add(info["service_path"].name)
    receipt = handle.rollback()
    assert not receipt["ok"]
    assert any("left running" in error for error in receipt["errors"])
    assert info["service_path"].name in manager.active
    assert common.snapshot_file(info["service_path"]) == before
    assert not info["path"].name in manager.active


def test_service_starting_during_disable_is_left_running(environment):
    spec, info, manager = environment
    _seed(info, manager)
    before = common.snapshot_file(info["service_path"])
    manager.activate_on_disable = True
    with pytest.raises(scheduler.SchedulerRecoveryError) as raised:
        scheduler.disable(spec)
    assert "left running" in str(raised.value)
    assert info["service_path"].name in manager.active
    assert common.snapshot_file(info["service_path"]) == before
    assert all(call[0] != "stop" for call in manager.calls)


def test_timeout_after_writes_restores_exact_artifacts(environment, monkeypatch):
    spec, info, manager = environment
    _seed(info, manager)
    before = common.snapshot_file(info["service_path"])
    timed_out = False

    def run(command, **kwargs):
        nonlocal timed_out
        if command[2] == "enable" and not timed_out:
            timed_out = True
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return manager.run(command, **kwargs)

    monkeypatch.setattr(common.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="timed out"):
        scheduler.enable(spec)
    assert common.snapshot_file(info["service_path"]) == before
    assert info["path"].name in manager.active & manager.enabled


def test_file_substitution_during_enable_is_recovery_failure(environment, monkeypatch, tmp_path):
    spec, info, manager = environment
    foreign = tmp_path / "foreign"
    foreign.write_text("keep")
    substituted = False

    def run(command, **kwargs):
        nonlocal substituted
        result = manager.run(command, **kwargs)
        if command[2] == "enable" and not substituted:
            substituted = True
            info["path"].unlink()
            info["path"].symlink_to(foreign)
        return result

    monkeypatch.setattr(common.subprocess, "run", run)
    with pytest.raises(scheduler.SchedulerRecoveryError) as raised:
        scheduler.enable(spec)
    assert not raised.value.receipt["ok"]
    assert info["path"].is_symlink()
    assert foreign.read_text() == "keep"


def test_failed_oneshot_can_be_removed_after_update_failure(environment):
    spec, info, manager = environment
    _seed(info, manager)
    manager.failed.add(info["service_path"].name)
    handle = scheduler.disable(spec)
    assert handle.removed
    assert not info["service_path"].exists()
    assert info["service_path"].name not in manager.failed
    assert ["reset-failed", info["service_path"].name] in manager.calls


def test_failed_oneshot_can_be_reenabled_without_running_it(environment):
    spec, info, manager = environment
    _seed(info, manager)
    manager.failed.add(info["service_path"].name)
    scheduler.enable(spec)
    assert info["path"].name in manager.active & manager.enabled
    assert info["service_path"].name not in manager.active


def test_rollback_reports_cleared_historical_failure_honestly(environment):
    spec, info, manager = environment
    _seed(info, manager)
    manager.failed.add(info["service_path"].name)
    before = common.snapshot_file(info["service_path"])
    handle = scheduler.disable(spec)
    receipt = handle.rollback()
    assert not receipt["ok"]
    assert common.snapshot_file(info["service_path"]) == before
    assert any("active_state" in error and "failed" in error for error in receipt["errors"])
