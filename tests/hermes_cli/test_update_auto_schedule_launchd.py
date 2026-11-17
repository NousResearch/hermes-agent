"""LaunchAgent transactions are tested only on macOS, with a fake launchctl."""

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import update_auto_schedule as scheduler
from hermes_cli import update_auto_schedule_common as common

pytestmark = pytest.mark.platforms("macos")


class FakeLaunchd:
    def __init__(self, path):
        self.path = path
        self.label = path.stem
        self.loaded = False
        self.enabled = True
        self.running = False
        self.fail_once = None
        self.ignore_bootstrap = False
        self.calls = []
        self.foreign_path = None
        self.reported_pid = None

    def run(self, command, **kwargs):
        assert command[0] == "/fake/launchctl"
        assert kwargs["timeout"] == 30
        assert not kwargs.get("shell")
        args = command[1:]
        self.calls.append(args)
        if self.fail_once == args[0]:
            self.fail_once = None
            return subprocess.CompletedProcess(command, 1, "", "injected manager failure")
        code, stdout, stderr = self._operate(args)
        return subprocess.CompletedProcess(command, code, stdout, stderr)

    def _operate(self, args):
        operation = args[0]
        if operation == "print":
            if not self.loaded:
                return 113, "", "Could not find service"
            state = "running" if self.running else "waiting"
            pid = f"pid = {self.reported_pid}\n" if self.reported_pid else ""
            return 0, f"path = {self.foreign_path or self.path}\nstate = {state}\n{pid}", ""
        if operation == "print-disabled":
            return 0, f'"{self.label}" => {str(not self.enabled).lower()}', ""
        if operation == "bootout":
            self.loaded = self.running = False
        if operation == "enable":
            self.enabled = True
        if operation == "disable":
            self.enabled = False
        if operation == "bootstrap":
            if not self.enabled:
                return 5, "", "Service is disabled"
            self.loaded = not self.ignore_bootstrap
        if operation == "kickstart":
            self.running = True
        return 0, "", ""


@pytest.fixture
def environment(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    spec = scheduler.SchedulerSpec("v1-" + "a" * 24, ["/opt/hermes", "update", "auto", "run-scheduled"], home, "04:00", ["21:00"])
    path = scheduler.paths(spec)["path"]
    manager = FakeLaunchd(path)
    monkeypatch.setattr(common, "locate_command", lambda name: SimpleNamespace(command=("/fake/" + name,)))
    monkeypatch.setattr(common.subprocess, "run", manager.run)
    return spec, path, manager


def _seed(path, manager):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"original plist bytes\n")
    path.chmod(0o640)
    manager.loaded = True


def test_enable_user_agent_and_restore_exact_prior_state(environment):
    spec, path, manager = environment
    _seed(path, manager)
    before = common.snapshot_file(path)
    manager.enabled = False
    handle = scheduler.enable(spec)
    assert manager.loaded and manager.enabled
    assert common.snapshot_file(path) != before
    assert handle.rollback()["ok"]
    assert common.snapshot_file(path) == before
    assert manager.loaded and not manager.enabled
    assert all("system" not in arg for args in manager.calls for arg in args)


def test_new_agent_rollback_removes_file(environment):
    spec, path, manager = environment
    handle = scheduler.enable(spec)
    assert handle.rollback()["ok"]
    assert not path.exists()
    assert not manager.loaded


def test_disable_verifies_unload_and_restores_loaded_job(environment):
    spec, path, manager = environment
    _seed(path, manager)
    handle = scheduler.disable(spec)
    assert handle.removed and not manager.loaded
    assert not path.exists()
    assert handle.rollback()["ok"]
    assert manager.loaded and not manager.running


def test_bootstrap_failure_restores_original_bytes_and_mode(environment):
    spec, path, manager = environment
    _seed(path, manager)
    before = common.snapshot_file(path)
    manager.fail_once = "bootstrap"
    with pytest.raises(RuntimeError, match="injected manager failure"):
        scheduler.enable(spec)
    assert common.snapshot_file(path) == before
    assert manager.loaded


def test_success_exit_without_adoption_is_not_success(environment):
    spec, path, manager = environment
    manager.ignore_bootstrap = True
    with pytest.raises(RuntimeError, match="did not adopt"):
        scheduler.enable(spec)
    assert not path.exists()


def test_unrestorable_manager_state_has_recovery_receipt(environment):
    spec, path, manager = environment
    _seed(path, manager)
    manager.ignore_bootstrap = True
    with pytest.raises(scheduler.SchedulerRecoveryError) as raised:
        scheduler.enable(spec)
    assert not raised.value.receipt["ok"]
    assert raised.value.receipt["files"][0]["ok"]


def test_foreign_loaded_job_is_not_unloaded(environment):
    spec, path, manager = environment
    _seed(path, manager)
    manager.foreign_path = "/foreign/agent.plist"
    with pytest.raises(RuntimeError, match="another or unverifiable path"):
        scheduler.disable(spec)
    assert all(args[0] in {"print", "print-disabled"} for args in manager.calls)
    assert manager.loaded


def test_missing_file_with_loaded_job_is_refused(environment):
    spec, path, manager = environment
    manager.loaded = True
    with pytest.raises(RuntimeError, match="without its scheduler file"):
        scheduler.disable(spec)
    assert manager.loaded
    assert not path.exists()


def test_symlink_refused_before_manager_mutation(environment, tmp_path):
    spec, path, manager = environment
    target = tmp_path / "foreign"
    target.write_text("keep")
    path.parent.mkdir(parents=True)
    path.symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink"):
        scheduler.enable(spec)
    assert manager.calls == []
    assert target.read_text() == "keep"


@pytest.mark.parametrize("operation", [scheduler.enable, scheduler.disable])
def test_running_updater_is_never_unloaded(environment, operation):
    spec, path, manager = environment
    _seed(path, manager)
    manager.running = True
    before = common.snapshot_file(path)
    with pytest.raises(RuntimeError, match="updater is active"):
        operation(spec)
    assert all(args[0] in {"print", "print-disabled"} for args in manager.calls)
    assert manager.running
    assert common.snapshot_file(path) == before


def test_rollback_does_not_kill_unexpected_active_updater(environment):
    spec, path, manager = environment
    handle = scheduler.enable(spec)
    before = common.snapshot_file(path)
    manager.running = True
    receipt = handle.rollback()
    assert not receipt["ok"]
    assert manager.running
    assert common.snapshot_file(path) == before


def test_positive_pid_also_protects_starting_updater(environment):
    spec, path, manager = environment
    _seed(path, manager)
    manager.reported_pid = 4321
    with pytest.raises(RuntimeError, match="updater is active"):
        scheduler.disable(spec)
    assert all(args[0] in {"print", "print-disabled"} for args in manager.calls)
    assert path.exists()
