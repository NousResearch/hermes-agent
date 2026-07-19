from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
import threading
import time as time_module
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

import pytest

from hermes_cli import update_auto
from hermes_cli.subcommands.update import build_update_parser
from hermes_cli.update_lock import UpdateLock


def _args(**overrides):
    base = {"branch": None, "force": False}
    base.update(overrides)
    return SimpleNamespace(**base)


def _read_status(hermes_home):
    return json.loads((hermes_home / "state" / "update-status.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _isolate_auto_update_lock(tmp_path, monkeypatch):
    lock_path = tmp_path / "source-checkout" / ".git" / "hermes-update.lock"
    monkeypatch.setattr(
        update_auto,
        "acquire_update_lock",
        lambda _project_root: UpdateLock(lock_path).acquire(),
    )


def test_write_status_uses_stable_schema_and_profile_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "profile-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    path = update_auto.write_status(
        {
            "status": update_auto.STATUS_SUCCESS,
            "previousVersion": "old",
            "latestVersion": "new",
            "currentVersion": "new",
            "backupPath": "/tmp/backup.zip",
        }
    )

    assert path == hermes_home / "state" / "update-status.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert list(data) == sorted(update_auto.DEFAULT_STATUS)
    assert data["mode"] == "manual"
    assert data["enabled"] is False
    assert data["schedule"] is None
    assert data["status"] == update_auto.STATUS_SUCCESS
    assert data["backupPath"] == "/tmp/backup.zip"
    assert data["error"] is None
    assert data["logPath"] == str(hermes_home / "logs" / "update.log")


def test_stale_generation_receipt_cannot_replace_new_running_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

    update_auto.write_status(
        {"status": update_auto.STATUS_RUNNING, "runGeneration": "new-generation"},
        expected_run_generation=None,
    )

    with pytest.raises(update_auto.StaleStatusWriteError):
        update_auto.write_status(
            {
                "status": update_auto.STATUS_SUCCESS,
                "runGeneration": "old-generation",
                "terminalReceipt": {
                    "generation": "old-generation",
                    "status": update_auto.STATUS_SUCCESS,
                },
            },
            expected_run_generation="old-generation",
        )

    status = update_auto.read_status()
    assert status["runGeneration"] == "new-generation"
    assert status["status"] == update_auto.STATUS_RUNNING
    assert status["terminalReceipt"] is None


def test_current_version_uses_fresh_distribution_metadata_for_non_git_install(
    tmp_path, monkeypatch
):
    from hermes_cli import main as hm

    monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        update_auto.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("not a git checkout")),
    )
    metadata_calls = []
    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda name: metadata_calls.append(name) or "9.9.9",
    )
    monkeypatch.setattr("hermes_cli.__version__", "stale-imported-version")

    assert update_auto._current_version() == "9.9.9"
    assert metadata_calls == ["hermes-agent"]


def test_pip_auto_update_accepts_a_fresh_new_distribution_version(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("hermes_cli.__version__", "stale-imported-version")

    from hermes_cli import backup
    from hermes_cli import main as hm

    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: {
            "update_available": True,
            "install_method": "pip",
            "latest_version": "2.0.0",
        },
    )
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: hermes_home / "backups" / "pre-update.zip",
    )
    monkeypatch.setattr(hm, "cmd_update", lambda _args: None)
    monkeypatch.setattr(update_auto, "_verify_health", lambda *_args: (True, "ok"))
    monkeypatch.setattr(
        hm,
        "PROJECT_ROOT",
        tmp_path / "pip-install",
    )
    versions = iter(["1.0.0", "2.0.0"])
    metadata_calls = []
    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda name: metadata_calls.append(name) or next(versions),
    )
    monkeypatch.setattr(
        update_auto.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("not a git checkout")),
    )

    update_auto.cmd_auto_run_now(_args())

    assert metadata_calls == ["hermes-agent", "hermes-agent"]
    assert _read_status(hermes_home)["status"] == update_auto.STATUS_SUCCESS


def test_live_process_metadata_does_not_use_proc_pid_paths_on_darwin(monkeypatch):
    from gateway import status as gateway_status

    monkeypatch.setattr(update_auto.sys, "platform", "darwin")
    monkeypatch.setattr(gateway_status, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(gateway_status, "get_process_start_time", lambda _pid: 456)
    monkeypatch.setattr(
        gateway_status, "_read_process_cmdline", lambda _pid: "hermes gateway run"
    )

    metadata = update_auto._live_process_metadata(123)

    assert metadata == {
        "pid": 123,
        "start_time": 456,
        "command": "hermes gateway run",
    }


def test_git_runtime_verification_matches_revision_not_semver(monkeypatch):
    from gateway import status as gateway_status

    profile_home = Path("/profile-a")
    monkeypatch.setattr(update_auto, "get_hermes_home", lambda: profile_home)
    monkeypatch.setattr(update_auto, "_installation_identity", lambda: "install-a")
    before = {
        "gateway_state": "running",
        "pid": 101,
        "start_time": 1,
        "runtime_version": "1.0.0",
        "runtime_revision": "oldsha1234567",
        "installation_identity": "install-a",
        "profile_home": str(profile_home),
        "_live_validated": True,
    }
    after = {
        "gateway_state": "running",
        "pid": 202,
        "start_time": 2,
        "runtime_version": "2.0.0",
        "runtime_revision": "newsha1234567890",
        "installation_identity": "install-a",
        "profile_home": str(profile_home),
    }
    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: after)
    monkeypatch.setattr(
        update_auto,
        "_live_gateway_identity",
        lambda runtime, **_kwargs: {
            "pid": runtime["pid"],
            "start_time": runtime["start_time"],
            "command": "hermes gateway run",
            "runtime_version": runtime["runtime_version"],
            "runtime_revision": runtime["runtime_revision"],
            "installation_identity": runtime["installation_identity"],
            "profile_home": runtime["profile_home"],
        },
    )
    monkeypatch.setattr(update_auto, "_HEALTH_STARTUP_GRACE_SECONDS", 0.0)

    ok, detail = update_auto._verify_health(
        before, expected_revision="newsha1234567890"
    )

    assert ok is True
    assert "202" in detail


def test_verify_health_accepts_absent_status_after_live_validated_stopped_pre_state(
    monkeypatch,
):
    from gateway import status as gateway_status

    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: None)
    monkeypatch.setattr(gateway_status, "get_running_pid", lambda: None)
    monkeypatch.setattr(update_auto, "_HEALTH_STARTUP_GRACE_SECONDS", 0.0)

    captured = update_auto._capture_gateway_runtime()
    assert captured is not None
    assert captured["gateway_state"] == "stopped"

    ok, detail = update_auto._verify_health(captured)

    assert ok is True
    assert "stopped" in detail


def test_scheduler_status_rmw_cannot_overwrite_terminal_receipt(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    update_auto.write_status(
        {"status": update_auto.STATUS_RUNNING, "runGeneration": "generation-a"},
        expected_run_generation=None,
    )

    original_read_status = update_auto.read_status
    snapshot_ready = threading.Event()
    release_scheduler = threading.Event()
    first_read = True

    def gated_read_status():
        nonlocal first_read
        snapshot = original_read_status()
        if first_read:
            first_read = False
            snapshot_ready.set()
            release_scheduler.wait(timeout=2)
        return snapshot

    monkeypatch.setattr(update_auto, "read_status", gated_read_status)
    scheduler_errors = []
    leader_done = threading.Event()

    def scheduler_update():
        try:
            update_auto.update_status_fields(
                mode="scheduled",
                enabled=True,
                schedule="03:00",
            )
        except Exception as exc:  # pragma: no cover - assertion below reports it
            scheduler_errors.append(exc)

    def terminal_success():
        from hermes_cli.update_lock import UpdateLockBusyError

        try:
            while True:
                try:
                    update_auto.write_status(
                        {
                            "status": update_auto.STATUS_SUCCESS,
                            "runGeneration": "generation-a",
                            "terminalReceipt": {
                                "generation": "generation-a",
                                "status": update_auto.STATUS_SUCCESS,
                            },
                        },
                        expected_run_generation="generation-a",
                    )
                    break
                except UpdateLockBusyError:
                    time_module.sleep(0.01)
        finally:
            leader_done.set()

    scheduler_thread = threading.Thread(target=scheduler_update)
    scheduler_thread.start()
    assert snapshot_ready.wait(timeout=2)

    leader_thread = threading.Thread(target=terminal_success)
    leader_thread.start()
    leader_done.wait(timeout=0.2)
    release_scheduler.set()
    scheduler_thread.join(timeout=2)
    leader_thread.join(timeout=2)

    assert scheduler_errors == []
    final = update_auto.read_status()
    assert final["status"] == update_auto.STATUS_SUCCESS
    assert final["terminalReceipt"]["generation"] == "generation-a"


def test_launchd_enable_requires_final_loaded_state(tmp_path, monkeypatch):
    _hermes_home, _calls = _set_macos_scheduler_env(tmp_path, monkeypatch)
    monkeypatch.setattr(
        update_auto,
        "_run_launchctl",
        lambda args: subprocess.CompletedProcess(
            ["launchctl"] + args,
            1 if args[0] == "print" else 0,
            stdout="",
            stderr="Could not find service" if args[0] == "print" else "",
        ),
    )

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_enable(_args(time="03:00"))

    assert exc.value.code == 1
    assert update_auto.read_status()["enabled"] is False


def test_systemd_enable_requires_persistent_timer_enablement(tmp_path, monkeypatch):
    service_path = tmp_path / "hermes.service"
    timer_path = tmp_path / "hermes.timer"
    service_path.write_text("service", encoding="utf-8")
    timer_path.write_text("timer", encoding="utf-8")

    states = {
        service_path.name: {
            "load_state": "loaded",
            "unit_file_state": "static",
            "active_state": "inactive",
        },
        timer_path.name: {
            "load_state": "loaded",
            "unit_file_state": "enabled-runtime",
            "active_state": "active",
        },
    }
    monkeypatch.setattr(update_auto, "_systemd_state", lambda name: states[name])

    with pytest.raises(RuntimeError, match="persistent"):
        update_auto._verify_systemd_enabled(service_path, timer_path)


def test_scheduler_reports_nonzero_when_run_loses_status_ownership(
    monkeypatch,
):
    monkeypatch.setattr(
        update_auto,
        "_read_persisted_scheduler_status",
        lambda: {"enabled": True},
    )
    monkeypatch.setattr(update_auto, "_validate_scheduler_status", lambda _status: None)
    monkeypatch.setattr(update_auto, "_scheduled_action_for_now", lambda _status: "run")
    monkeypatch.setattr(
        update_auto,
        "_cmd_auto_run_now_locked_impl",
        lambda _args, **_kwargs: (_ for _ in ()).throw(
            update_auto.StaleStatusWriteError("ownership lost")
        ),
    )

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_scheduled(_args())

    assert exc.value.code == update_auto.EXIT_UPDATE_FAILED


def test_pip_auto_update_rejects_an_unchanged_fresh_distribution_version(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("hermes_cli.__version__", "stale-imported-version")

    from hermes_cli import backup
    from hermes_cli import main as hm

    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: {
            "update_available": True,
            "install_method": "pip",
            "latest_version": "2.0.0",
        },
    )
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: hermes_home / "backups" / "pre-update.zip",
    )
    monkeypatch.setattr(hm, "cmd_update", lambda _args: None)
    monkeypatch.setattr(update_auto, "_verify_health", lambda *_args: (True, "ok"))
    monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path / "pip-install")
    metadata_calls = []
    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda name: metadata_calls.append(name) or "1.0.0",
    )
    monkeypatch.setattr(
        update_auto.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("not a git checkout")),
    )

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_UPDATE_FAILED
    assert metadata_calls == ["hermes-agent", "hermes-agent", "hermes-agent"]
    assert "without applying" in _read_status(hermes_home)["error"]


def test_pip_auto_update_records_pip_failure(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import backup
    from hermes_cli import main as hm

    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: {
            "update_available": True,
            "install_method": "pip",
            "latest_version": "2.0.0",
        },
    )
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: hermes_home / "backups" / "pre-update.zip",
    )
    monkeypatch.setattr(hm, "cmd_update", lambda _args: (_ for _ in ()).throw(SystemExit(7)))
    monkeypatch.setattr(update_auto, "_current_version", lambda: "1.0.0")

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_UPDATE_FAILED
    assert "exit code 7" in _read_status(hermes_home)["error"]


def test_status_output_when_no_status_file_exists(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(update_auto, "_current_version", lambda: "abc123")

    update_auto.cmd_auto_status(SimpleNamespace())

    out = capsys.readouterr().out
    assert "Phase:            2 (scheduled wrapper around run-now)" in out
    assert "Mode:             manual" in out
    assert "Enabled:          no" in out
    assert "Schedule:         not configured" in out
    assert "Scheduler:        not configured" in out
    assert "Last run:         never" in out
    assert "Last result:      not_configured" in out
    assert "Current version:  abc123" in out
    assert str(hermes_home / "logs" / "update.log") in out


def test_status_output_after_failure(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    update_auto.write_status(
        {
            "lastRunAt": "2026-05-27T12:00:00+00:00",
            "status": update_auto.STATUS_HEALTH_FAILED,
            "previousVersion": "abc123",
            "latestVersion": "def456",
            "currentVersion": "def456",
            "backupPath": str(hermes_home / "backups" / "pre-update.zip"),
            "error": "gateway startup failed",
        }
    )

    update_auto.cmd_auto_status(SimpleNamespace())

    out = capsys.readouterr().out
    assert "Enabled:          no" in out
    assert "Schedule:         not configured" in out
    assert "Last result:      health_failed" in out
    assert "Previous version: abc123" in out
    assert "Latest known:     def456" in out
    assert "Current version:  def456" in out
    assert "gateway startup failed" in out


def test_status_output_tolerates_corrupt_status_file(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    status_path = hermes_home / "state" / "update-status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(update_auto, "_current_version", lambda: "abc123")

    update_auto.cmd_auto_status(SimpleNamespace())

    out = capsys.readouterr().out
    assert "Last result:      not_configured" in out
    assert "Current version:  abc123" in out


def _set_macos_scheduler_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(update_auto.sys, "platform", "darwin")
    monkeypatch.setattr(update_auto.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(update_auto, "_hermes_command_prefix", lambda: ["hermes"])
    calls = []
    state = {"loaded": False, "disabled": False}

    def fake_launchctl(args):
        calls.append(args)
        if args[0] == "print":
            if state["loaded"]:
                return subprocess.CompletedProcess(
                    ["launchctl"] + args,
                    0,
                    stdout="state = running\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(
                ["launchctl"] + args,
                1,
                stdout="",
                stderr="Could not find service",
            )
        if args[0] == "print-disabled":
            disabled = "true" if state["disabled"] else "false"
            return subprocess.CompletedProcess(
                ["launchctl"] + args,
                0,
                stdout=f'"{update_auto._launchd_label()}" => {disabled}\n',
                stderr="",
            )
        if args[0] == "bootstrap":
            state["loaded"] = True
        elif args[0] == "bootout":
            state["loaded"] = False
        elif args[0] == "enable":
            state["disabled"] = False
        elif args[0] == "disable":
            state["disabled"] = True
        return subprocess.CompletedProcess(["launchctl"] + args, 0, stdout="", stderr="")

    monkeypatch.setattr(update_auto, "_run_launchctl", fake_launchctl)
    return hermes_home, calls


def test_launchd_target_requires_posix_user_session(monkeypatch):
    monkeypatch.delattr(update_auto.os, "getuid", raising=False)

    with pytest.raises(RuntimeError, match="POSIX user session"):
        update_auto._launchd_target()


def test_enable_creates_expected_launchd_plist_on_macos(tmp_path, monkeypatch, capsys):
    hermes_home, calls = _set_macos_scheduler_env(tmp_path, monkeypatch)

    update_auto.cmd_auto_enable(_args(time="03:00"))

    plist_path = update_auto._launchd_plist_path()
    assert plist_path.exists()
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)
    assert plist["Label"] == update_auto._launchd_label()
    assert plist["ProgramArguments"] == [
        "hermes", "--profile", "default", "update", "auto", "run-scheduled"
    ]
    assert plist["StartCalendarInterval"] == {"Hour": 3, "Minute": 0}
    assert plist["EnvironmentVariables"] == {"HERMES_HOME": str(hermes_home)}
    assert plist["StandardOutPath"] == str(hermes_home / "logs" / "update-auto.out.log")
    assert plist["StandardErrorPath"] == str(hermes_home / "logs" / "update-auto.err.log")
    assert any(call[0] == "bootstrap" for call in calls)
    assert any(call[0] == "enable" for call in calls)

    status = _read_status(hermes_home)
    assert status["enabled"] is True
    assert status["mode"] == "scheduled"
    assert status["schedule"] == "03:00"
    assert status["schedulerType"] == "launchd"
    assert status["schedulerPath"] == str(plist_path)
    out = capsys.readouterr().out
    assert "Hermes auto-update scheduled" in out
    assert "03:00" in out


def test_scheduled_command_pins_default_profile_selector(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default-home"))
    monkeypatch.setattr(update_auto, "_hermes_command_prefix", lambda: ["hermes"])

    assert update_auto._scheduled_command() == [
        "hermes",
        "--profile",
        "default",
        "update",
        "auto",
        "run-scheduled",
    ]


def test_default_scheduler_stays_default_after_active_profile_switch(
    tmp_path, monkeypatch
):
    hermes_root = tmp_path / ".hermes"
    named_profile = hermes_root / "profiles" / "work"
    named_profile.mkdir(parents=True)
    hermes_root.mkdir(exist_ok=True)
    (hermes_root / "active_profile").write_text("default", encoding="utf-8")

    monkeypatch.setattr(update_auto.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(hermes_root))
    monkeypatch.setattr(update_auto.sys, "platform", "darwin")
    monkeypatch.setattr(update_auto, "_hermes_command_prefix", lambda: ["hermes"])
    monkeypatch.setattr(
        update_auto,
        "_run_launchctl",
        lambda args: subprocess.CompletedProcess(
            ["launchctl"] + args,
            1 if args[0] == "print" else 0,
            stdout="",
            stderr="Could not find service" if args[0] == "print" else "",
        ),
    )
    monkeypatch.setattr(
        update_auto,
        "_launchd_state",
        lambda _target: {
            "loaded": update_auto._launchd_plist_path().is_file(),
            "enabled": True,
            "running": False,
        },
    )

    update_auto.cmd_auto_enable(_args(time="03:00"))
    plist_path = update_auto._launchd_plist_path()
    with plist_path.open("rb") as handle:
        scheduled_argv = plistlib.load(handle)["ProgramArguments"]
    assert scheduled_argv[:3] == ["hermes", "--profile", "default"]
    default_identity = update_auto.read_status()["schedulerIdentity"]

    # The user switches the sticky active profile after the scheduler was
    # enabled. Execute the persisted command's profile-selection prefix in the
    # same way a fresh Hermes process does; it must still resolve the default.
    (hermes_root / "active_profile").write_text("work", encoding="utf-8")
    monkeypatch.delenv("HERMES_HOME")
    monkeypatch.setattr(sys, "argv", scheduled_argv)
    from hermes_cli import main

    main._apply_profile_override()

    assert os.environ["HERMES_HOME"] == str(hermes_root)
    assert update_auto.get_status_path().is_relative_to(hermes_root)
    assert update_auto._scheduler_identity() == default_identity

    dispatched_homes = []
    monkeypatch.setattr(
        update_auto, "cmd_auto_run_now", lambda _args: dispatched_homes.append(update_auto.get_hermes_home())
    )
    update_auto.cmd_auto_run_scheduled(_args())
    assert dispatched_homes == [hermes_root]
    assert not (named_profile / "state" / update_auto.STATUS_FILENAME).exists()


def test_enable_launchd_requires_bootstrap_success_and_reports_both_streams(
    tmp_path, monkeypatch, capsys
):
    _hermes_home, calls = _set_macos_scheduler_env(tmp_path, monkeypatch)

    def failed_bootstrap(args):
        calls.append(args)
        if args[0] == "print":
            return subprocess.CompletedProcess(
                ["launchctl"] + args,
                1,
                stdout="",
                stderr="Could not find service",
            )
        if args[0] == "bootstrap":
            return subprocess.CompletedProcess(
                ["launchctl"] + args,
                1,
                stdout="bootstrap stdout",
                stderr="bootstrap stderr",
            )
        return subprocess.CompletedProcess(["launchctl"] + args, 0, stdout="", stderr="")

    monkeypatch.setattr(update_auto, "_run_launchctl", failed_bootstrap)

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_enable(_args(time="03:00"))

    assert exc.value.code == 1
    assert update_auto.read_status()["enabled"] is False
    error = capsys.readouterr().err
    assert "bootstrap stdout" in error
    assert "bootstrap stderr" in error


def test_enable_launchd_requires_enable_success(tmp_path, monkeypatch, capsys):
    _hermes_home, calls = _set_macos_scheduler_env(tmp_path, monkeypatch)

    def failed_enable(args):
        calls.append(args)
        if args[0] == "print":
            return subprocess.CompletedProcess(
                ["launchctl"] + args,
                1,
                stdout="",
                stderr="Could not find service",
            )
        if args[0] == "enable":
            return subprocess.CompletedProcess(
                ["launchctl"] + args,
                1,
                stdout="enable stdout",
                stderr="enable stderr",
            )
        return subprocess.CompletedProcess(["launchctl"] + args, 0, stdout="", stderr="")

    monkeypatch.setattr(update_auto, "_run_launchctl", failed_enable)

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_enable(_args(time="03:00"))

    assert exc.value.code == 1
    assert update_auto.read_status()["enabled"] is False
    error = capsys.readouterr().err
    assert "enable stdout" in error
    assert "enable stderr" in error


def test_enable_launchd_reconfiguration_rolls_back_an_already_enabled_job(
    tmp_path, monkeypatch
):
    _hermes_home, calls = _set_macos_scheduler_env(tmp_path, monkeypatch)
    plist_path = update_auto._launchd_plist_path()
    prior_plist = b"prior launchd plist bytes\n"
    plist_path.parent.mkdir(parents=True)
    plist_path.write_bytes(prior_plist)

    def launchctl(args):
        calls.append(args)
        if args[0] == "print":
            return subprocess.CompletedProcess(
                ["launchctl"] + args,
                0,
                stdout="state = running\n",
                stderr="",
            )
        if args[0] == "print-disabled":
            return subprocess.CompletedProcess(
                ["launchctl"] + args,
                0,
                stdout='"com.hermes.agent.auto-update" => false\n',
                stderr="",
            )
        if args[0] == "enable":
            return subprocess.CompletedProcess(
                ["launchctl"] + args,
                1,
                stdout="enable stdout",
                stderr="enable stderr",
            )
        return subprocess.CompletedProcess(["launchctl"] + args, 0, stdout="", stderr="")

    monkeypatch.setattr(update_auto, "_run_launchctl", launchctl)

    with pytest.raises(SystemExit):
        update_auto.cmd_auto_enable(_args(time="04:00"))

    assert plist_path.read_bytes() == prior_plist
    assert any(call[:2] == ["bootstrap", f"gui/{update_auto.os.getuid()}"] for call in calls)
    assert any(call[0] == "kickstart" for call in calls)


def test_enable_systemd_reconfiguration_rolls_back_an_already_enabled_timer(
    tmp_path, monkeypatch
):
    _hermes_home, _calls = _set_linux_scheduler_env(tmp_path, monkeypatch)
    systemd_dir = tmp_path / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True)
    service_path, timer_path = update_auto._systemd_paths()
    prior_service = "prior service unit\n"
    prior_timer = "prior timer unit\n"
    service_path.write_text(prior_service, encoding="utf-8")
    timer_path.write_text(prior_timer, encoding="utf-8")
    calls = []

    def systemctl(args):
        calls.append(args)
        if args[0] == "show":
            return subprocess.CompletedProcess(
                ["systemctl", "--user"] + args,
                0,
                stdout="LoadState=loaded\nUnitFileState=enabled\nActiveState=active\n",
                stderr="",
            )
        if args[:2] == ["enable", "--now"]:
            return subprocess.CompletedProcess(
                ["systemctl", "--user"] + args,
                1,
                stdout="enable stdout",
                stderr="enable stderr",
            )
        return subprocess.CompletedProcess(["systemctl", "--user"] + args, 0, stdout="", stderr="")

    monkeypatch.setattr(update_auto, "_systemctl_user", systemctl)

    with pytest.raises(SystemExit):
        update_auto.cmd_auto_enable(_args(time="04:00"))

    assert service_path.read_text(encoding="utf-8") == prior_service
    assert timer_path.read_text(encoding="utf-8") == prior_timer
    assert ["disable", "--now", timer_path.name] in calls
    assert ["enable", timer_path.name] in calls
    assert ["start", timer_path.name] in calls



def test_enable_systemd_requires_daemon_reload_success(tmp_path, monkeypatch, capsys):
    hermes_home, _calls = _set_linux_scheduler_env(tmp_path, monkeypatch)
    _service_path, timer_path = update_auto._systemd_paths()
    calls = []

    def failed_reload(args):
        calls.append(args)
        if args == ["daemon-reload"]:
            return subprocess.CompletedProcess(
                ["systemctl", "--user"] + args,
                1,
                stdout="reload stdout",
                stderr="reload stderr",
            )
        return subprocess.CompletedProcess(["systemctl", "--user"] + args, 0, stdout="", stderr="")

    monkeypatch.setattr(update_auto, "_systemctl_user", failed_reload)

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_enable(_args(time="03:00"))

    assert exc.value.code == 1
    assert calls == [
        [
            "show",
            timer_path.name,
            "--property=LoadState,UnitFileState,ActiveState",
        ],
        [
            "show",
            _service_path.name,
            "--property=LoadState,UnitFileState,ActiveState",
        ],
        ["daemon-reload"],
        ["stop", _service_path.name],
        ["disable", _service_path.name],
        ["disable", "--now", timer_path.name],
        ["daemon-reload"],
        [
            "show",
            _service_path.name,
            "--property=LoadState,UnitFileState,ActiveState",
        ],
        [
            "show",
            timer_path.name,
            "--property=LoadState,UnitFileState,ActiveState",
        ],
    ]
    assert update_auto.read_status()["enabled"] is False
    error = capsys.readouterr().err
    assert "reload stdout" in error
    assert "reload stderr" in error


def test_enable_systemd_requires_enable_now_success(tmp_path, monkeypatch, capsys):
    hermes_home, _calls = _set_linux_scheduler_env(tmp_path, monkeypatch)
    _service_path, timer_path = update_auto._systemd_paths()
    calls = []

    def failed_enable(args):
        calls.append(args)
        if args == ["enable", "--now", timer_path.name]:
            return subprocess.CompletedProcess(
                ["systemctl", "--user"] + args,
                1,
                stdout="enable stdout",
                stderr="enable stderr",
            )
        return subprocess.CompletedProcess(["systemctl", "--user"] + args, 0, stdout="", stderr="")

    monkeypatch.setattr(update_auto, "_systemctl_user", failed_enable)

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_enable(_args(time="03:00"))

    assert exc.value.code == 1
    assert calls == [
        [
            "show",
            timer_path.name,
            "--property=LoadState,UnitFileState,ActiveState",
        ],
        [
            "show",
            _service_path.name,
            "--property=LoadState,UnitFileState,ActiveState",
        ],
        ["daemon-reload"],
        ["enable", "--now", timer_path.name],
        ["stop", _service_path.name],
        ["disable", _service_path.name],
        ["disable", "--now", timer_path.name],
        ["daemon-reload"],
        [
            "show",
            _service_path.name,
            "--property=LoadState,UnitFileState,ActiveState",
        ],
        [
            "show",
            timer_path.name,
            "--property=LoadState,UnitFileState,ActiveState",
        ],
    ]
    assert update_auto.read_status()["enabled"] is False
    error = capsys.readouterr().err
    assert "enable stdout" in error
    assert "enable stderr" in error


def test_enable_with_plan_time_creates_single_launchd_scheduler_with_two_triggers(tmp_path, monkeypatch):
    hermes_home, _calls = _set_macos_scheduler_env(tmp_path, monkeypatch)

    update_auto.cmd_auto_enable(_args(time="04:00", plan_time=["21:00"]))

    plist_path = update_auto._launchd_plist_path()
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)
    assert plist["ProgramArguments"] == [
        "hermes", "--profile", "default", "update", "auto", "run-scheduled"
    ]
    assert plist["StartCalendarInterval"] == [
        {"Hour": 21, "Minute": 0},
        {"Hour": 4, "Minute": 0},
    ]
    status = _read_status(hermes_home)
    assert status["schedule"] == "04:00"
    assert status["planSchedule"] == ["21:00"]


def test_enable_is_idempotent_and_updates_existing_launchd_plist(tmp_path, monkeypatch):
    hermes_home, _calls = _set_macos_scheduler_env(tmp_path, monkeypatch)

    update_auto.cmd_auto_enable(_args(time="03:00"))
    update_auto.cmd_auto_enable(_args(time="04:30"))

    plist_path = update_auto._launchd_plist_path()
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)
    assert plist["StartCalendarInterval"] == {"Hour": 4, "Minute": 30}
    status = _read_status(hermes_home)
    assert status["enabled"] is True
    assert status["schedule"] == "04:30"


def test_disable_removes_only_hermes_launchd_plist(tmp_path, monkeypatch, capsys):
    hermes_home, calls = _set_macos_scheduler_env(tmp_path, monkeypatch)
    launch_agents = tmp_path / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True)
    hermes_plist = update_auto._launchd_plist_path()
    other_plist = launch_agents / "com.example.other.plist"
    hermes_plist.write_text("hermes", encoding="utf-8")
    other_plist.write_text("other", encoding="utf-8")
    update_auto.write_status(
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "03:00",
            "schedulerType": "launchd",
            "schedulerPath": str(hermes_plist),
            "schedulerIdentity": update_auto._scheduler_identity(),
        }
    )

    update_auto.cmd_auto_disable(_args())

    assert not hermes_plist.exists()
    assert other_plist.exists()
    assert any(call[0] == "bootout" for call in calls)
    status = _read_status(hermes_home)
    assert status["enabled"] is False
    assert status["schedule"] is None
    assert status["schedulerType"] is None
    assert status["schedulerPath"] is None
    assert "disabled" in capsys.readouterr().out


def test_disable_launchd_keeps_file_and_status_when_bootout_fails(
    tmp_path, monkeypatch, capsys
):
    hermes_home, calls = _set_macos_scheduler_env(tmp_path, monkeypatch)
    launch_agents = tmp_path / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True)
    plist_path = update_auto._launchd_plist_path()
    plist_path.write_text("hermes", encoding="utf-8")
    update_auto.write_status(
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "03:00",
            "schedulerType": "launchd",
            "schedulerPath": str(plist_path),
            "schedulerIdentity": update_auto._scheduler_identity(),
        }
    )

    def failed_bootout(args):
        calls.append(args)
        if args[0] == "print":
            return subprocess.CompletedProcess(
                ["launchctl"] + args,
                1,
                stdout="",
                stderr="Could not find service",
            )
        return subprocess.CompletedProcess(
            ["launchctl"] + args,
            1,
            stdout="bootout stdout",
            stderr="bootout stderr",
        )

    monkeypatch.setattr(update_auto, "_run_launchctl", failed_bootout)

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_disable(_args())

    assert exc.value.code == 1
    assert plist_path.exists()
    assert _read_status(hermes_home)["enabled"] is True
    error = capsys.readouterr().err
    assert "bootout stdout" in error
    assert "bootout stderr" in error


def test_disable_is_idempotent_when_launchd_plist_missing(tmp_path, monkeypatch, capsys):
    hermes_home, calls = _set_macos_scheduler_env(tmp_path, monkeypatch)

    update_auto.cmd_auto_disable(_args())

    assert calls
    status = _read_status(hermes_home)
    assert status["enabled"] is False
    assert status["schedule"] is None
    assert "already disabled" in capsys.readouterr().out


@pytest.mark.parametrize("bad_time", ["3:00", "24:00", "03:60", "0300", "ab:cd"])
def test_enable_rejects_invalid_time_without_launchctl(tmp_path, monkeypatch, bad_time, capsys):
    _hermes_home, calls = _set_macos_scheduler_env(tmp_path, monkeypatch)

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_enable(_args(time=bad_time))

    assert exc.value.code == 2
    assert calls == []
    assert "Invalid schedule time" in capsys.readouterr().err


def _set_linux_scheduler_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(update_auto.sys, "platform", "linux")
    monkeypatch.setattr(update_auto.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(update_auto, "_hermes_command_prefix", lambda: ["hermes"])
    monkeypatch.setattr(update_auto.shutil, "which", lambda name: "/usr/bin/systemctl" if name == "systemctl" else None)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if len(cmd) >= 4 and cmd[0:3] == ["systemctl", "--user", "show"]:
            service_path, timer_path = update_auto._systemd_paths()
            unit_path = service_path if cmd[3] == service_path.name else timer_path
            if unit_path.exists():
                active = "active" if unit_path == timer_path else "inactive"
                unit_file = "enabled" if unit_path == timer_path else "static"
                output = (
                    "LoadState=loaded\n"
                    f"UnitFileState={unit_file}\n"
                    f"ActiveState={active}\n"
                )
            else:
                output = (
                    "LoadState=not-found\n"
                    "UnitFileState=not-found\n"
                    "ActiveState=inactive\n"
                )
            return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(update_auto.subprocess, "run", fake_run)
    return hermes_home, calls


def test_enable_creates_expected_systemd_user_timer_on_linux(tmp_path, monkeypatch):
    hermes_home, calls = _set_linux_scheduler_env(tmp_path, monkeypatch)

    update_auto.cmd_auto_enable(_args(time="03:00"))

    systemd_dir = tmp_path / ".config" / "systemd" / "user"
    service_path, timer_path = update_auto._systemd_paths()
    service_text = service_path.read_text(encoding="utf-8")
    timer_text = timer_path.read_text(encoding="utf-8")
    assert "ExecStart=hermes --profile default update auto run-scheduled" in service_text
    assert f"Environment=HERMES_HOME={hermes_home}" in service_text
    assert f"StandardOutput=append:{hermes_home / 'logs' / 'update-auto.out.log'}" in service_text
    assert "OnCalendar=*-*-* 03:00:00" in timer_text
    assert "Persistent=true" in timer_text
    assert ["systemctl", "--version"] in calls
    assert ["systemctl", "--user", "daemon-reload"] in calls
    assert ["systemctl", "--user", "enable", "--now", timer_path.name] in calls

    status = _read_status(hermes_home)
    assert status["enabled"] is True
    assert status["schedule"] == "03:00"
    assert status["schedulerType"] == "systemd-user"
    assert status["schedulerPath"] == str(timer_path)


def test_disable_removes_only_hermes_systemd_user_files(tmp_path, monkeypatch):
    hermes_home, calls = _set_linux_scheduler_env(tmp_path, monkeypatch)
    systemd_dir = tmp_path / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True)
    service_path, timer_path = update_auto._systemd_paths()
    other_timer = systemd_dir / "other.timer"
    service_path.write_text("service", encoding="utf-8")
    timer_path.write_text("timer", encoding="utf-8")
    other_timer.write_text("other", encoding="utf-8")
    update_auto.write_status(
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "03:00",
            "schedulerType": "systemd-user",
            "schedulerPath": str(timer_path),
            "schedulerIdentity": update_auto._scheduler_identity(),
        }
    )

    update_auto.cmd_auto_disable(_args())

    assert not service_path.exists()
    assert not timer_path.exists()
    assert other_timer.exists()
    assert ["systemctl", "--user", "disable", "--now", timer_path.name] in calls
    assert ["systemctl", "--user", "daemon-reload"] in calls
    status = _read_status(hermes_home)
    assert status["enabled"] is False
    assert status["schedule"] is None


def test_disable_systemd_keeps_files_and_status_when_stop_fails(tmp_path, monkeypatch, capsys):
    hermes_home, _calls = _set_linux_scheduler_env(tmp_path, monkeypatch)
    systemd_dir = tmp_path / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True)
    service_path, timer_path = update_auto._systemd_paths()
    service_path.write_text("service", encoding="utf-8")
    timer_path.write_text("timer", encoding="utf-8")
    update_auto.write_status(
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "03:00",
            "schedulerType": "systemd-user",
            "schedulerPath": str(timer_path),
            "schedulerIdentity": update_auto._scheduler_identity(),
        }
    )

    def failed_stop(args):
        return subprocess.CompletedProcess(
            ["systemctl", "--user"] + args,
            1,
            stdout="stop stdout",
            stderr="stop stderr",
        )

    monkeypatch.setattr(update_auto, "_systemctl_user", failed_stop)

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_disable(_args())

    assert exc.value.code == 1
    assert service_path.exists()
    assert timer_path.exists()
    assert _read_status(hermes_home)["enabled"] is True
    error = capsys.readouterr().err
    assert "stop stdout" in error
    assert "stop stderr" in error


def test_status_output_shows_enabled_schedule(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    update_auto.write_status(
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "03:00",
            "planSchedule": ["21:00"],
            "schedulerType": "launchd",
            "schedulerPath": "/tmp/com.hermes.agent.auto-update.plist",
            "status": update_auto.STATUS_SUCCESS,
        }
    )

    update_auto.cmd_auto_status(SimpleNamespace())

    out = capsys.readouterr().out
    assert "Enabled:          yes" in out
    assert "Schedule:         03:00" in out
    assert "Plan schedule:    21:00" in out
    assert "Scheduler:        launchd" in out
    assert "Scheduler path:   /tmp/com.hermes.agent.auto-update.plist" in out


def test_scheduled_dispatcher_uses_most_recent_configured_time():
    status = {"schedule": "04:00", "planSchedule": ["21:00"]}

    assert update_auto._scheduled_action_for_now(status, datetime(2026, 5, 27, 21, 5)) == "plan"
    assert update_auto._scheduled_action_for_now(status, datetime(2026, 5, 28, 3, 59)) == "plan"
    assert update_auto._scheduled_action_for_now(status, datetime(2026, 5, 28, 4, 5)) == "run"


def test_run_scheduled_is_noop_when_persisted_status_is_disabled(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    update_auto.write_status(
        {
            "enabled": False,
            "schedule": "04:00",
            "planSchedule": ["21:00"],
        }
    )
    monkeypatch.setattr(
        update_auto,
        "cmd_auto_plan",
        lambda _args: pytest.fail("disabled scheduled trigger must not plan"),
    )
    monkeypatch.setattr(
        update_auto,
        "cmd_auto_run_now",
        lambda _args: pytest.fail("disabled scheduled trigger must not run"),
    )

    assert update_auto.cmd_auto_run_scheduled(_args()) is None
    assert _read_status(hermes_home)["enabled"] is False


@pytest.mark.parametrize(
    "persisted",
    [
        {},
        {"enabled": "false", "schedule": "04:00"},
        {"enabled": True, "mode": "scheduled", "schedule": None},
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "04:00",
            "planSchedule": None,
            "schedulerType": "mystery",
            "schedulerPath": "/tmp/mystery",
        },
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "4:00",
            "planSchedule": [],
            "schedulerType": "launchd",
            "schedulerPath": "/tmp/launchd.plist",
        },
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "04:00",
            "planSchedule": [],
            "schedulerType": "launchd",
        },
    ],
)
def test_run_scheduled_fails_closed_for_malformed_persisted_state(
    tmp_path, monkeypatch, persisted
):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    status_path = hermes_home / "state" / "update-status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(json.dumps(persisted), encoding="utf-8")
    monkeypatch.setattr(update_auto, "_scheduled_action_for_now", lambda _status: "run")
    monkeypatch.setattr(
        update_auto,
        "cmd_auto_run_now",
        lambda _args: pytest.fail("malformed scheduler state must never run an update"),
    )
    monkeypatch.setattr(
        update_auto,
        "cmd_auto_plan",
        lambda _args: pytest.fail("malformed scheduler state must never plan an update"),
    )

    update_auto.cmd_auto_run_scheduled(_args())

    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_UPDATE_FAILED
    assert "scheduler state" in data["error"].lower()


@pytest.mark.parametrize(
    "before,after,expected_ok",
    [
        (
            {"gateway_state": "running", "pid": 101, "start_time": 1},
            {"gateway_state": "running", "pid": 101, "start_time": 1},
            False,
        ),
        (
            {"gateway_state": "running", "pid": 101, "start_time": 1},
            {"gateway_state": "running", "pid": 202, "start_time": 2},
            True,
        ),
        (
            {"gateway_state": "running", "pid": 101, "start_time": 1},
            {"gateway_state": "stopped", "pid": 101, "start_time": 1},
            False,
        ),
        (
            {"gateway_state": "stopped", "pid": 101, "start_time": 1},
            {"gateway_state": "stopped", "pid": 101, "start_time": 1},
            True,
        ),
    ],
)
def test_verify_health_requires_a_distinct_new_gateway_when_previously_running(
    monkeypatch, before, after, expected_ok
):
    from gateway import status as gateway_status

    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: after)
    monkeypatch.setattr(
        update_auto,
        "_live_gateway_identity",
        lambda runtime, **_kwargs: (
            {
                "pid": runtime["pid"],
                "start_time": runtime.get("start_time"),
                "command": "hermes gateway run",
            }
            if runtime.get("gateway_state") == "running"
            else None
        ),
    )
    monkeypatch.setattr(
        update_auto,
        "_stopped_runtime_is_live_validated",
        lambda runtime: runtime.get("gateway_state") == "stopped",
    )
    monkeypatch.setattr(update_auto, "_HEALTH_STARTUP_GRACE_SECONDS", 0.0)

    before = {**before, "_live_validated": True}
    ok, _detail = update_auto._verify_health(before)

    assert ok is expected_ok


def test_run_now_rejects_a_surviving_pre_update_gateway_process(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from gateway import status as gateway_status
    from hermes_cli import backup
    from hermes_cli import main as hm

    old_runtime = {"gateway_state": "running", "pid": 101, "start_time": 1}
    monkeypatch.setattr(
        gateway_status,
        "read_runtime_status",
        lambda: old_runtime,
    )
    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: {"update_available": True, "latest_version": "new"},
    )
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: hermes_home / "backups" / "pre-update.zip",
    )
    monkeypatch.setattr(hm, "cmd_update", lambda _args: None)
    versions = iter(["old", "new", "new"])
    monkeypatch.setattr(update_auto, "_current_version", lambda: next(versions))

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_HEALTH_FAILED
    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_HEALTH_FAILED
    assert "pre-update process" in data["error"]


def test_run_now_aborts_before_backup_when_running_gateway_has_no_reported_version(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from gateway import status as gateway_status
    from hermes_cli import backup
    from hermes_cli import main as hm

    runtime = {
        "gateway_state": "running",
        "pid": 101,
        "start_time": 1,
        "argv": ["hermes", "gateway", "run"],
    }
    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: runtime)
    monkeypatch.setattr(
        gateway_status,
        "get_runtime_status_running_pid",
        lambda *_args, **_kwargs: 101,
    )
    monkeypatch.setattr(
        gateway_status,
        "looks_like_gateway_runtime_command_line",
        lambda _command: True,
    )
    monkeypatch.setattr(
        update_auto,
        "_live_process_metadata",
        lambda _pid: {
            "pid": 101,
            "start_time": 1,
            "command": "hermes gateway run",
            "exe": update_auto.sys.executable,
        },
    )
    monkeypatch.setattr(update_auto, "_current_version", lambda: "old")
    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kwargs: {"update_available": True, "latest_version": "new"},
    )
    backup_calls = []
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: backup_calls.append(True) or hermes_home / "backups" / "pre-update.zip",
    )
    monkeypatch.setattr(hm, "cmd_update", lambda _args: pytest.fail("update must not run"))

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_HEALTH_FAILED
    assert backup_calls == []
    assert "version" in _read_status(hermes_home)["error"]


def test_verify_health_rejects_a_new_process_from_another_installation(monkeypatch):
    from gateway import status as gateway_status

    before = {
        "gateway_state": "running",
        "pid": 101,
        "start_time": 1,
        "runtime_version": "old",
        "installation_identity": "install-a",
        "_live_validated": True,
    }
    after = {
        "gateway_state": "running",
        "pid": 202,
        "start_time": 2,
        "runtime_version": "new",
        "installation_identity": "install-b",
        "argv": ["hermes", "gateway", "run"],
    }
    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: after)
    monkeypatch.setattr(
        update_auto,
        "_live_gateway_identity",
        lambda runtime, **_kwargs: {
            "pid": runtime["pid"],
            "start_time": runtime["start_time"],
            "command": "hermes gateway run",
            "runtime_version": runtime["runtime_version"],
            "installation_identity": runtime["installation_identity"],
        },
    )
    monkeypatch.setattr(update_auto, "_HEALTH_STARTUP_GRACE_SECONDS", 0.0)

    ok, detail = update_auto._verify_health(before, expected_version="new")

    assert ok is False
    assert "install" in detail.lower()


def test_busy_follower_does_not_overwrite_successful_leader_receipt(
    tmp_path, monkeypatch, capsys
):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    leader_receipt = {
        "generation": "leader-generation",
        "status": update_auto.STATUS_SUCCESS,
        "completedAt": "2026-07-20T00:00:00+00:00",
    }
    update_auto.write_status(
        {
            "status": update_auto.STATUS_SUCCESS,
            "runGeneration": "leader-generation",
            "terminalReceipt": leader_receipt,
        }
    )

    from hermes_cli.update_lock import UpdateLockBusyError

    monkeypatch.setattr(
        update_auto,
        "acquire_update_lock",
        lambda _root: (_ for _ in ()).throw(UpdateLockBusyError("busy")),
    )
    monkeypatch.setattr(update_auto, "_current_version", lambda: pytest.fail("follower must not write status"))

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_UPDATE_FAILED
    status = _read_status(hermes_home)
    assert status["status"] == update_auto.STATUS_SUCCESS
    assert status["terminalReceipt"] == leader_receipt
    assert "busy" in capsys.readouterr().err


def test_restore_systemd_returns_failure_receipt_when_a_rollback_substep_raises(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    service_path = tmp_path / "hermes.service"
    timer_path = tmp_path / "hermes.timer"
    monkeypatch.setattr(
        update_auto,
        "_systemctl_user",
        lambda _args: (_ for _ in ()).throw(OSError("systemctl unavailable")),
    )
    monkeypatch.setattr(
        update_auto,
        "_restore_file",
        lambda *_args: (_ for _ in ()).throw(OSError("restore write failed")),
    )

    receipt = update_auto._restore_systemd(
        service_path=service_path,
        timer_path=timer_path,
        timer_name=timer_path.name,
        prior_service=None,
        prior_timer=None,
        prior_state={
            "load_state": "not-found",
            "unit_file_state": "not-found",
            "active_state": "inactive",
        },
    )

    assert receipt["ok"] is False
    assert receipt["errors"]
    assert any("restore write failed" in error for error in receipt["errors"])
    assert _read_status(hermes_home)["status"] == update_auto.STATUS_RECOVERY_REQUIRED


def test_enable_systemd_refuses_loaded_timer_when_only_service_file_exists(
    tmp_path, monkeypatch
):
    _hermes_home, calls = _set_linux_scheduler_env(tmp_path, monkeypatch)
    service_path, timer_path = update_auto._systemd_paths()
    service_path.parent.mkdir(parents=True, exist_ok=True)
    service_path.write_text("service before\n", encoding="utf-8")

    def systemctl(args):
        calls.append(args)
        if args[0] == "show" and args[1] == timer_path.name:
            return subprocess.CompletedProcess(
                ["systemctl", "--user"] + args,
                0,
                stdout="LoadState=loaded\nUnitFileState=enabled\nActiveState=active\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            ["systemctl", "--user"] + args, 0, stdout="", stderr=""
        )

    monkeypatch.setattr(update_auto, "_systemctl_user", systemctl)

    with pytest.raises(RuntimeError, match="loaded"):
        update_auto._enable_systemd(3, 0, "03:00", [])

    assert service_path.read_text(encoding="utf-8") == "service before\n"
    assert not timer_path.exists()
    assert not any(call[0] == "disable" for call in calls)
    assert not any(call[0] == "daemon-reload" for call in calls)


def _parse_update_args(argv):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_update_parser(subparsers, cmd_update=lambda _args: None)
    return parser.parse_args(["update", *argv])


@pytest.mark.parametrize(
    "argv",
    [
        ["--branch", "release", "--force", "auto", "run-now"],
        ["auto", "run-now", "--branch", "release", "--force"],
    ],
)
def test_run_now_preserves_branch_and_force_options_on_either_side_of_subcommand(argv):
    args = _parse_update_args(argv)

    assert args.branch == "release"
    assert args.force is True


def test_run_now_child_branch_wins_when_parent_and_child_both_set_it():
    args = _parse_update_args(
        ["--branch", "before", "auto", "run-now", "--branch", "after"]
    )

    assert args.branch == "after"


@pytest.mark.parametrize("phase", ["backup", "update", "health"])
def test_run_now_converts_unexpected_phase_exceptions_to_terminal_failure(
    tmp_path, monkeypatch, capsys, phase
):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import backup
    from hermes_cli import main as hm

    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: {"update_available": True, "latest_version": "new"},
    )
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: hermes_home / "backups" / "pre-update.zip",
    )
    monkeypatch.setattr(
        hm,
        "cmd_update",
        lambda _args: None,
    )
    monkeypatch.setattr(
        update_auto,
        "_verify_health",
        lambda *_args: (True, "ok"),
    )
    if phase == "health":
        versions = iter(["old", "new", "new"])
        monkeypatch.setattr(update_auto, "_current_version", lambda: next(versions))
        monkeypatch.setattr(
            update_auto,
            "_verify_health",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("health probe crashed")),
        )
    else:
        monkeypatch.setattr(update_auto, "_current_version", lambda: "old")
    if phase == "update":
        monkeypatch.setattr(
            hm,
            "cmd_update",
            lambda _args: (_ for _ in ()).throw(RuntimeError("update process crashed")),
        )
    elif phase == "backup":
        monkeypatch.setattr(
            backup,
            "create_pre_update_backup",
            lambda: (_ for _ in ()).throw(OSError("backup disk full")),
        )

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_UPDATE_FAILED
    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_UPDATE_FAILED
    assert "running" not in data["status"]
    expected_error = {
        "backup": "backup disk full",
        "update": "update process crashed",
        "health": "health probe crashed",
    }[phase]
    assert expected_error in data["error"]
    assert "failed" in capsys.readouterr().err.lower()

    probe = UpdateLock(
        tmp_path / "source-checkout" / ".git" / "hermes-update.lock"
    ).acquire()
    probe.release()


def test_enable_rejects_plan_time_equal_to_update_time_without_scheduler_call(
    tmp_path, monkeypatch, capsys
):
    hermes_home, calls = _set_macos_scheduler_env(tmp_path, monkeypatch)

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_enable(_args(time="03:00", plan_time=["03:00"]))

    assert exc.value.code == 2
    assert calls == []
    assert not (hermes_home / "state" / "update-status.json").exists()
    assert "must differ" in capsys.readouterr().err


def test_plan_prints_concise_notice_and_records_status(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    update_auto.write_status({"enabled": True, "mode": "scheduled", "schedule": "04:00"})

    from hermes_cli import main as hm

    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: {
            "update_available": True,
            "current_version": "oldsha",
            "latest_version": "newsha",
            "behind": 1,
        },
    )

    update_auto.cmd_auto_plan(_args())

    out = capsys.readouterr().out
    assert "Hermes update available" in out
    assert "oldsha → newsha" in out
    assert "Scheduled auto-update: 04:00" in out
    status = _read_status(hermes_home)
    assert status["status"] == update_auto.STATUS_PLANNED
    assert status["lastPlanAt"] is not None
    assert status["plannedVersion"] == "newsha"


def test_auto_plan_holds_shared_lock_before_check_and_status_work(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from hermes_cli import main as hm

    events = []

    class FakeLock:
        held = True

        def release(self):
            events.append("release")
            self.held = False

    def acquire(_root):
        events.append("acquire")
        return FakeLock()

    monkeypatch.setattr(update_auto, "acquire_update_lock", acquire)
    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: events.append("check")
        or {
            "update_available": True,
            "current_version": "oldsha",
            "latest_version": "newsha",
        },
    )
    original_update_status_fields = update_auto.update_status_fields

    def record_status(**fields):
        events.append("status")
        return original_update_status_fields(**fields)

    monkeypatch.setattr(update_auto, "update_status_fields", record_status)

    update_auto.cmd_auto_plan(_args())

    assert events.index("acquire") < events.index("check") < events.index("status")
    assert events[-1] == "release"


def test_auto_plan_busy_does_not_check_or_overwrite_status(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    leader_receipt = {
        "generation": "leader-generation",
        "status": update_auto.STATUS_SUCCESS,
    }
    update_auto.write_status(
        {
            "status": update_auto.STATUS_SUCCESS,
            "runGeneration": "leader-generation",
            "terminalReceipt": leader_receipt,
        }
    )
    from hermes_cli import main as hm
    from hermes_cli.update_lock import UpdateLockBusyError

    monkeypatch.setattr(
        update_auto,
        "acquire_update_lock",
        lambda _root: (_ for _ in ()).throw(UpdateLockBusyError("busy")),
    )
    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: pytest.fail("busy plan must not check for updates"),
    )

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_plan(_args())

    assert exc.value.code == update_auto.EXIT_UPDATE_FAILED
    assert _read_status(hermes_home)["terminalReceipt"] == leader_receipt


def test_auto_status_holds_shared_lock_before_version_probe(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from hermes_cli import main as hm

    events = []

    class FakeLock:
        held = True

        def release(self):
            events.append("release")
            self.held = False

    monkeypatch.setattr(
        update_auto,
        "acquire_update_lock",
        lambda _root: events.append("acquire") or FakeLock(),
    )
    monkeypatch.setattr(
        update_auto,
        "_current_version",
        lambda: events.append("version") or "abc123",
    )
    monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path / "checkout")

    update_auto.cmd_auto_status(SimpleNamespace())

    assert events == ["acquire", "version", "release"]
    assert "Current version:  abc123" in capsys.readouterr().out


def test_auto_status_busy_uses_snapshot_without_subprocess_or_overwrite(
    tmp_path, monkeypatch, capsys
):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    update_auto.write_status({"status": update_auto.STATUS_SUCCESS, "currentVersion": "snapshot"})
    from hermes_cli.update_lock import UpdateLockBusyError

    monkeypatch.setattr(
        update_auto,
        "acquire_update_lock",
        lambda _root: (_ for _ in ()).throw(UpdateLockBusyError("busy")),
    )
    monkeypatch.setattr(
        update_auto,
        "_current_version",
        lambda: pytest.fail("busy status must not run a version subprocess"),
    )

    update_auto.cmd_auto_status(SimpleNamespace())

    out = capsys.readouterr().out
    assert "Current version:  snapshot" in out
    assert _read_status(hermes_home)["currentVersion"] == "snapshot"


def test_run_now_preserves_scheduler_configuration(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    update_auto.write_status(
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "03:00",
            "schedulerType": "launchd",
            "schedulerPath": "/tmp/hermes.plist",
        }
    )

    from hermes_cli import backup
    from hermes_cli import main as hm

    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: {"update_available": False, "latest_version": "same"},
    )
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: pytest.fail("backup should not run when already up to date"),
    )
    monkeypatch.setattr(update_auto, "_current_version", lambda: "same")

    update_auto.cmd_auto_run_now(_args())

    status = _read_status(hermes_home)
    assert status["enabled"] is True
    assert status["mode"] == "scheduled"
    assert status["schedule"] == "03:00"
    assert status["schedulerType"] == "launchd"
    assert status["schedulerPath"] == "/tmp/hermes.plist"
    assert status["status"] == update_auto.STATUS_UP_TO_DATE


def test_run_now_success_reuses_existing_update_flow_and_logs(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import backup
    from hermes_cli import main as hm

    update_calls = []
    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: {
            "update_available": True,
            "current_version": "oldsha",
            "latest_version": "newsha",
            "behind": 1,
        },
    )
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: hermes_home / "backups" / "pre-update.zip",
    )
    monkeypatch.setattr(hm, "cmd_update", lambda args: update_calls.append(args))
    versions = iter(["oldsha", "newsha"])
    monkeypatch.setattr(update_auto, "_current_version", lambda: next(versions))
    monkeypatch.setattr(update_auto, "_verify_health", lambda *_args: (True, "ok"))

    update_auto.cmd_auto_run_now(_args())

    assert len(update_calls) == 1
    called_args = update_calls[0]
    assert called_args.yes is True
    assert called_args.no_backup is True
    assert called_args.backup is False
    assert called_args._update_lock_held is True
    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_SUCCESS
    assert data["previousVersion"] == "oldsha"
    assert data["latestVersion"] == "newsha"
    assert data["currentVersion"] == "newsha"
    assert data["backupPath"].endswith("pre-update.zip")
    assert data["error"] is None
    assert data["runGeneration"]
    assert data["terminalReceipt"]["generation"] == data["runGeneration"]
    log_text = (hermes_home / "logs" / "update.log").read_text(encoding="utf-8")
    assert "event=start" in log_text
    assert "event=end result=success" in log_text
    assert "previous=oldsha" in log_text
    assert "latest=newsha" in log_text
    assert "current=newsha" in log_text
    assert "backup=" in log_text
    assert "Auto-update run complete" in capsys.readouterr().out


def test_run_now_fails_when_available_git_update_has_no_expected_sha(
    tmp_path, monkeypatch, capsys
):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import backup
    from hermes_cli import main as hm

    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: {
            "update_available": True,
            "install_method": "git",
            "latest_version": "newsha",
        },
    )
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: pytest.fail("missing expected SHA must abort before backup"),
    )
    monkeypatch.setattr(
        hm,
        "cmd_update",
        lambda _args: pytest.fail("missing expected SHA must abort before update"),
    )
    monkeypatch.setattr(update_auto, "_current_version", lambda: "oldsha")

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_UPDATE_FAILED
    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_UPDATE_FAILED
    assert "SHA" in data["error"]
    assert "update_failed" in capsys.readouterr().err


def test_run_now_fails_when_checked_latest_sha_is_not_in_current_head(
    tmp_path, monkeypatch, capsys
):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import backup
    from hermes_cli import main as hm

    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: {
            "update_available": True,
            "install_method": "git",
            "latest_version": "newsha",
            "latest_sha": "0123456789abcdef0123456789abcdef01234567",
        },
    )
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: hermes_home / "backups" / "pre-update.zip",
    )
    monkeypatch.setattr(hm, "cmd_update", lambda _args: None)
    monkeypatch.setattr(update_auto, "_verify_expected_sha", lambda _sha: (False, "not an ancestor"))
    monkeypatch.setattr(update_auto, "_verify_health", lambda *_args: (True, "ok"))
    monkeypatch.setattr(update_auto, "_current_version", lambda: "newsha")

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_UPDATE_FAILED
    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_UPDATE_FAILED
    assert "not an ancestor" in data["error"]
    assert "update_failed" in capsys.readouterr().err


def test_run_now_reports_lock_busy_without_checking_or_updating(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli.update_lock import UpdateLockBusyError
    from hermes_cli import main as hm

    monkeypatch.setattr(
        update_auto,
        "acquire_update_lock",
        lambda _root: (_ for _ in ()).throw(UpdateLockBusyError("busy")),
    )
    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: pytest.fail("busy update must not check for updates"),
    )
    monkeypatch.setattr(update_auto, "_current_version", lambda: "oldsha")

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_UPDATE_FAILED
    assert not (hermes_home / "state" / "update-status.json").exists()
    assert "already running" in capsys.readouterr().err


def test_run_now_up_to_date_skips_backup_and_update(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import backup
    from hermes_cli import main as hm

    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: {
            "update_available": False,
            "current_version": "same",
            "latest_version": "same",
            "behind": 0,
        },
    )
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: pytest.fail("backup should not run when already up to date"),
    )
    monkeypatch.setattr(
        hm,
        "cmd_update",
        lambda _args: pytest.fail("update should not run when already up to date"),
    )
    monkeypatch.setattr(update_auto, "_current_version", lambda: "same")

    update_auto.cmd_auto_run_now(_args())

    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_UP_TO_DATE
    assert data["backupPath"] is None
    assert data["error"] is None
    log_text = (hermes_home / "logs" / "update.log").read_text(encoding="utf-8")
    assert "event=start" in log_text
    assert "event=end result=up_to_date" in log_text
    assert "Already up to date" in capsys.readouterr().out


def test_run_now_records_check_failure(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import main as hm

    monkeypatch.setattr(hm, "_get_update_check_result", lambda **_kw: (_ for _ in ()).throw(RuntimeError("network down")))
    monkeypatch.setattr(update_auto, "_current_version", lambda: "old")

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_CHECK_FAILED
    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_CHECK_FAILED
    assert "network down" in data["error"]
    assert "check_failed" in capsys.readouterr().err


def test_run_now_aborts_when_backup_fails(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import backup
    from hermes_cli import main as hm

    monkeypatch.setattr(hm, "_get_update_check_result", lambda **_kw: {"update_available": True, "latest_version": "new"})
    monkeypatch.setattr(backup, "create_pre_update_backup", lambda: None)
    monkeypatch.setattr(hm, "cmd_update", lambda _args: pytest.fail("update must not run without backup"))
    monkeypatch.setattr(update_auto, "_current_version", lambda: "old")

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_BACKUP_FAILED
    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_BACKUP_FAILED
    assert "backup failed" in data["error"]
    assert data["backupPath"] is None
    assert "backup_failed" in capsys.readouterr().err


def test_run_now_records_update_failure(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import backup
    from hermes_cli import main as hm

    monkeypatch.setattr(hm, "_get_update_check_result", lambda **_kw: {"update_available": True, "latest_version": "new"})
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: hermes_home / "backups" / "pre-update.zip",
    )
    monkeypatch.setattr(hm, "cmd_update", lambda _args: (_ for _ in ()).throw(SystemExit(7)))
    monkeypatch.setattr(update_auto, "_current_version", lambda: "old")

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_UPDATE_FAILED
    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_UPDATE_FAILED
    assert "exit code 7" in data["error"]
    assert data["backupPath"].endswith("pre-update.zip")
    assert "update_failed" in capsys.readouterr().err


def test_run_now_records_health_failure_after_update(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import backup
    from hermes_cli import main as hm

    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: {
            "update_available": True,
            "latest_version": "new",
            "install_method": "git",
            "latest_sha": "newsha",
        },
    )
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: hermes_home / "backups" / "pre-update.zip",
    )
    monkeypatch.setattr(hm, "cmd_update", lambda _args: None)
    monkeypatch.setattr(update_auto, "_verify_expected_sha", lambda _sha: (True, "ok"))
    monkeypatch.setattr(update_auto, "_current_version", lambda: "old")
    monkeypatch.setattr(
        update_auto,
        "_verify_health",
        lambda *_args, **_kwargs: (False, "gateway startup failed"),
    )

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_HEALTH_FAILED
    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_HEALTH_FAILED
    assert "health check failed" in data["error"]
    assert "gateway startup failed" in data["error"]
    assert "health_failed" in capsys.readouterr().err


def test_run_now_does_not_report_success_when_managed_update_returns_normally(
    tmp_path, monkeypatch
):
    """A managed install's normal-return guard must not become auto-update success."""
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_MANAGED", "homebrew")

    from hermes_cli import backup
    from hermes_cli import main as hm

    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: {
            "update_available": True,
            "current_version": "oldsha",
            "latest_version": "newsha",
            "behind": 1,
        },
    )
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: hermes_home / "backups" / "pre-update.zip",
    )
    monkeypatch.setattr(update_auto, "_current_version", lambda: "oldsha")
    monkeypatch.setattr(update_auto, "_verify_health", lambda *_args: (True, "ok"))

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_UPDATE_FAILED
    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_UPDATE_FAILED
    assert "managed" in data["error"].lower()


def test_run_now_does_not_report_success_when_update_returns_without_applying(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import backup
    from hermes_cli import main as hm

    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kw: {
            "update_available": True,
            "current_version": "oldsha",
            "latest_version": "newsha",
            "behind": 1,
        },
    )
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: hermes_home / "backups" / "pre-update.zip",
    )
    monkeypatch.setattr(hm, "cmd_update", lambda _args: None)
    monkeypatch.setattr(update_auto, "_current_version", lambda: "oldsha")
    monkeypatch.setattr(update_auto, "_verify_health", lambda *_args: (True, "ok"))

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_UPDATE_FAILED
    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_UPDATE_FAILED
    assert "without applying" in data["error"]


def test_verify_health_treats_operator_stopped_gateway_as_informational(monkeypatch):
    from gateway import status as gateway_status

    monkeypatch.setattr(
        gateway_status,
        "read_runtime_status",
        lambda: {"gateway_state": "stopped"},
    )
    monkeypatch.setattr(gateway_status, "get_running_pid", lambda: None)

    ok, detail = update_auto._verify_health(
        {"gateway_state": "stopped", "_live_validated": True}
    )

    assert ok is True
    assert "already stopped" in detail


def test_verify_health_rejects_unvalidated_stopped_pre_state(monkeypatch):
    from gateway import status as gateway_status

    monkeypatch.setattr(
        gateway_status,
        "read_runtime_status",
        lambda: {"gateway_state": "stopped"},
    )

    ok, detail = update_auto._verify_health({"gateway_state": "stopped"})

    assert ok is False
    assert "live-validated" in detail


def test_capture_gateway_runtime_rejects_a_dead_recorded_pid(monkeypatch):
    from gateway import status as gateway_status

    runtime = {
        "gateway_state": "running",
        "pid": 4242,
        "start_time": 10,
        "argv": ["hermes", "gateway", "run"],
    }
    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: runtime)
    monkeypatch.setattr(
        gateway_status,
        "get_runtime_status_running_pid",
        lambda *_args, **_kwargs: None,
    )

    assert update_auto._capture_gateway_runtime() is None


def test_verify_health_rejects_an_unrelated_live_runtime(monkeypatch):
    from gateway import status as gateway_status

    after = {
        "gateway_state": "running",
        "pid": 202,
        "start_time": 2,
        "argv": ["unrelated", "process"],
    }
    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: after)
    monkeypatch.setattr(
        gateway_status,
        "get_runtime_status_running_pid",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(update_auto, "_HEALTH_STARTUP_GRACE_SECONDS", 0.0)

    ok, detail = update_auto._verify_health(
        {
            "gateway_state": "running",
            "pid": 101,
            "start_time": 1,
            "_live_validated": True,
        }
    )

    assert ok is False
    assert "live" in detail.lower() or "identity" in detail.lower()


def test_verify_health_polls_through_delayed_gateway_startup(monkeypatch):
    from gateway import status as gateway_status

    statuses = iter(
        [
            {"gateway_state": "starting", "pid": 202, "start_time": 2},
            {
                "gateway_state": "running",
                "pid": 202,
                "start_time": 2,
                "argv": ["hermes", "gateway", "run"],
            },
        ]
    )
    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: next(statuses))
    monkeypatch.setattr(
        gateway_status,
        "get_runtime_status_running_pid",
        lambda runtime=None, **_kwargs: 202
        if runtime and runtime.get("gateway_state") == "running"
        else None,
    )
    monkeypatch.setattr(
        update_auto,
        "_live_gateway_identity",
        lambda runtime, **_kwargs: {"pid": 202, "start_time": 2, "command": "hermes gateway run"}
        if runtime.get("gateway_state") == "running"
        else None,
    )
    monkeypatch.setattr(update_auto, "time", time_module, raising=False)
    clock = [0.0]
    monkeypatch.setattr(update_auto.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        update_auto.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    ok, detail = update_auto._verify_health(
        {
            "gateway_state": "running",
            "pid": 101,
            "start_time": 1,
            "_live_validated": True,
        }
    )

    assert ok is True
    assert "running" in detail


def test_verify_health_polls_while_old_gateway_identity_remains_then_accepts_new_pid(
    monkeypatch,
):
    from gateway import status as gateway_status

    statuses = iter(
        [
            {"gateway_state": "running", "pid": 101, "start_time": 1},
            {"gateway_state": "running", "pid": 101, "start_time": 1},
            {"gateway_state": "running", "pid": 202, "start_time": 2},
        ]
    )
    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: next(statuses))
    monkeypatch.setattr(
        update_auto,
        "_live_gateway_identity",
        lambda runtime, **_kwargs: {
            "pid": runtime["pid"],
            "start_time": runtime["start_time"],
            "command": "hermes gateway run",
        },
    )
    monkeypatch.setattr(update_auto, "time", time_module, raising=False)
    clock = [0.0]
    monkeypatch.setattr(update_auto.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        update_auto.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    ok, detail = update_auto._verify_health(
        {
            "gateway_state": "running",
            "pid": 101,
            "start_time": 1,
            "_live_validated": True,
        }
    )

    assert ok is True
    assert "202" in detail


def test_verify_health_accepts_same_pid_with_a_new_start_time(monkeypatch):
    from gateway import status as gateway_status

    after = {
        "gateway_state": "running",
        "pid": 101,
        "start_time": 2,
        "argv": ["hermes", "gateway", "run"],
    }
    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: after)
    monkeypatch.setattr(
        gateway_status,
        "get_runtime_status_running_pid",
        lambda *_args, **_kwargs: 101,
    )
    monkeypatch.setattr(update_auto, "_HEALTH_STARTUP_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(
        update_auto,
        "_live_process_metadata",
        lambda _pid: {
            "pid": 101,
            "start_time": 2,
            "command": "hermes gateway run",
        },
    )

    ok, detail = update_auto._verify_health(
        {
            "gateway_state": "running",
            "pid": 101,
            "start_time": 1,
            "_live_validated": True,
        }
    )

    assert ok is True
    assert "101" in detail


@pytest.mark.parametrize(
    "state",
    ["starting", "stopping", "unknown", "degraded", None],
)
def test_capture_gateway_runtime_does_not_assume_nonterminal_state_is_running(
    monkeypatch, state
):
    from gateway import status as gateway_status

    runtime = {
        "gateway_state": state,
        "pid": 4242,
        "start_time": 10,
    }
    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: runtime)
    monkeypatch.setattr(
        update_auto,
        "_live_gateway_identity",
        lambda *_args, **_kwargs: pytest.fail(
            "ambiguous gateway state must not be treated as running"
        ),
    )

    assert update_auto._capture_gateway_runtime() is None


def test_run_now_aborts_before_update_for_ambiguous_gateway_state(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from gateway import status as gateway_status
    from hermes_cli import main as hm

    monkeypatch.setattr(
        gateway_status,
        "read_runtime_status",
        lambda: {"gateway_state": "starting", "pid": 4242, "start_time": 10},
    )
    monkeypatch.setattr(
        hm,
        "_get_update_check_result",
        lambda **_kwargs: {"update_available": True, "latest_version": "new"},
    )
    monkeypatch.setattr(
        hm,
        "cmd_update",
        lambda _args: pytest.fail("ambiguous gateway state must abort before update"),
    )
    monkeypatch.setattr(
        "hermes_cli.backup.create_pre_update_backup",
        lambda: pytest.fail("ambiguous gateway state must abort before backup"),
    )
    monkeypatch.setattr(update_auto, "_current_version", lambda: "old")

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_HEALTH_FAILED
    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_HEALTH_FAILED
    assert "starting" in data["error"]


def test_capture_gateway_runtime_marks_stopped_only_after_live_validation(monkeypatch):
    from gateway import status as gateway_status

    runtime = {
        "gateway_state": "stopped",
        "pid": 4242,
        "start_time": 10,
    }
    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: runtime)
    monkeypatch.setattr(gateway_status, "get_running_pid", lambda: None)

    captured = update_auto._capture_gateway_runtime()

    assert captured is not None
    assert captured["gateway_state"] == "stopped"
    assert captured["_live_validated"] is True


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_scheduler_status_write_failure_rolls_back_activation(tmp_path, monkeypatch, platform):
    if platform == "darwin":
        _hermes_home, calls = _set_macos_scheduler_env(tmp_path, monkeypatch)
    else:
        _hermes_home, calls = _set_linux_scheduler_env(tmp_path, monkeypatch)

    monkeypatch.setattr(
        update_auto,
        "update_status_fields",
        lambda **_fields: (_ for _ in ()).throw(OSError("status disk full")),
    )

    with pytest.raises((SystemExit, OSError)):
        update_auto.cmd_auto_enable(_args(time="03:00"))

    if platform == "darwin":
        assert not update_auto._launchd_plist_path().exists()
        assert any(call[0] == "bootout" for call in calls)
    else:
        service_path, timer_path = update_auto._systemd_paths()
        assert not service_path.exists()
        assert not timer_path.exists()
        assert ["systemctl", "--user", "disable", "--now", timer_path.name] in calls


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_scheduler_disable_status_write_failure_restores_files_and_manager_state(
    tmp_path, monkeypatch, platform
):
    if platform == "darwin":
        hermes_home, calls = _set_macos_scheduler_env(tmp_path, monkeypatch)
        scheduler_path = update_auto._launchd_plist_path()
        scheduler_path.parent.mkdir(parents=True)
        scheduler_path.write_bytes(b"prior launchd bytes\n")

        def launchctl(args):
            calls.append(args)
            if args[0] == "print":
                return subprocess.CompletedProcess(
                    ["launchctl"] + args,
                    0,
                    stdout="state = running\n",
                    stderr="",
                )
            if args[0] == "print-disabled":
                return subprocess.CompletedProcess(
                    ["launchctl"] + args,
                    0,
                    stdout='"com.hermes.agent.auto-update" => false\n',
                    stderr="",
                )
            return subprocess.CompletedProcess(
                ["launchctl"] + args, 0, stdout="", stderr=""
            )

        monkeypatch.setattr(update_auto, "_run_launchctl", launchctl)
        prior_bytes = scheduler_path.read_bytes()
    else:
        hermes_home, calls = _set_linux_scheduler_env(tmp_path, monkeypatch)
        systemd_dir = tmp_path / ".config" / "systemd" / "user"
        systemd_dir.mkdir(parents=True)
        service_path, scheduler_path = update_auto._systemd_paths()
        service_path.write_bytes(b"prior service bytes\n")
        scheduler_path.write_bytes(b"prior timer bytes\n")
        prior_bytes = scheduler_path.read_bytes()
        show_count = 0

        def systemctl(args):
            nonlocal show_count
            calls.append(args)
            if args[0] == "show":
                show_count += 1
                output = (
                    "LoadState=loaded\nUnitFileState=enabled\nActiveState=active\n"
                    if show_count == 1
                    else "LoadState=not-found\nUnitFileState=not-found\nActiveState=inactive\n"
                )
                return subprocess.CompletedProcess(
                    ["systemctl", "--user"] + args,
                    0,
                    stdout=output,
                    stderr="",
                )
            return subprocess.CompletedProcess(
                ["systemctl", "--user"] + args, 0, stdout="", stderr=""
            )

        monkeypatch.setattr(update_auto, "_systemctl_user", systemctl)

    update_auto.write_status(
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "03:00",
            "schedulerType": "launchd" if platform == "darwin" else "systemd-user",
            "schedulerPath": str(scheduler_path),
            "schedulerIdentity": update_auto._scheduler_identity(),
        }
    )
    monkeypatch.setattr(
        update_auto,
        "update_status_fields",
        lambda **_fields: (_ for _ in ()).throw(OSError("status disk full")),
    )

    with pytest.raises(SystemExit):
        update_auto.cmd_auto_disable(_args())

    assert scheduler_path.read_bytes() == prior_bytes
    assert _read_status(hermes_home)["enabled"] is True
    if platform == "darwin":
        assert any(call[0] == "bootstrap" for call in calls)
        assert any(call[0] == "kickstart" for call in calls)
    else:
        assert ["enable", scheduler_path.name] in calls
        assert ["start", scheduler_path.name] in calls


def test_disable_systemd_orders_delete_before_reload_and_verifies_not_found_inactive(
    tmp_path, monkeypatch
):
    hermes_home, calls = _set_linux_scheduler_env(tmp_path, monkeypatch)
    systemd_dir = tmp_path / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True)
    service_path, timer_path = update_auto._systemd_paths()
    service_path.write_bytes(b"service\n")
    timer_path.write_bytes(b"timer\n")
    update_auto.write_status(
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "03:00",
            "schedulerType": "systemd-user",
            "schedulerPath": str(timer_path),
            "schedulerIdentity": update_auto._scheduler_identity(),
        }
    )
    show_results = iter(
        [
            "LoadState=loaded\nUnitFileState=enabled\nActiveState=active\n",
            "LoadState=loaded\nUnitFileState=enabled\nActiveState=active\n",
            "LoadState=not-found\nUnitFileState=not-found\nActiveState=inactive\n",
            "LoadState=not-found\nUnitFileState=not-found\nActiveState=inactive\n",
        ]
    )

    def systemctl(args):
        calls.append(args)
        if args[0] == "show":
            return subprocess.CompletedProcess(
                ["systemctl", "--user"] + args,
                0,
                stdout=next(show_results),
                stderr="",
            )
        if args == ["disable", "--now", timer_path.name]:
            assert service_path.exists() and timer_path.exists()
        if args == ["daemon-reload"]:
            assert not service_path.exists() and not timer_path.exists()
        return subprocess.CompletedProcess(
            ["systemctl", "--user"] + args, 0, stdout="", stderr=""
        )

    monkeypatch.setattr(update_auto, "_systemctl_user", systemctl)

    update_auto.cmd_auto_disable(_args())

    assert not service_path.exists()
    assert not timer_path.exists()
    assert _read_status(hermes_home)["enabled"] is False


@pytest.mark.parametrize("failure", ["reload", "verification"])
def test_disable_systemd_failure_restores_exact_files_and_modes(
    tmp_path, monkeypatch, failure
):
    _hermes_home, calls = _set_linux_scheduler_env(tmp_path, monkeypatch)
    systemd_dir = tmp_path / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True)
    service_path, timer_path = update_auto._systemd_paths()
    service_path.write_bytes(b"service before\n")
    timer_path.write_bytes(b"timer before\n")
    service_path.chmod(0o640)
    timer_path.chmod(0o600)
    prior = {
        service_path: (service_path.read_bytes(), service_path.stat().st_mode & 0o7777),
        timer_path: (timer_path.read_bytes(), timer_path.stat().st_mode & 0o7777),
    }
    update_auto.write_status(
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "03:00",
            "schedulerType": "systemd-user",
            "schedulerPath": str(timer_path),
            "schedulerIdentity": update_auto._scheduler_identity(),
        }
    )
    show_count = 0
    reload_count = 0

    def systemctl(args):
        nonlocal show_count, reload_count
        calls.append(args)
        if args[0] == "show":
            show_count += 1
            output = (
                "LoadState=loaded\nUnitFileState=enabled\nActiveState=active\n"
                if show_count == 1 or failure == "verification"
                else "LoadState=not-found\nUnitFileState=not-found\nActiveState=inactive\n"
            )
            return subprocess.CompletedProcess(
                ["systemctl", "--user"] + args, 0, stdout=output, stderr=""
            )
        if args == ["daemon-reload"]:
            reload_count += 1
            if failure == "reload" and reload_count == 1:
                return subprocess.CompletedProcess(
                    ["systemctl", "--user"] + args,
                    1,
                    stdout="reload failed",
                    stderr="reload error",
                )
        return subprocess.CompletedProcess(
            ["systemctl", "--user"] + args, 0, stdout="", stderr=""
        )

    monkeypatch.setattr(update_auto, "_systemctl_user", systemctl)

    with pytest.raises(SystemExit):
        update_auto.cmd_auto_disable(_args())

    for path, (content, mode) in prior.items():
        assert path.read_bytes() == content
        assert path.stat().st_mode & 0o7777 == mode


def test_disable_systemd_refuses_fileless_loaded_unit_without_mutation(
    tmp_path, monkeypatch
):
    _hermes_home, calls = _set_linux_scheduler_env(tmp_path, monkeypatch)
    _service_path, timer_path = update_auto._systemd_paths()
    update_auto.write_status(
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "03:00",
            "schedulerType": "systemd-user",
            "schedulerPath": str(timer_path),
            "schedulerIdentity": update_auto._scheduler_identity(),
        }
    )

    def systemctl(args):
        calls.append(args)
        if args[0] == "show":
            return subprocess.CompletedProcess(
                ["systemctl", "--user"] + args,
                0,
                stdout="LoadState=loaded\nUnitFileState=enabled\nActiveState=active\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            ["systemctl", "--user"] + args, 0, stdout="", stderr=""
        )

    monkeypatch.setattr(update_auto, "_systemctl_user", systemctl)

    with pytest.raises(SystemExit):
        update_auto.cmd_auto_disable(_args())

    assert not any(call[0] == "disable" for call in calls)
    assert not any(call[0] == "daemon-reload" for call in calls)


def test_disable_systemd_deletion_failure_restores_files_and_prior_active_state(
    tmp_path, monkeypatch
):
    _hermes_home, calls = _set_linux_scheduler_env(tmp_path, monkeypatch)
    systemd_dir = tmp_path / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True)
    service_path, timer_path = update_auto._systemd_paths()
    service_path.write_bytes(b"service before delete\n")
    timer_path.write_bytes(b"timer before delete\n")
    service_mode = service_path.stat().st_mode & 0o7777
    timer_mode = timer_path.stat().st_mode & 0o7777
    update_auto.write_status(
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "03:00",
            "schedulerType": "systemd-user",
            "schedulerPath": str(timer_path),
            "schedulerIdentity": update_auto._scheduler_identity(),
        }
    )

    def systemctl(args):
        calls.append(args)
        if args[0] == "show":
            return subprocess.CompletedProcess(
                ["systemctl", "--user"] + args,
                0,
                stdout="LoadState=loaded\nUnitFileState=enabled\nActiveState=active\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            ["systemctl", "--user"] + args, 0, stdout="", stderr=""
        )

    monkeypatch.setattr(update_auto, "_systemctl_user", systemctl)
    original_unlink = Path.unlink

    def fail_timer_delete(path, *args, **kwargs):
        if path == timer_path:
            raise OSError("timer delete failed")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_timer_delete)

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_disable(_args())

    assert exc.value.code == 1
    assert service_path.read_bytes() == b"service before delete\n"
    assert timer_path.read_bytes() == b"timer before delete\n"
    assert service_path.stat().st_mode & 0o7777 == service_mode
    assert timer_path.stat().st_mode & 0o7777 == timer_mode
    assert ["enable", timer_path.name] in calls
    assert ["start", timer_path.name] in calls


def test_scheduler_identity_is_scoped_to_installation_and_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(update_auto.sys, "platform", "darwin")
    monkeypatch.setattr(update_auto.Path, "home", staticmethod(lambda: tmp_path / "os-home"))
    monkeypatch.setattr(update_auto, "_hermes_command_prefix", lambda: ["hermes"])
    monkeypatch.setattr(
        update_auto,
        "_run_launchctl",
        lambda args: subprocess.CompletedProcess(
            ["launchctl"] + args,
            1 if args[0] == "print" else 0,
            stdout="",
            stderr="Could not find service" if args[0] == "print" else "",
        ),
    )
    monkeypatch.setattr(
        update_auto,
        "_launchd_state",
        lambda _target: {
            "loaded": update_auto._launchd_plist_path().is_file(),
            "enabled": True,
            "running": False,
        },
    )

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    update_auto.cmd_auto_enable(_args(time="03:00"))
    status_a = _read_status(home_a)
    path_a = Path(status_a["schedulerPath"])

    monkeypatch.setenv("HERMES_HOME", str(home_b))
    update_auto.cmd_auto_enable(_args(time="04:00"))
    status_b = _read_status(home_b)
    path_b = Path(status_b["schedulerPath"])

    assert path_a != path_b
    assert status_a["schedulerIdentity"] != status_b["schedulerIdentity"]
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    update_auto.cmd_auto_disable(_args())
    assert not path_a.exists()
    assert path_b.exists()
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    assert update_auto._validate_scheduler_status(status_a) is not None


def test_launchd_enable_aborts_before_unloading_fileless_loaded_job(tmp_path, monkeypatch):
    _hermes_home, calls = _set_macos_scheduler_env(tmp_path, monkeypatch)

    def launchctl(args):
        calls.append(args)
        if args[0] == "print":
            return subprocess.CompletedProcess(
                ["launchctl"] + args,
                0,
                stdout="state = running\n",
                stderr="",
            )
        return subprocess.CompletedProcess(["launchctl"] + args, 0, stdout="", stderr="")

    monkeypatch.setattr(update_auto, "_run_launchctl", launchctl)

    with pytest.raises(SystemExit):
        update_auto.cmd_auto_enable(_args(time="03:00"))

    assert not any(call[0] == "bootout" for call in calls)
    assert not update_auto._launchd_plist_path().exists()


def test_launchd_scheduler_symlink_is_rejected_without_touching_target(tmp_path, monkeypatch):
    _hermes_home, _calls = _set_macos_scheduler_env(tmp_path, monkeypatch)
    external = tmp_path / "external.plist"
    external.write_bytes(b"external scheduler\n")
    plist_path = update_auto._launchd_plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.symlink_to(external)

    with pytest.raises(SystemExit):
        update_auto.cmd_auto_enable(_args(time="03:00"))

    assert plist_path.is_symlink()
    assert external.read_bytes() == b"external scheduler\n"


@pytest.mark.parametrize(
    "unit_file_state,expected",
    [
        ("enabled-runtime", ["enable", "--runtime"]),
        ("linked", ["link"]),
        ("linked-runtime", ["link", "--runtime"]),
    ],
)
def test_systemd_rollback_restores_raw_unit_file_state(
    tmp_path, monkeypatch, unit_file_state, expected
):
    service_path = tmp_path / "hermes.service"
    timer_path = tmp_path / "hermes.timer"
    service_path.write_bytes(b"prior service\n")
    timer_path.write_bytes(b"prior timer\n")
    calls = []

    def systemctl(args):
        calls.append(args)
        return subprocess.CompletedProcess(
            ["systemctl", "--user"] + args, 0, stdout="", stderr=""
        )

    monkeypatch.setattr(update_auto, "_systemctl_user", systemctl)
    update_auto._restore_systemd(
        service_path=service_path,
        timer_path=timer_path,
        timer_name=timer_path.name,
        prior_service=(b"prior service\n", 0o100644),
        prior_timer=(b"prior timer\n", 0o100644),
        prior_state={
            "load_state": "loaded",
            "unit_file_state": unit_file_state,
            "active_state": "active",
        },
    )

    expected_call = (
        expected + [timer_path.name]
        if expected[0] == "enable"
        else expected + [str(timer_path)]
    )
    assert expected_call in calls
    assert ["start", timer_path.name] in calls
    assert timer_path.read_bytes() == b"prior timer\n"


def test_systemd_rollback_restores_service_and_timer_manager_state_in_order(
    tmp_path, monkeypatch
):
    service_path = tmp_path / "hermes.service"
    timer_path = tmp_path / "hermes.timer"
    service_path.write_bytes(b"prior service\n")
    timer_path.write_bytes(b"prior timer\n")
    calls = []
    states = {
        service_path.name: {
            "load": "loaded",
            "unit": "enabled",
            "active": "active",
        },
        timer_path.name: {
            "load": "loaded",
            "unit": "enabled",
            "active": "active",
        },
    }

    def systemctl(args):
        calls.append(args)
        if args[0] == "show":
            state = states[args[1]]
            return subprocess.CompletedProcess(
                ["systemctl", "--user"] + args,
                0,
                stdout=(
                    f"LoadState={state['load']}\n"
                    f"UnitFileState={state['unit']}\n"
                    f"ActiveState={state['active']}\n"
                ),
                stderr="",
            )
        unit = args[-1]
        if args[0] == "stop":
            states[unit]["active"] = "inactive"
        elif args[0] == "disable":
            states[unit]["unit"] = "disabled"
            states[unit]["active"] = "inactive"
        elif args[0] == "enable":
            states[unit]["unit"] = "enabled"
        elif args[0] == "start":
            states[unit]["active"] = "active"
        return subprocess.CompletedProcess(
            ["systemctl", "--user"] + args, 0, stdout="", stderr=""
        )

    monkeypatch.setattr(update_auto, "_systemctl_user", systemctl)
    receipt = update_auto._restore_systemd(
        service_path=service_path,
        timer_path=timer_path,
        timer_name=timer_path.name,
        prior_service=(b"prior service\n", 0o100644),
        prior_timer=(b"prior timer\n", 0o100644),
        prior_state={
            "service_state": {
                "load_state": "loaded",
                "unit_file_state": "enabled",
                "active_state": "active",
            },
            "timer_state": {
                "load_state": "loaded",
                "unit_file_state": "enabled",
                "active_state": "active",
            },
        },
    )

    assert receipt["ok"] is True
    service_stop = calls.index(["stop", service_path.name])
    service_disable = calls.index(["disable", service_path.name])
    service_enable = calls.index(["enable", service_path.name])
    service_start = calls.index(["start", service_path.name])
    assert service_stop < service_disable < service_enable < service_start
    assert ["start", timer_path.name] in calls
    assert receipt["manager"]["actual"]["service"]["active_state"] == "active"
    assert receipt["manager"]["actual"]["timer"]["active_state"] == "active"
