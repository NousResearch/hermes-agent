from __future__ import annotations

import argparse
import json
import plistlib
import subprocess
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

    def fake_launchctl(args):
        calls.append(args)
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

    plist_path = tmp_path / "Library" / "LaunchAgents" / f"{update_auto.LAUNCHD_LABEL}.plist"
    assert plist_path.exists()
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)
    assert plist["Label"] == update_auto.LAUNCHD_LABEL
    assert plist["ProgramArguments"] == ["hermes", "update", "auto", "run-scheduled"]
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


def test_enable_launchd_requires_bootstrap_success_and_reports_both_streams(
    tmp_path, monkeypatch, capsys
):
    _hermes_home, calls = _set_macos_scheduler_env(tmp_path, monkeypatch)

    def failed_bootstrap(args):
        calls.append(args)
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
    plist_path = tmp_path / "Library" / "LaunchAgents" / f"{update_auto.LAUNCHD_LABEL}.plist"
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
    service_path = systemd_dir / "hermes-auto-update.service"
    timer_path = systemd_dir / "hermes-auto-update.timer"
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
        ["daemon-reload"],
        ["disable", "--now", "hermes-auto-update.timer"],
        ["daemon-reload"],
    ]
    assert update_auto.read_status()["enabled"] is False
    error = capsys.readouterr().err
    assert "reload stdout" in error
    assert "reload stderr" in error


def test_enable_systemd_requires_enable_now_success(tmp_path, monkeypatch, capsys):
    hermes_home, _calls = _set_linux_scheduler_env(tmp_path, monkeypatch)
    calls = []

    def failed_enable(args):
        calls.append(args)
        if args == ["enable", "--now", "hermes-auto-update.timer"]:
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
        ["daemon-reload"],
        ["enable", "--now", "hermes-auto-update.timer"],
        ["disable", "--now", "hermes-auto-update.timer"],
        ["daemon-reload"],
    ]
    assert update_auto.read_status()["enabled"] is False
    error = capsys.readouterr().err
    assert "enable stdout" in error
    assert "enable stderr" in error


def test_enable_with_plan_time_creates_single_launchd_scheduler_with_two_triggers(tmp_path, monkeypatch):
    hermes_home, _calls = _set_macos_scheduler_env(tmp_path, monkeypatch)

    update_auto.cmd_auto_enable(_args(time="04:00", plan_time=["21:00"]))

    plist_path = tmp_path / "Library" / "LaunchAgents" / f"{update_auto.LAUNCHD_LABEL}.plist"
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)
    assert plist["ProgramArguments"] == ["hermes", "update", "auto", "run-scheduled"]
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

    plist_path = tmp_path / "Library" / "LaunchAgents" / f"{update_auto.LAUNCHD_LABEL}.plist"
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
    hermes_plist = launch_agents / f"{update_auto.LAUNCHD_LABEL}.plist"
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
    plist_path = launch_agents / f"{update_auto.LAUNCHD_LABEL}.plist"
    plist_path.write_text("hermes", encoding="utf-8")
    update_auto.write_status(
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "03:00",
            "schedulerType": "launchd",
            "schedulerPath": str(plist_path),
        }
    )

    def failed_bootout(args):
        calls.append(args)
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
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(update_auto.subprocess, "run", fake_run)
    return hermes_home, calls


def test_enable_creates_expected_systemd_user_timer_on_linux(tmp_path, monkeypatch):
    hermes_home, calls = _set_linux_scheduler_env(tmp_path, monkeypatch)

    update_auto.cmd_auto_enable(_args(time="03:00"))

    systemd_dir = tmp_path / ".config" / "systemd" / "user"
    service_path = systemd_dir / "hermes-auto-update.service"
    timer_path = systemd_dir / "hermes-auto-update.timer"
    service_text = service_path.read_text(encoding="utf-8")
    timer_text = timer_path.read_text(encoding="utf-8")
    assert "ExecStart=hermes update auto run-scheduled" in service_text
    assert f"Environment=HERMES_HOME={hermes_home}" in service_text
    assert f"StandardOutput=append:{hermes_home / 'logs' / 'update-auto.out.log'}" in service_text
    assert "OnCalendar=*-*-* 03:00:00" in timer_text
    assert "Persistent=true" in timer_text
    assert ["systemctl", "--version"] in calls
    assert ["systemctl", "--user", "daemon-reload"] in calls
    assert ["systemctl", "--user", "enable", "--now", "hermes-auto-update.timer"] in calls

    status = _read_status(hermes_home)
    assert status["enabled"] is True
    assert status["schedule"] == "03:00"
    assert status["schedulerType"] == "systemd-user"
    assert status["schedulerPath"] == str(timer_path)


def test_disable_removes_only_hermes_systemd_user_files(tmp_path, monkeypatch):
    hermes_home, calls = _set_linux_scheduler_env(tmp_path, monkeypatch)
    systemd_dir = tmp_path / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True)
    service_path = systemd_dir / "hermes-auto-update.service"
    timer_path = systemd_dir / "hermes-auto-update.timer"
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
        }
    )

    update_auto.cmd_auto_disable(_args())

    assert not service_path.exists()
    assert not timer_path.exists()
    assert other_timer.exists()
    assert ["systemctl", "--user", "disable", "--now", "hermes-auto-update.timer"] in calls
    assert ["systemctl", "--user", "daemon-reload"] in calls
    status = _read_status(hermes_home)
    assert status["enabled"] is False
    assert status["schedule"] is None


def test_disable_systemd_keeps_files_and_status_when_stop_fails(tmp_path, monkeypatch, capsys):
    hermes_home, _calls = _set_linux_scheduler_env(tmp_path, monkeypatch)
    systemd_dir = tmp_path / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True)
    service_path = systemd_dir / "hermes-auto-update.service"
    timer_path = systemd_dir / "hermes-auto-update.timer"
    service_path.write_text("service", encoding="utf-8")
    timer_path.write_text("timer", encoding="utf-8")
    update_auto.write_status(
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "03:00",
            "schedulerType": "systemd-user",
            "schedulerPath": str(timer_path),
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
    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_UPDATE_FAILED
    assert "busy" in data["error"]
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
    monkeypatch.setattr(update_auto, "_verify_health", lambda *_args: (False, "gateway startup failed"))

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

    ok, detail = update_auto._verify_health({"gateway_state": "stopped"})

    assert ok is True
    assert "already stopped" in detail
