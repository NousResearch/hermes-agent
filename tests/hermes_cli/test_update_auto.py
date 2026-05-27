from __future__ import annotations

import json
import plistlib
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import update_auto


def _args(**overrides):
    base = {"branch": None, "force": False}
    base.update(overrides)
    return SimpleNamespace(**base)


def _read_status(hermes_home):
    return json.loads((hermes_home / "state" / "update-status.json").read_text(encoding="utf-8"))


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


def test_enable_creates_expected_launchd_plist_on_macos(tmp_path, monkeypatch, capsys):
    hermes_home, calls = _set_macos_scheduler_env(tmp_path, monkeypatch)

    update_auto.cmd_auto_enable(_args(time="03:00"))

    plist_path = tmp_path / "Library" / "LaunchAgents" / f"{update_auto.LAUNCHD_LABEL}.plist"
    assert plist_path.exists()
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)
    assert plist["Label"] == update_auto.LAUNCHD_LABEL
    assert plist["ProgramArguments"] == ["hermes", "update", "auto", "run-now"]
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
    assert "ExecStart=hermes update auto run-now" in service_text
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


def test_status_output_shows_enabled_schedule(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    update_auto.write_status(
        {
            "enabled": True,
            "mode": "scheduled",
            "schedule": "03:00",
            "schedulerType": "launchd",
            "schedulerPath": "/tmp/com.hermes.agent.auto-update.plist",
            "status": update_auto.STATUS_SUCCESS,
        }
    )

    update_auto.cmd_auto_status(SimpleNamespace())

    out = capsys.readouterr().out
    assert "Enabled:          yes" in out
    assert "Schedule:         03:00" in out
    assert "Scheduler:        launchd" in out
    assert "Scheduler path:   /tmp/com.hermes.agent.auto-update.plist" in out


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
    monkeypatch.setattr(update_auto, "_verify_health", lambda: (True, "ok"))

    update_auto.cmd_auto_run_now(_args())

    assert len(update_calls) == 1
    called_args = update_calls[0]
    assert called_args.yes is True
    assert called_args.no_backup is True
    assert called_args.backup is False
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

    monkeypatch.setattr(hm, "_get_update_check_result", lambda **_kw: {"update_available": True, "latest_version": "new"})
    monkeypatch.setattr(
        backup,
        "create_pre_update_backup",
        lambda: hermes_home / "backups" / "pre-update.zip",
    )
    monkeypatch.setattr(hm, "cmd_update", lambda _args: None)
    monkeypatch.setattr(update_auto, "_current_version", lambda: "old")
    monkeypatch.setattr(update_auto, "_verify_health", lambda: (False, "gateway startup failed"))

    with pytest.raises(SystemExit) as exc:
        update_auto.cmd_auto_run_now(_args())

    assert exc.value.code == update_auto.EXIT_HEALTH_FAILED
    data = _read_status(hermes_home)
    assert data["status"] == update_auto.STATUS_HEALTH_FAILED
    assert "health check failed" in data["error"]
    assert "gateway startup failed" in data["error"]
    assert "health_failed" in capsys.readouterr().err
