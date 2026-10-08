"""Scheduling contracts, pure rendering and exact artifact restoration."""

from dataclasses import FrozenInstanceError
from datetime import datetime
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import update_auto_schedule as scheduler
from hermes_cli import update_auto_schedule_common as common
from hermes_cli import update_auto_schedule_launchd as launchd
from hermes_cli import update_auto_schedule_systemd as systemd


@pytest.fixture
def spec(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return scheduler.SchedulerSpec("v1-" + "a" * 24, ["/opt/hermes", "update", "auto", "run-scheduled"], home, "04:00", ["21:00"])


@pytest.mark.parametrize("time", ["00:00", "23:59", "03:07"])
def test_valid_time(time):
    assert scheduler.validate_time(time) == time


@pytest.mark.parametrize("time", ["3:00", "03:0", "24:00", "00:60", " 03:00", "03:00\n", "０３:００", "", None])
def test_invalid_time(time):
    with pytest.raises(ValueError, match="HH:MM"):
        scheduler.validate_time(time)


@pytest.mark.parametrize("hour,minute,action", [(21, 5, "plan"), (0, 15, "plan"), (3, 59, "plan"), (4, 0, "run"), (20, 59, "run")])
def test_late_daily_dispatch_preserves_plan_after_midnight(hour, minute, action):
    assert scheduler.scheduled_action("04:00", ["21:00"], datetime(2026, 7, 10, hour, minute)) == action


def test_overlap_prefers_nonmutating_plan():
    assert scheduler.scheduled_action("04:00", ["04:00"], datetime(2026, 7, 10, 4, 1)) == "plan"


def test_spec_copies_sequences_and_freezes_identity(spec):
    command, times = ["/bin/hermes"], ["01:00"]
    copied = scheduler.SchedulerSpec(spec.identity, command, spec.home, "04:00", times)
    command.append("untrusted")
    times.append("02:00")
    assert copied.command == ("/bin/hermes",)
    assert copied.plan_times == ("01:00",)
    with pytest.raises(FrozenInstanceError):
        copied.identity = "b" * 24


@pytest.mark.parametrize("identity", ["../foreign", "x" * 24, "a" * 25, "a/b"])
def test_identity_cannot_escape_owned_namespace(spec, identity):
    with pytest.raises(ValueError, match="identity"):
        scheduler.SchedulerSpec(identity, spec.command, spec.home, "04:00")


@pytest.mark.parametrize("command", [[], ["hermes"], ["/bin/hermes", "one\nExecStart=other"], ["/bin/hermes", "x\0y"]])
def test_invalid_command_refused(spec, command):
    with pytest.raises(ValueError):
        scheduler.SchedulerSpec(spec.identity, command, spec.home, "04:00")


def test_systemd_render_keeps_argument_values_literal(spec):
    special = scheduler.SchedulerSpec(spec.identity, ["/path with space/hermes", "$NAME", "%h", "quote\"back\\slash"], spec.home, "04:00", ["21:00", "21:00"])
    service, timer = systemd.render(special)
    text = service.decode()
    assert 'ExecStart=:"/path with space/hermes" "$NAME" "%%h" "quote\\"back\\\\slash"' in text
    assert f'Environment="HERMES_HOME={spec.home}"' in text
    assert timer.decode().count("OnCalendar=*-*-* 21:00:00") == 1
    assert "Persistent=true" in timer.decode()
    assert "TimeoutStartSec=infinity" in text
    assert f"StandardOutput=append:{spec.home}/logs/update-auto.out.log\n" in text
    assert f"StandardError=append:{spec.home}/logs/update-auto.err.log\n" in text


def test_launchd_render_is_user_only_and_never_runs_at_load(spec):
    payload = plistlib.loads(launchd.render(spec))
    assert payload["ProgramArguments"] == list(spec.command)
    assert payload["EnvironmentVariables"]["HERMES_HOME"] == str(spec.home)
    assert payload["RunAtLoad"] is False
    assert payload["StartCalendarInterval"] == [{"Hour": 21, "Minute": 0}, {"Hour": 4, "Minute": 0}]
    assert "UserName" not in payload
    assert launchd.paths(spec)["path"].parent == Path.home() / "Library" / "LaunchAgents"


@pytest.mark.platforms("posix")
def test_restore_preserves_exact_bytes_and_permissions(tmp_path):
    path = tmp_path / "job.timer"
    path.write_bytes(b"prior\x00bytes\n")
    path.chmod(0o640)
    snapshot = common.snapshot_file(path)
    common.write_file(path, b"new", 0o600)
    receipt = common.restore_file(path, snapshot)
    assert receipt["ok"]
    assert common.snapshot_file(path) == snapshot
    json.dumps(receipt)


@pytest.mark.platforms("posix")
def test_symlink_refusal_never_overwrites_target(tmp_path):
    target = tmp_path / "foreign"
    target.write_text("keep")
    link = tmp_path / "job.timer"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink"):
        common.write_file(link, b"new")
    receipt = common.restore_file(link, (b"old", 0o600))
    assert not receipt["ok"]
    assert target.read_text() == "keep"
    assert link.is_symlink()


def test_rollback_failure_remains_failure_on_repeated_calls(tmp_path):
    calls = []

    def fail():
        calls.append(1)
        raise OSError("cannot restore")

    handle = scheduler.SchedulerHandle("test", tmp_path, fail)
    first = handle.rollback()
    assert not first["ok"]
    assert handle.rollback() == first
    assert calls == [1]


def test_process_boundary_has_timeout_and_never_uses_shell(monkeypatch):
    monkeypatch.setattr(common, "locate_command", lambda name: SimpleNamespace(command=("/resolved/" + name,)))
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(common.subprocess, "run", run)
    common.run_command("systemctl", ["--user", "show", "owned.timer"])
    command, kwargs = calls[0]
    assert command == ["/resolved/systemctl", "--user", "show", "owned.timer"]
    assert kwargs["timeout"] == 30
    assert kwargs.get("shell", False) is False


def test_process_timeout_is_reported(monkeypatch):
    monkeypatch.setattr(common, "locate_command", lambda name: SimpleNamespace(command=(name,)))

    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(common.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        common.run_command("systemctl", ["--user", "show"])


def test_unavailable_tool_is_not_an_absent_job(monkeypatch):
    monkeypatch.setattr(common, "locate_command", lambda name: SimpleNamespace(command=()))
    with pytest.raises(RuntimeError, match="unavailable"):
        common.run_command("launchctl", ["print", "owned"])
    assert not common.missing_scheduler(subprocess.CompletedProcess([], 127, "", "not found"))


def test_plan_cannot_suppress_update_schedule(spec):
    with pytest.raises(ValueError, match="must differ"):
        scheduler.SchedulerSpec(spec.identity, spec.command, spec.home, "04:00", ["04:00"])


@pytest.mark.platforms("linux")
def test_rendered_units_pass_installed_systemd_parser(spec, tmp_path):
    """Offline parser validation only: never load or start a service."""
    resolution = common.locate_command("systemd-analyze")
    if not resolution.command:
        pytest.skip("systemd-analyze is not installed")
    unusual_home = tmp_path / 'profile with space\\quote"$name%h'
    actual = scheduler.SchedulerSpec(spec.identity, [sys.executable, "-c", "print('unused')"],
                                     unusual_home, spec.schedule, spec.plan_times)
    service, timer = systemd.render(actual)
    service_path = tmp_path / "hermes-auto-check.service"
    timer_path = tmp_path / "hermes-auto-check.timer"
    service_path.write_bytes(service)
    timer_path.write_bytes(timer)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    result = subprocess.run([*resolution.command, "--user", "verify", "--man=no",
                             str(service_path), str(timer_path)],
                            env={**os.environ, "XDG_RUNTIME_DIR": str(runtime)},
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    # The host may ship unrelated user sockets whose paths exceed sun_path in
    # long test roots. Any diagnostic from OUR generated units is a failure.
    diagnostics = [line for line in result.stderr.splitlines()
                   if line.startswith((str(service_path), str(timer_path)))]
    assert diagnostics == [], result.stderr
