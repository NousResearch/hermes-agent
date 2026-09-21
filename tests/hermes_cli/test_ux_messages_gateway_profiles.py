"""Invariant tests for the user-facing copy of gateway/profile/update-lock failures.

Contract, not snapshots: each message leads with plain words (never a raw exception), names the
next command, and every ``hermes <sub> <cmd>`` it cites is a registered subcommand.
"""

import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import gateway as gateway_cli
from hermes_cli import profiles
from hermes_cli.update_lock import UpdateHolder, describe_holder


# --- cli-05: gateway start/stop/restart failures leave gateway_command() as guidance ------------


def _run_gateway_command_raising(monkeypatch, capsys, exc):
    monkeypatch.setattr(gateway_cli, "_gateway_command_inner", lambda args: (_ for _ in ()).throw(exc))
    with pytest.raises(SystemExit) as info:
        gateway_cli.gateway_command(SimpleNamespace(gateway_command="start"))
    assert info.value.code == 1
    return capsys.readouterr().out


def test_systemctl_failure_is_explained_not_tracebacked(monkeypatch, capsys):
    exc = subprocess.CalledProcessError(1, ["systemctl", "--user", "start", "hermes-gateway"])
    out = _run_gateway_command_raising(monkeypatch, capsys, exc)
    assert not out.lstrip().startswith("Command '")
    assert "hermes gateway status --deep" in out
    assert "hermes gateway install --force" in out
    assert "journalctl" in out
    # The raw detail survives as a secondary line, not the lead sentence.
    assert "Details:" in out


def test_missing_systemctl_points_at_foreground_run(monkeypatch, capsys):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("systemctl")

    monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as info:
        gateway_cli._run_systemctl(["start", "hermes-gateway"])
    out = _run_gateway_command_raising(monkeypatch, capsys, info.value)
    assert "hermes gateway run" in out
    assert "systemd" in out
    assert not out.lstrip().startswith("systemctl is not available")


# --- cli-15: already-running guard and no-backend copy ---------------------------------------


def test_existing_gateway_guard_says_bots_are_online_and_names_status(monkeypatch, capsys):
    import gateway.status as status_mod

    monkeypatch.setattr(gateway_cli, "_running_under_gateway_supervisor", lambda: False)
    monkeypatch.setattr(status_mod, "get_running_pid", lambda: 12345)
    with pytest.raises(SystemExit):
        gateway_cli._guard_existing_gateway_process_conflict()
    out = capsys.readouterr().out
    assert "12345" in out
    assert "online" in out
    for cmd in ("hermes gateway status", "hermes gateway restart", "hermes gateway stop"):
        assert cmd in out
    assert "Another gateway instance" not in out


@pytest.mark.parametrize("subcommand", ["start", "uninstall"])
def test_unsupported_platform_copy_says_what_is_unsupported(subcommand):
    _code, *lines = gateway_cli._NO_BACKEND_MESSAGES[(subcommand, "unsupported")]
    text = "\n".join(lines)
    assert text != "Not supported on this platform."
    assert "background service" in text
    assert "hermes gateway run" in text or subcommand == "uninstall"


# --- cli-13 / cli-14: profile name errors -----------------------------------------------------


def test_invalid_profile_name_explains_rule_in_words_with_example():
    with pytest.raises(ValueError) as info:
        profiles.validate_profile_name("My Work")
    msg = str(info.value)
    assert "Must match" not in msg and "[a-z0-9]" not in msg
    assert "lowercase" in msg
    assert "my-work" in msg
    assert "hermes profile create my-work" in msg


def test_existing_profile_error_offers_use_and_list(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profiles.create_profile("work", no_alias=True, no_skills=True)
    with pytest.raises(FileExistsError) as info:
        profiles.create_profile("work", no_alias=True, no_skills=True)
    msg = str(info.value)
    assert "already exists at" not in msg
    assert "hermes profile use work" in msg
    assert "hermes profile list" in msg


def test_missing_profile_error_points_at_list(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError) as info:
        profiles.delete_profile("wrk")
    msg = str(info.value)
    assert "No profile named 'wrk'" in msg
    assert "hermes profile list" in msg


# --- cli-28: concurrent update refusal ---------------------------------------------------------


def test_describe_holder_avoids_jargon_and_names_next_steps():
    message = describe_holder(UpdateHolder(pid=4242, age_seconds=190))
    assert "4242" in message and "3m 10s" in message
    assert "PID" not in message and "checkout" not in message
    assert "hermes logs" in message
    assert "hermes update" in message
