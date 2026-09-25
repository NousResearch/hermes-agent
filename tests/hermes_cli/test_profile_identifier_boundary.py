"""Strict profile identifiers must not admit trailing shell line separators."""
import asyncio
import os
from pathlib import Path
import subprocess

import pytest
from fastapi import HTTPException
from hermes_cli import profiles


@pytest.mark.parametrize("validator", [profiles.validate_profile_name, profiles.validate_alias_name])
def test_strict_identifier_rejects_trailing_newline(validator):
    validator("work-bot")
    with pytest.raises(ValueError):
        validator("work-bot\n")


def test_setup_command_rejects_newline_before_shell_dispatch(tmp_path, monkeypatch):
    from hermes_cli.web_routers.profiles import _profile_setup_command

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _profile_setup_command("default") == "hermes setup"
    with pytest.raises(HTTPException) as error:
        _profile_setup_command("default\n")
    assert error.value.status_code == 400


def test_linux_terminal_commands_keep_profile_out_of_shell_text():
    from hermes_cli.web_routers.profiles import _linux_terminal_commands

    # Exercise the command builder even with a value a future caller might forget to validate.
    name = "worker'; touch /tmp/should-not-run; echo '"
    commands = _linux_terminal_commands(name)
    assert commands
    assert all(name not in arg for _, argv in commands for arg in argv)
    assert all("HERMES_SETUP_PROFILE_NAME" in " ".join(argv) for _, argv in commands)


def test_macos_terminal_argv_keeps_profile_out_of_applescript():
    from hermes_cli.web_routers.profiles import _macos_terminal_argv

    name = "worker\" & do shell script \"touch should-not-run"
    args = _macos_terminal_argv(name)
    assert args[:2] == ["osascript", "-e"]
    assert args[-2:] == ["--", name]
    assert name not in args[2]
    assert "quoted form of (item 1 of argv)" in args[2]
    assert _macos_terminal_argv("default")[-1] == "--"


@pytest.mark.macos_only
def test_osascript_option_delimiter_is_not_a_run_argument():
    result = subprocess.run(
        ["osascript", "-e", "on run argv\nreturn item 1 of argv\nend run", "--", "worker"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "worker"


@pytest.mark.linux_only
def test_linux_terminal_launch_passes_profile_as_one_argument(tmp_path):
    from hermes_cli.web_routers.profiles import _linux_terminal_commands

    capture = tmp_path / "args"
    fake_hermes = tmp_path / "hermes"
    fake_hermes.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    fake_hermes.chmod(0o755)
    name = "worker; touch should-not-run"
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
           "CAPTURE": str(capture), "HERMES_SETUP_PROFILE_NAME": name}
    _, terminal_argv = _linux_terminal_commands(name)[0]

    # Exercise the exact script with sh; login-shell PATH setup is terminal/host specific.
    subprocess.run(["sh", "-c", terminal_argv[-1]], env=env, cwd=tmp_path, check=True)

    assert capture.read_text().splitlines() == ["-p", name, "setup"]
    assert not (tmp_path / "should-not-run").exists()


@pytest.mark.windows_only
def test_open_profile_terminal_scopes_child_home_on_windows(monkeypatch, tmp_path):
    from hermes_cli.web_routers import profiles as routes

    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n")
    (home / "active_profile").write_text("worker\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE_NAME", "other")
    monkeypatch.setenv("HERMES_PROFILE", "other")
    calls = []
    monkeypatch.setattr(routes.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))

    # A→B→A checks that opening a named profile never retargets the dashboard's
    # own environment or a later default launch.
    for name, expected in (
        ("default", ["hermes", "-p", "default", "setup"]),
        ("worker", ["hermes", "setup"]),
        ("default", ["hermes", "-p", "default", "setup"]),
    ):
        result = asyncio.run(routes.open_profile_terminal_endpoint(name))
        assert result["ok"] is True
        args, kwargs = calls[-1]
        assert args == (expected,)
        assert kwargs["creationflags"] == subprocess.CREATE_NEW_CONSOLE
        assert kwargs["env"]["HERMES_HOME"] == str(home if name == "default" else profile)
        assert "HERMES_PROFILE_NAME" not in kwargs["env"]
        assert "HERMES_PROFILE" not in kwargs["env"]
        assert os.environ["HERMES_HOME"] == str(home), "the dashboard process stays on its own home"

        # The launched command has no profile name in argv; the child still resolves the
        # selected home to the intended profile identity.
        with monkeypatch.context() as child:
            child.setenv("HERMES_HOME", kwargs["env"]["HERMES_HOME"])
            child.delenv("HERMES_PROFILE_NAME")
            child.delenv("HERMES_PROFILE")
            assert profiles.get_active_profile_name() == name
    assert len(calls) == 3
