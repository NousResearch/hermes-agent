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
@pytest.mark.parametrize("name,expected", [
    ("default", ["hermes", "setup"]),
    ("worker", ["hermes", "-p", "worker", "setup"]),
])
def test_open_profile_terminal_uses_direct_argv_on_windows(name, expected, monkeypatch, tmp_path):
    from hermes_cli.web_routers import profiles as routes

    calls = []
    monkeypatch.setattr(routes, "_resolve_profile_dir", lambda profile: tmp_path)
    monkeypatch.setattr(routes.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))

    result = asyncio.run(routes.open_profile_terminal_endpoint(name))

    assert result["ok"] is True
    assert calls == [((expected,), {"creationflags": subprocess.CREATE_NEW_CONSOLE})]
