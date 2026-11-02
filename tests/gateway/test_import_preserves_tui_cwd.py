"""Importing gateway helpers must not apply messaging-only cwd defaults."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("configured", ["", ".", "explicit"])
def test_gateway_import_preserves_cwdless_tui_workspace(tmp_path, configured):
    repo = Path(__file__).resolve().parents[2]
    home = tmp_path / "home"
    launch = tmp_path / "workspace"
    home.mkdir()
    launch.mkdir()
    cwd = launch.as_posix() if configured == "explicit" else configured
    (home / "config.yaml").write_text(f"terminal:\n  cwd: '{cwd}'\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k.upper() in {
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
    }}
    for key in ("HOME", "USERPROFILE", "HERMES_HOME", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP"):
        env[key] = str(home)
    env["PYTHONPATH"] = str(repo)
    result = subprocess.run(
        [sys.executable, "-c", """
from pathlib import Path
import tui_gateway.server as server
launch = Path.cwd()
assert Path(server._completion_cwd({})) == launch
import gateway.run
assert Path(server._completion_cwd({})) == launch, (
    'gateway import changed cwd-less TUI workspace', server._completion_cwd({}), str(launch)
)
"""],
        cwd=launch, env=env, capture_output=True, text=True, timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "backend,mount,configured,legacy,expected",
    [
        ("local", "false", ".", None, "home"),
        ("", "false", "", None, "home"),
        ("docker", "false", ".", None, None),
        ("docker", "true", ".", None, None),
        ("docker", "true", ".", "legacy", "legacy"),
        ("ssh", "false", ".", None, None),
        ("local", "false", "explicit", None, "explicit"),
        ("local", "false", ".", "legacy", "legacy"),
    ],
)
def test_gateway_start_keeps_messaging_cwd_defaults(
    monkeypatch, tmp_path, backend, mount, configured, legacy, expected,
):
    import asyncio
    from gateway.run import start_gateway
    from hermes_cli import resource_limits

    class StopBeforeStartup(Exception):
        pass

    def stop():
        raise StopBeforeStartup

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("TERMINAL_ENV", backend)
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", mount)
    monkeypatch.setenv("TERMINAL_CWD", configured)
    if legacy is None:
        monkeypatch.delenv("MESSAGING_CWD", raising=False)
    else:
        monkeypatch.setenv("MESSAGING_CWD", legacy)
    monkeypatch.setattr(resource_limits, "apply_nofile_soft_limit", stop)
    with pytest.raises(StopBeforeStartup):
        asyncio.run(start_gateway())
    assert os.environ.get("TERMINAL_CWD") == (str(tmp_path) if expected == "home" else expected)
