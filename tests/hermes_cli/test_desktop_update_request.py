"""One-shot Desktop launch action: no build and no lingering launcher (#118098)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from hermes_cli.subcommands.gui import build_gui_parser


def _args(argv):
    parser = argparse.ArgumentParser()
    build_gui_parser(parser.add_subparsers(dest="command"), cmd_gui=lambda _: None)
    return parser.parse_args(argv)


@pytest.mark.parametrize("alias", ["desktop", "gui"])
def test_update_action_bypasses_build_and_rejects_build_flags(alias, tmp_path, monkeypatch):
    from hermes_cli import main_desktop as desktop
    from hermes_cli import main as cli
    from hermes_cli import desktop_update_request as request

    root = tmp_path / "checkout"
    monkeypatch.setattr(cli, "PROJECT_ROOT", root)
    executable = tmp_path / "Hermes"
    executable.touch()
    monkeypatch.setattr(desktop, "_desktop_packaged_executable", lambda _: executable)
    monkeypatch.setattr(desktop, "_packaged_desktop_launch_command", lambda exe: [str(exe)])
    monkeypatch.setattr(desktop, "_desktop_launch_env", lambda _: ({}, []))
    calls = []
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return 0
    monkeypatch.setattr(request, "launch_desktop_update_all", launch)
    def build_forbidden(*args, **kwargs):
        pytest.fail("an update launch must not build or resolve npm")
    monkeypatch.setattr(desktop, "_desktop_build_needed", build_forbidden)
    monkeypatch.setattr(desktop, "build_prepared_desktop", build_forbidden)

    with pytest.raises(SystemExit) as exited:
        desktop.cmd_gui(_args([alias, "--update-all"]))
    assert exited.value.code == 0
    assert calls == [([str(executable)], {"cwd": root / "apps" / "desktop", "env": {}})]

    for flag in ("--source", "--build-only", "--force-build", "--setup-tcc-identity", "--ignore-existing"):
        with pytest.raises(SystemExit) as refused:
            desktop.cmd_gui(_args([alias, "--update-all", flag]))
        assert refused.value.code == 2
    assert len(calls) == 1
    monkeypatch.setattr(desktop, "_desktop_packaged_executable", lambda _: None)
    with pytest.raises(SystemExit) as missing:
        desktop.cmd_gui(_args([alias, "--update-all"]))
    assert missing.value.code == 1
    assert len(calls) == 1


# These cases run on their named, real hosts. No monkeypatch of sys.platform.
@pytest.mark.parametrize("host", [
    pytest.param("linux", marks=pytest.mark.platforms("linux")),
    pytest.param("darwin", marks=pytest.mark.platforms("macos")),
    pytest.param("win32", marks=pytest.mark.platforms("windows")),
])
def test_real_process_handoff_exits_before_dispatch_and_times_out_old_apps(host, tmp_path):
    marker = tmp_path / "consumed.json"
    app = tmp_path / "app.py"
    app.write_text("""
import json, pathlib, sys, time
import psutil
request = pathlib.Path(next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--hermes-update-all-request=')))
value = json.loads(request.read_text(encoding='utf-8'))
def write(value):
    staged = request.with_suffix('.tmp')
    staged.write_text(json.dumps(value), encoding='utf-8')
    staged.replace(request)
write({**value, 'state': 'accepted'})
while time.time() * 1000 < value['expires_at']:
    current = json.loads(request.read_text(encoding='utf-8'))
    if current['state'] == 'committed' and not any(psutil.pid_exists(p) for p in value['launcher_pids']):
        pathlib.Path(sys.argv[1]).write_text(json.dumps(current), encoding='utf-8')
        request.unlink()
        request.parent.rmdir()
        break
    time.sleep(0.05)
""", encoding="utf-8")
    launcher = (
        "from pathlib import Path; import os, sys; "
        "from hermes_cli.desktop_update_request import launch_desktop_update_all; "
        f"sys.exit(launch_desktop_update_all([sys.executable, {str(app)!r}, {str(marker)!r}], "
        "cwd=Path.cwd(), env=dict(os.environ), timeout=5))"
    )
    result = subprocess.run([sys.executable, "-c", launcher], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
    assert result.returncode == 0, result.stderr
    assert "accepted" in result.stdout
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.exists(), "app must survive the launcher and consume only after its exit"
    assert json.loads(marker.read_text(encoding="utf-8"))["state"] == "committed"

    # An older app ignores the switch. The request must time out without commit.
    old = (
        "from pathlib import Path; import os, sys; "
        "from hermes_cli.desktop_update_request import launch_desktop_update_all; "
        "sys.exit(launch_desktop_update_all([sys.executable, '-c', 'pass'], "
        "cwd=Path.cwd(), env=dict(os.environ), timeout=0.2))"
    )
    result = subprocess.run([sys.executable, "-c", old], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
    assert result.returncode == 1
    assert "no update was committed" in result.stderr
