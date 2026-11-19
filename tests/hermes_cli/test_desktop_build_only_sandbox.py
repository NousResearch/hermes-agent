"""Build-only artifacts receive the same Linux sandbox repair as launch (#58593)."""
import argparse
import stat
import subprocess

import pytest

from hermes_cli import main as cli_main
from hermes_cli import main_desktop as desktop
from hermes_cli import main_desktop_apparmor


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("sudo_ok", [True, False])
def test_build_only_repairs_packaged_sandbox_without_launch(tmp_path, monkeypatch, sudo_ok):
    root = tmp_path / "repo"
    app = root / "apps/desktop/release/linux-unpacked/hermes"
    app.parent.mkdir(parents=True)
    app.write_text("fixture", encoding="utf-8")
    (root / "apps/desktop/package.json").write_text("{}", encoding="utf-8")
    sandbox = app.with_name("chrome-sandbox")
    sandbox.write_text("fixture", encoding="utf-8")
    sandbox.chmod(0o644)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(desktop, "_desktop_launch_env", lambda args: ({}, []))
    monkeypatch.setattr(desktop, "_desktop_build_needed", lambda *a, **kw: False)
    monkeypatch.setattr(desktop, "_register_linux_desktop_entry", lambda **kw: None)
    monkeypatch.setattr(main_desktop_apparmor, "chromium_userns_sandbox_verdict", lambda exe: False)
    monkeypatch.setattr(desktop.shutil, "which", lambda name, **kw: "/fixture/sudo" if name == "sudo" else None)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert command[0] == "/fixture/sudo"  # No Electron launch or real privileged operation.
        if "chmod" in command and sudo_ok:
            sandbox.chmod(0o4755)
        return subprocess.CompletedProcess(command, 0 if sudo_ok else 1)

    monkeypatch.setattr(desktop.subprocess, "run", run)
    args = argparse.Namespace(build_only=True, source=False, skip_build=False, force_build=False,
                              setup_tcc_identity=False)
    desktop.cmd_gui(args)
    normalized = [[arg for arg in command[1:] if arg != "-n"] for command in commands]
    expected = [["chown", "root:root", str(sandbox)]]
    if sudo_ok:
        expected.append(["chmod", "4755", str(sandbox)])
    assert normalized == expected
    assert stat.S_IMODE(sandbox.stat().st_mode) == (0o4755 if sudo_ok else 0o644)
