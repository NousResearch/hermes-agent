"""Native regression coverage for the Windows Desktop handoff cwd."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.windows_only

REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_UPDATE_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


def _run_cwd_self_test(
    install_root: Path,
    launch_cwd: Path,
    temp_dir: Path,
) -> subprocess.CompletedProcess[str]:
    powershell = shutil.which("powershell.exe")
    assert powershell, "Windows updater tests require Windows PowerShell."
    temp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    return subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WINDOWS_UPDATE_PS1),
            "-InstallRoot",
            str(install_root),
            "-SelfTestWorkingDirectory",
            "-NoUi",
        ],
        cwd=launch_cwd,
        env=env,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )


def test_handoff_children_run_from_install_root(tmp_path: Path) -> None:
    install_root = tmp_path / "checkout"
    launch_cwd = tmp_path / "profile-home"
    install_root.mkdir()
    launch_cwd.mkdir()

    result = _run_cwd_self_test(install_root, launch_cwd, tmp_path / "temp")

    assert result.returncode == 0, result.stdout
    assert "WORKING-DIRECTORY SELF-TEST: PASS" in result.stdout


def test_handoff_fails_closed_when_install_root_cannot_be_entered(tmp_path: Path) -> None:
    install_root = tmp_path / "missing" / "checkout"
    launch_cwd = tmp_path / "profile-home"
    launch_cwd.mkdir()

    result = _run_cwd_self_test(install_root, launch_cwd, tmp_path / "temp")

    assert result.returncode == 3, result.stdout
    assert "cannot enter the install root" in result.stdout
