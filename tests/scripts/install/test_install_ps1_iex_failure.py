"""A failed install run through `iex (irm ...)` must not end the caller's session.

`iex` runs the installer text inside the user's PowerShell session, so an
`exit` on failure closed their window. A script file run keeps its exit code.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.platforms("windows")
INSTALLER = Path(__file__).resolve().parents[3] / "scripts" / "install.ps1"
REFUSAL = "exists and is not a Hermes git checkout"


def _failing_install_env(tmp_path):
    """An occupied non-checkout makes the repository stage Fail before any
    network or git call; a present pinned-git slot satisfies prerequisites."""
    home = tmp_path / "home"
    (home / "hermes-agent").mkdir(parents=True)
    (home / "hermes-agent" / "user-file").write_text("preserve me", encoding="utf-8")
    tools = tmp_path / "tools"
    for arch in ("x64", "arm64"):
        git = tools / f"git-2.53.0+3-win32-{arch}" / "cmd" / "git.exe"
        git.parent.mkdir(parents=True)
        git.write_bytes(b"")
    return dict(os.environ, HERMES_HOME=str(home), HERMES_RUNTIME_DIR=str(tools))


def _powershell():
    powershell = shutil.which("powershell")
    assert powershell
    return powershell


def test_failed_iex_install_reports_and_returns_to_the_callers_session(tmp_path):
    command = (f"iex (Get-Content -Raw -LiteralPath '{INSTALLER}'); "
               "Write-Output \"session alive: $LASTEXITCODE\"")
    result = subprocess.run([_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
                            env=_failing_install_env(tmp_path), capture_output=True, text=True, timeout=60)
    assert REFUSAL in result.stdout
    assert "session alive: 1" in result.stdout, result.stdout + result.stderr
    assert (tmp_path / "home" / "hermes-agent" / "user-file").read_text(encoding="utf-8") == "preserve me"


def test_failed_file_install_still_exits_nonzero(tmp_path):
    result = subprocess.run([_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(INSTALLER)],
                            env=_failing_install_env(tmp_path), capture_output=True, text=True, timeout=60)
    assert REFUSAL in result.stdout
    assert result.returncode == 1
