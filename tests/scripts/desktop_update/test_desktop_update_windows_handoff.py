"""Behavioral coverage for the Windows updater's relaunch gate.

The fixture executes the real ``windows.ps1``.  A temporary ``runtime.ps1``
stands in for the installed runtime so the test can fail either before the
update command is handed off or after it has actually run; the assertion is on
the hand-off log, not on PowerShell source text.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.platforms("windows")

REPO_ROOT = Path(__file__).resolve().parents[3]
WINDOWS_UPDATE_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"

_RUNTIME_PS1 = r'''
function Get-HermesRuntimeCommand {
    param([string]$InstallRoot, [string]$Module)
    if ($env:HERMES_HANDOFF_PRE_UPDATE_FAILURE -eq "1") {
        throw "fixture runtime lookup failed"
    }
    return @(
        (Join-Path $PSHOME "powershell.exe"),
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", $env:HERMES_HANDOFF_BACKEND
    )
}
'''

_BACKEND_PS1 = r'''
param([Parameter(ValueFromRemainingArguments = $true)] [string[]]$Arguments)
if ($Arguments -contains "--help") {
    Write-Output "fixture backend help"
    exit 0
}
if ($Arguments -contains "update") {
    Write-Output "fixture update failed after hand-off"
    exit 42
}
exit 0
'''


def _run_handoff(tmp_path: Path, *, pre_update_failure: bool) -> tuple[subprocess.CompletedProcess[str], str, dict]:
    powershell = shutil.which("powershell.exe")
    assert powershell, "native Windows acceptance requires PowerShell"

    install_root = tmp_path / "checkout"
    install_root.mkdir()
    # A PM-shaped install avoids the legacy retry path; the fixture only tests
    # the finalizer's relaunch decision.
    (install_root / "pm").mkdir()
    (install_root / "runtime.ps1").write_text(_RUNTIME_PS1, encoding="utf-8")
    backend = tmp_path / "backend.ps1"
    backend.write_text(_BACKEND_PS1, encoding="utf-8")

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    env = {
        **os.environ,
        "HERMES_HOME": str(hermes_home),
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "HERMES_HANDOFF_BACKEND": str(backend),
    }
    if pre_update_failure:
        env["HERMES_HANDOFF_PRE_UPDATE_FAILURE"] = "1"
    else:
        env.pop("HERMES_HANDOFF_PRE_UPDATE_FAILURE", None)

    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WINDOWS_UPDATE_PS1),
            "-InstallRoot",
            str(install_root),
            "-RelaunchExe",
            os.environ.get("ComSpec", r"C:\Windows\System32\cmd.exe"),
            "-NoUi",
            "-NoGateway",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
        check=False,
    )
    log = (hermes_home / "logs" / "desktop-update-handoff.log").read_text(
        encoding="utf-8", errors="replace"
    )
    receipt = json.loads(
        (hermes_home / ".hermes-update-result.json").read_text(
            encoding="utf-8-sig"
        )
    )
    return result, log, receipt


@pytest.mark.parametrize(
    ("pre_update_failure", "expected_code", "relaunch_expected"),
    [(True, 3, True), (False, 42, False)],
    ids=["before-update", "after-update"],
)
def test_relaunch_only_happens_before_update_boundary(
    tmp_path: Path,
    pre_update_failure: bool,
    expected_code: int,
    relaunch_expected: bool,
) -> None:
    result, log, receipt = _run_handoff(tmp_path, pre_update_failure=pre_update_failure)

    assert result.returncode == expected_code, result.stdout
    assert receipt["exit_code"] == expected_code
    assert ("relaunching desktop:" in log) is relaunch_expected
    if not relaunch_expected:
        assert "fixture update failed after hand-off" in log
