"""Runtime regression for #60132: safe native capture in install.ps1."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
POWERSHELL = next(
    (
        executable
        for name in ("powershell.exe", "pwsh.exe", "pwsh")
        if (executable := shutil.which(name))
    ),
    None,
)

pytestmark = pytest.mark.skipif(
    POWERSHELL is None,
    reason="PowerShell is required to execute the Windows installer helper",
)


def test_native_capture_reports_streams_exit_status_and_cleans_temp_files(
    tmp_path: Path,
) -> None:
    """Execute the real helper against a controlled native Python process."""
    capture_temp = tmp_path / "capture-temp"
    capture_temp.mkdir()
    install_home = tmp_path / "hermes-home"
    install_dir = tmp_path / "hermes-agent"
    install_dir.mkdir()

    probe = tmp_path / "native_probe.py"
    probe.write_text(
        "import sys\n"
        "print('native-stdout')\n"
        "print('native-stderr', file=sys.stderr)\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    harness = tmp_path / "capture_harness.ps1"
    harness.write_text(
        r'''
$ErrorActionPreference = "Stop"
. $env:HERMES_TEST_INSTALLER `
    -HermesHome $env:HERMES_TEST_HOME `
    -InstallDir $env:HERMES_TEST_INSTALL_DIR

$before = @(Get-ChildItem -LiteralPath $env:HERMES_TEST_CAPTURE_TEMP -Force |
    ForEach-Object { $_.FullName })
$quotedProbe = '"' + $env:HERMES_TEST_PROBE.Replace('"', '\"') + '"'
$result = Invoke-NativeCaptureSafe `
    -FilePath $env:HERMES_TEST_PYTHON `
    -ArgumentList @($quotedProbe) `
    -WorkingDirectory $env:HERMES_TEST_WORK_DIR
$after = @(Get-ChildItem -LiteralPath $env:HERMES_TEST_CAPTURE_TEMP -Force |
    ForEach-Object { $_.FullName })
$leaked = @($after | Where-Object { $before -notcontains $_ })

[pscustomobject]@{
    ExitCode = $result.ExitCode
    LastExitCode = $global:LASTEXITCODE
    Stdout = $result.Stdout
    Stderr = $result.Stderr
    LeakedTempFiles = $leaked
} | ConvertTo-Json -Compress | Write-Output
''',
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "TEMP": str(capture_temp),
        "TMP": str(capture_temp),
        "TMPDIR": str(capture_temp),
        "HERMES_TEST_INSTALLER": str(INSTALL_PS1),
        "HERMES_TEST_HOME": str(install_home),
        "HERMES_TEST_INSTALL_DIR": str(install_dir),
        "HERMES_TEST_CAPTURE_TEMP": str(capture_temp),
        "HERMES_TEST_PYTHON": os.fspath(Path(os.sys.executable).resolve()),
        "HERMES_TEST_PROBE": str(probe),
        "HERMES_TEST_WORK_DIR": str(tmp_path),
    }
    completed = subprocess.run(
        [
            str(POWERSHELL),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(harness),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )

    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["ExitCode"] == 7
    assert payload["LastExitCode"] == 7
    assert payload["Stdout"].strip() == "native-stdout"
    assert payload["Stderr"].strip() == "native-stderr"
    assert payload["LeakedTempFiles"] == []
