"""Runtime regression for #60131: desktop failure must not block finalization."""

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
    reason="PowerShell is required to execute the Windows installer stage protocol",
)


def test_desktop_failure_emits_skipped_frame_and_reaches_finalization(
    tmp_path: Path,
) -> None:
    """Run the real stage protocol with controlled desktop/path workers."""
    install_home = tmp_path / "hermes-home"
    install_dir = tmp_path / "hermes-agent"
    install_home.mkdir()
    install_dir.mkdir()

    harness = tmp_path / "desktop_soft_fail_harness.ps1"
    harness.write_text(
        r'''
$ErrorActionPreference = "Stop"
. $env:HERMES_TEST_INSTALLER `
    -HermesHome $env:HERMES_TEST_HOME `
    -InstallDir $env:HERMES_TEST_INSTALL_DIR `
    -IncludeDesktop `
    -Json

function Install-Desktop { throw "forced desktop failure" }
function Set-PathVariable { $script:PathFinalizationReached = $true }

$script:PathFinalizationReached = $false
$frames = @()
foreach ($stageDef in $InstallStages) {
    if ($stageDef.Name -notin @("desktop", "path")) { continue }
    $frameJson = Invoke-Stage -StageDef $stageDef
    $frames += ($frameJson | ConvertFrom-Json)
}

[pscustomobject]@{
    Frames = $frames
    PathFinalizationReached = $script:PathFinalizationReached
} | ConvertTo-Json -Depth 5 -Compress | Write-Output
''',
        encoding="utf-8",
    )

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
        env={
            **os.environ,
            "HERMES_TEST_INSTALLER": str(INSTALL_PS1),
            "HERMES_TEST_HOME": str(install_home),
            "HERMES_TEST_INSTALL_DIR": str(install_dir),
        },
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )

    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    desktop_frame, path_frame = payload["Frames"]

    assert desktop_frame["stage"] == "desktop"
    assert desktop_frame["ok"] is True
    assert desktop_frame["skipped"] is True
    assert "forced desktop failure" in desktop_frame["reason"]
    assert path_frame["stage"] == "path"
    assert path_frame["ok"] is True
    assert path_frame["skipped"] is False
    assert payload["PathFinalizationReached"] is True
