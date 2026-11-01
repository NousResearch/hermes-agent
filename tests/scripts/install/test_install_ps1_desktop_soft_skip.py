"""Windows desktop stage is optional: an npm/Electron failure skips it, not the install.

Stage-Desktop used to let any desktop build failure Fail the whole installer
(the npm "Cannot read properties of null" class of breakage, a missing Hermes.exe
after a failed pack, an unreachable Electron download). The desktop app is an
optional product on top of a working CLI: the installer must finish the stages
that make the CLI usable (products, config, setup, gateway, complete) and report
the desktop stage as skipped with the reason and a manual rebuild command, the
same way the needs-user-input stages report themselves skipped.

The boundary harness dot-sources the REAL install.ps1, forces
Invoke-SourceCompletion to throw, and runs Stage-Desktop: it must return, set
the stage's skipped reason, and print the manual rebuild hint -- while
Invoke-StageByName stays usable for the stages after it.
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
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"

# The forced failure must carry text the skip reason can quote, so the
# assertion checks the protocol, not the installer's exact wording.
FORCED_REASON = "forced desktop build failure"

_HARNESS = r'''
param(
    [Parameter(Mandatory = $true)][string]$InstallerPath,
    [Parameter(Mandatory = $true)][string]$HermesHome,
    [Parameter(Mandatory = $true)][string]$InstallDir
)
$ErrorActionPreference = "Stop"
. $InstallerPath -HermesHome $HermesHome -InstallDir $InstallDir
function Invoke-SourceCompletion { throw "''' + FORCED_REASON + r'''" }
function Publish-UserCommand { }
Stage-Desktop
[pscustomobject]@{
    reason = $script:StageSkippedReason
} | ConvertTo-Json -Compress | Write-Output
'''


def _powershell() -> str:
    for name in ("powershell", "pwsh"):
        found = shutil.which(name)
        if found:
            return found
    pytest.fail("no PowerShell host found")

def test_desktop_stage_failure_skips_the_stage_and_keeps_the_install_alive(tmp_path: Path) -> None:
    powershell = _powershell()
    harness = tmp_path / "desktop-soft-skip-harness.ps1"
    harness.write_text(_HARNESS, encoding="utf-8-sig")

    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("HERMES_")},
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "HERMES_RUNTIME_DIR": str(tmp_path / "runtime"),
    }
    run = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(harness),
         "-InstallerPath", str(INSTALL_PS1),
         "-HermesHome", env["HERMES_HOME"],
         "-InstallDir", str(tmp_path / "install")],
        cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, check=False, timeout=180,
    )
    # Stage-Desktop itself must not throw: reaching the harness's tail with a
    # JSON line on stdout is the proof the installer survived the failure.
    assert run.returncode == 0, run.stdout + run.stderr
    payload = json.loads(run.stdout.strip().splitlines()[-1])
    assert FORCED_REASON in payload["reason"], payload
    # The manual rebuild command the user needs is part of the report.
    assert "hermes desktop --build-only --force-build" in payload["reason"], payload
    # The skip was also explained on the console, not only the JSON channel.
    assert "Desktop app build failed" in run.stdout, run.stdout


def test_other_stage_failures_still_fail_the_stage(tmp_path: Path) -> None:
    """Only the desktop stage soft-fails: another stage's Fail still throws.

    Stage-Prerequisites' Ensure-Git failure is the existing hard-fail path
    (used by Stage-Complete too); the try/catch must be scoped to
    Stage-Desktop, not bolted onto the dispatcher for every stage.
    """
    powershell = _powershell()
    harness = tmp_path / "hard-fail-harness.ps1"
    harness.write_text(
        '$ErrorActionPreference = "Stop"\n'
        f'. "{INSTALL_PS1}" -HermesHome "{tmp_path / "hermes-home"}" -InstallDir "{tmp_path / "install"}"\n'
        "function Ensure-Git { return $false }\n"
        "try { Stage-Prerequisites; exit 0 } catch { exit 1 }\n",
        encoding="utf-8-sig",
    )
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("HERMES_")},
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "HERMES_RUNTIME_DIR": str(tmp_path / "runtime"),
    }
    run = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(harness)],
        cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, check=False, timeout=120,
    )
    assert run.returncode == 1, run.stdout + run.stderr
