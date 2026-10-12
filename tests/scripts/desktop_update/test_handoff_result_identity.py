"""Real PowerShell producer/heartbeat/result contract (portable, not native UI proof)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


@pytest.mark.platforms("any")
@pytest.mark.parametrize("run_id", ["", "receipt-test.bridge-2"])
def test_powershell_result_keeps_identity_after_actual_heartbeat(tmp_path: Path, run_id: str) -> None:
    if not POWERSHELL:
        pytest.skip("PowerShell is not installed")
    home = tmp_path / "home"
    home.mkdir()
    install = tmp_path / "checkout"
    (install / "pm").mkdir(parents=True)
    harness = tmp_path / "receipt.ps1"
    harness.write_text(r'''
param([string]$HandoffPath, [string]$Install, [string]$Run)
$started = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - 10
$env:HERMES_UPDATE_STARTED_AT = [string]$started
$marker = Join-Path $env:HERMES_HOME '.hermes-update-in-progress'
$invoke = @{ InstallRoot = $Install; NoUi = $true }
if ($Run) {
    $ct = [DateTimeOffset]::new([Diagnostics.Process]::GetCurrentProcess().StartTime.ToUniversalTime()).ToUnixTimeMilliseconds() / 1000.0
    $ctText = $ct.ToString('F3', [Globalization.CultureInfo]::InvariantCulture)
    [IO.File]::WriteAllText($marker, "$PID`n$started`nct:$ctText`nrun:$Run`n")
    $invoke.DesktopPid = $PID
    $invoke.HandoffRun = $Run
}
# Schedule a real heartbeat immediately before the real result writer. No
# function replacement, result JSON fabrication, or Windows-host emulation.
$breakpoint = Set-PSBreakpoint -Command Write-Result -Action {
    $script:MarkerLastHeartbeat = (Get-Date).AddMinutes(-10)
    Update-MarkerHeartbeat
    [IO.File]::Copy($MarkerPath, "$MarkerPath.observed", $true)
}
& $HandoffPath @invoke
Remove-PSBreakpoint $breakpoint
''', encoding="utf-8")
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-File", str(harness),
         "-HandoffPath", str(ROOT / "scripts/desktop-update/windows.ps1"),
         "-Install", str(install), "-Run", run_id],
        env={**os.environ, "HOME": str(home), "HERMES_HOME": str(home),
             "TEMP": str(tmp_path), "TMP": str(tmp_path)},
        capture_output=True, text=True, timeout=60,
    )
    receipt = json.loads((home / ".hermes-update-result.json").read_text(encoding="utf-8"))
    assert receipt["ok"] is False, (receipt, result.stdout, result.stderr)
    # Native Windows reaches the missing-launcher refusal. PowerShell on POSIX
    # cannot call the Win32 console APIs, but its real finally writer still runs.
    if os.name == "nt":
        assert receipt["exit_code"] == 3, (receipt, result.stdout, result.stderr)
    else:
        assert receipt["exit_code"] == 1 and "DisableQuickEdit" in result.stderr
    lines = (home / ".hermes-update-in-progress.observed").read_text().splitlines()
    assert int(lines[1]) > receipt["started_at"], "actual heartbeat must change line 2"
    assert receipt.get("run_id"), receipt
    assert f"run:{receipt['run_id']}" in lines
    if run_id:
        assert receipt["run_id"] == run_id
    assert not (home / ".hermes-update-in-progress").exists(), "normal release still runs"
