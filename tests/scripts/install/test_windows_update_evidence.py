"""A failed Windows packaged update keeps App Installer's version and the AppX logs.

Stable rehearsal rc.2-v0.23.2's x64 leg showed App Installer's "Cannot open app package:
Error in parsing the app package" for the NEW bundle on the Windows Server 2025 runner, and
its artifact held neither App Installer's version nor the AppX event that names the cause.
"""
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest

pytestmark = pytest.mark.platforms("windows")
COLLECTOR = Path(__file__).resolve().parents[3] / "tests" / "install" / "e2e-assets" / "windows-update-evidence.ps1"
IDENTITY = "NousResearch.Hermes.EvidenceTest"
LOGS = ("AppxPackaging_Operational", "AppXDeploymentServer_Operational",
        "AppXDeployment_Operational", "AppModel-Runtime_Admin")

# Functions defined before the call shadow the cmdlets inside the collector's scope.
FAKE_HOST = r"""
$at = [datetime]::new(2026, 11, 13, 9, 0, 0, [DateTimeKind]::Utc)
function Get-AppxPackage { [CmdletBinding()] param([string]$Name)
    $versions = @{ 'Microsoft.DesktopAppInstaller' = '1.29.379.0'; 'IDENTITY' = '0.22.0.0' }
    if ($versions.ContainsKey($Name)) {
        [pscustomobject]@{ Name = $Name; Version = $versions[$Name]; Architecture = 'X64'; Status = 'Ok'
                           PackageFullName = "$($Name)_x64"; InstallLocation = "C:\Program Files\WindowsApps\$Name" }
    }
}
function Get-Process { [CmdletBinding()] param([string[]]$Name)
    if ($Name -contains 'AppInstaller') { [pscustomobject]@{ Id = 4242; StartTime = $at; Path = 'C:\AppInstaller.exe' } }
}
function New-FakeEvent($Minutes, $Id, $Level, $Message) {
    $names = @{ 2 = 'Error'; 3 = 'Warning'; 4 = 'Information' }
    [pscustomobject]@{ TimeCreated = $at.AddMinutes($Minutes); Id = $Id; Level = $Level
                       LevelDisplayName = $names[$Level]; Message = $Message }
}
$fixtureEvents = @{
    'Microsoft-Windows-AppxPackaging/Operational' = @(
        (New-FakeEvent 2 404 2 "Error in parsing the app package.`r`n  Hermes-0.22.0.msixbundle"),
        (New-FakeEvent 1 216 3 'App manifest validation warning for Microsoft.WindowsStore'),
        (New-FakeEvent 3 400 4 'Started deployment from http://127.0.0.1:8443/Hermes.appinstaller'))
    'Microsoft-Windows-AppModel-Runtime/Admin' = @(New-FakeEvent 5 69 3 'IDENTITY could not be activated')
}
function Get-WinEvent { [CmdletBinding()] param([hashtable]$FilterHashtable)
    $records = $fixtureEvents[$FilterHashtable.LogName]
    if ($records) { return $records }
    Write-Error -Message 'No events were found that match the specified selection criteria.' -ErrorId NoMatchingEventsFound
}
""".replace("IDENTITY", IDENTITY)

# The real Get-WinEvent for two channels: one that does not exist, one with nothing in the window.
MISSING_AND_EMPTY_CHANNEL = FAKE_HOST + r"""
function Get-WinEvent { [CmdletBinding()] param([hashtable]$FilterHashtable)
    $filter = $FilterHashtable.Clone()
    switch ($filter.LogName) {
        'Microsoft-Windows-AppXDeploymentServer/Operational' { $filter.LogName = 'Microsoft-Windows-HermesEvidenceTest-Missing/Operational' }
        'Microsoft-Windows-AppXDeployment/Operational' { $filter.StartTime = (Get-Date).AddDays(1) }
        default { return $fixtureEvents[$filter.LogName] }
    }
    Microsoft.PowerShell.Diagnostics\Get-WinEvent -FilterHashtable $filter
}
"""


def _collect(tmp_path: Path, fakes: str = "") -> tuple[subprocess.CompletedProcess, Path]:
    """Run the collector in the e2e script's context: Stop, StrictMode Latest, inside its try."""
    powershell = shutil.which("powershell")
    assert powershell, "Windows PowerShell 5.1 is part of every supported Windows"
    out = tmp_path / "evidence"
    script = (
        "$ErrorActionPreference = 'Stop'\nSet-StrictMode -Version Latest\n" + fakes
        + f"\ntry {{ & '{COLLECTOR}' -OutDir '{out}' -Since (Get-Date).AddHours(-1) -Identity '{IDENTITY}' }}\n"
        + "catch { Write-Host \"  [evidence] collection failed: $($_.Exception.Message)\" }\n"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
                            capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "collection failed" not in result.stdout, result.stdout
    return result, out


def _lines(out: Path, log: str) -> list[str]:
    return (out / f"Microsoft-Windows-{log}.txt").read_text(encoding="utf-8-sig").splitlines()


def test_collects_packages_and_appx_logs_without_the_package_installed(tmp_path):
    result, out = _collect(tmp_path)

    packages = json.loads((out / "packages.json").read_text(encoding="utf-8-sig"))
    assert all(p["Name"] != IDENTITY for p in packages)
    for log in LOGS:
        assert (out / f"Microsoft-Windows-{log}.txt").is_file()
    assert f"saved under {out}" in result.stdout


def test_saves_package_process_and_event_records_and_logs_only_relevant_problems(tmp_path):
    result, out = _collect(tmp_path, FAKE_HOST)

    packages = json.loads((out / "packages.json").read_text(encoding="utf-8-sig"))
    assert {(p["Name"], p["Version"], p["Status"]) for p in packages} == {
        ("Microsoft.DesktopAppInstaller", "1.29.379.0", "Ok"), (IDENTITY, "0.22.0.0", "Ok")}
    processes = json.loads((out / "appinstaller-processes.json").read_text(encoding="utf-8-sig"))
    assert [p["Id"] for p in processes] == [4242]
    assert "Microsoft.DesktopAppInstaller 1.29.379.0 X64 Ok" in result.stdout

    # Every record of the window is saved, oldest first, one line each.
    assert _lines(out, "AppxPackaging_Operational") == [
        "2026-11-13T09:01:00.0000000Z 216 Warning App manifest validation warning for Microsoft.WindowsStore",
        "2026-11-13T09:02:00.0000000Z 404 Error Error in parsing the app package. Hermes-0.22.0.msixbundle",
        "2026-11-13T09:03:00.0000000Z 400 Information Started deployment from http://127.0.0.1:8443/Hermes.appinstaller",
    ]
    assert "Microsoft-Windows-AppxPackaging/Operational: 3 events" in result.stdout
    assert "Microsoft-Windows-AppXDeployment/Operational: 0 events" in result.stdout
    assert _lines(out, "AppXDeployment_Operational") == []

    # The job log shows warnings and errors that name this update, nothing else.
    assert "404 Error: Error in parsing the app package. Hermes-0.22.0.msixbundle" in result.stdout
    assert f"69 Warning: {IDENTITY} could not be activated" in result.stdout
    assert "Microsoft.WindowsStore" not in result.stdout
    assert "Started deployment" not in result.stdout


def test_a_missing_channel_is_saved_as_unread_and_the_other_logs_are_still_read(tmp_path):
    result, out = _collect(tmp_path, MISSING_AND_EMPTY_CHANNEL)

    unread = _lines(out, "AppXDeploymentServer_Operational")
    assert len(unread) == 1 and unread[0].startswith("could not read: "), unread
    assert "Microsoft-Windows-AppXDeploymentServer/Operational: could not read: " in result.stdout
    # A window with no events is not a read failure.
    assert _lines(out, "AppXDeployment_Operational") == []
    assert "Microsoft-Windows-AppXDeployment/Operational: 0 events" in result.stdout
    assert len(_lines(out, "AppxPackaging_Operational")) == 3
    assert _lines(out, "AppModel-Runtime_Admin") == [
        f"2026-11-13T09:05:00.0000000Z 69 Warning {IDENTITY} could not be activated"]
    assert f"69 Warning: {IDENTITY} could not be activated" in result.stdout


def test_keeps_app_installer_diagnostics_written_since_the_leg_began(tmp_path):
    local = tmp_path / "localappdata"
    diag = local / "Packages" / "Microsoft.DesktopAppInstaller_8wekyb3d8bbwe" / "LocalState" / "DiagOutputDir"
    diag.mkdir(parents=True)
    older = diag / "WinGetCOM-before.log"
    older.write_text("before the leg\n", encoding="utf-8")
    os.utime(older, (time.time() - 7200, time.time() - 7200))
    log = ("[2026/11/13 14:34:07.537] GoToState -> LoadingState\n"
           "[2026/11/13 14:34:07.602] ERROR: AppsInfo -> Error: [0x8051100f]\n")
    (diag / "AppInstaller-during.log").write_text(log, encoding="utf-8")

    result, out = _collect(tmp_path, f"$env:LOCALAPPDATA = '{local}'\n" + FAKE_HOST)

    saved = out / "appinstaller-diagnostics"
    assert [p.name for p in saved.iterdir()] == ["AppInstaller-during.log"]
    assert (saved / "AppInstaller-during.log").read_text(encoding="utf-8") == log
    assert "App Installer diagnostics: 1 files since the leg began" in result.stdout
    assert "    [2026/11/13 14:34:07.602] ERROR: AppsInfo -> Error: [0x8051100f]" in result.stdout
    assert "GoToState" not in result.stdout
