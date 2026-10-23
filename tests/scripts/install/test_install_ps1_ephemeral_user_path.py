"""Regression for #125614: ephemeral Windows homes must never mutate persistent User PATH.

The test dot-sources the real installer and replaces only its tiny User-PATH
accessors with an in-memory value. This exercises the real classifier and
Set-LauncherUserPath control flow without ever writing HKCU, so killing the
test cannot leave the machine in the polluted state this regression prevents.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALLER = REPO_ROOT / "scripts" / "install.ps1"
FAKE_USER_PATH = r"C:\existing\one;C:\existing\two"


def _quote(value: object) -> str:
    return str(value).replace("'", "''")


def _run(body: str) -> tuple[dict[str, str], str]:
    powershell = shutil.which("powershell")
    assert powershell
    command = ". '" + _quote(INSTALLER) + "'; " + body
    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    markers: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.isupper():
            markers[key] = value
    return markers, output


FAKE_ACCESSORS = r"""
$script:FakeUserPath = 'C:\existing\one;C:\existing\two'
$script:UserPathWrites = 0
function Get-LauncherUserPathValue { return $script:FakeUserPath }
function Set-LauncherUserPathValue([string]$Value) {
    $script:UserPathWrites += 1
    $script:FakeUserPath = $Value
}
"""


def test_temp_home_skips_persistent_write_but_updates_process_path():
    body = FAKE_ACCESSORS + r"""
$bin = Join-Path $env:TEMP 'arbitrary-hermes-home\bin'
Set-LauncherUserPath $bin
Write-Output ('USERPATH=' + $script:FakeUserPath)
Write-Output ('WRITES=' + $script:UserPathWrites)
Write-Output ('PROCESS_HAS_BIN=' + $env:Path.StartsWith(
    $bin, [StringComparison]::OrdinalIgnoreCase))
"""
    markers, output = _run(body)
    assert markers.get("USERPATH") == FAKE_USER_PATH, output
    assert markers.get("WRITES") == "0", output
    assert markers.get("PROCESS_HAS_BIN") == "True", output


def test_production_home_prepends_once_without_touching_hkcu():
    body = FAKE_ACCESSORS + r"""
$bin = Join-Path $env:USERPROFILE 'hermes-prod-like-125614\bin'
Set-LauncherUserPath $bin
Set-LauncherUserPath $bin
Write-Output ('USERPATH=' + $script:FakeUserPath)
Write-Output ('WRITES=' + $script:UserPathWrites)
Write-Output ('EXPECTED=' + ($bin + ';C:\existing\one;C:\existing\two'))
"""
    markers, output = _run(body)
    assert markers.get("USERPATH") == markers.get("EXPECTED"), output
    assert markers.get("WRITES") == "1", output


def test_classifier_uses_path_identity_not_textual_spelling():
    body = r"""
$tempHome = Join-Path $env:TEMP 'arbitrary-home\bin'
$tempForward = $tempHome.Replace('\', '/')
$tempSibling = $env:TEMP.TrimEnd('\', '/') + '-sibling\bin'
$markerHome = Join-Path $env:USERPROFILE 'scratch\hermes_test_home_abc\bin'
$markerSubstring = Join-Path $env:USERPROFILE 'scratch\not_hermes_test_home_backup\bin'
$production = Join-Path $env:LOCALAPPDATA 'hermes\bin'
Write-Output ('TEMP_NATIVE=' + (Test-EphemeralLauncherHome $tempHome))
Write-Output ('TEMP_FORWARD=' + (Test-EphemeralLauncherHome $tempForward))
Write-Output ('TEMP_SIBLING=' + (Test-EphemeralLauncherHome $tempSibling))
Write-Output ('MARKER_SEGMENT=' + (Test-EphemeralLauncherHome $markerHome))
Write-Output ('MARKER_SUBSTRING=' + (Test-EphemeralLauncherHome $markerSubstring))
Write-Output ('PRODUCTION=' + (Test-EphemeralLauncherHome $production))
"""
    markers, output = _run(body)
    expected = {
        "TEMP_NATIVE": "True",
        "TEMP_FORWARD": "True",
        "TEMP_SIBLING": "False",
        "MARKER_SEGMENT": "True",
        "MARKER_SUBSTRING": "False",
        "PRODUCTION": "False",
    }
    for key, value in expected.items():
        assert markers.get(key) == value, f"{key}: expected {value}\n{output}"
