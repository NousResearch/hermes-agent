"""install.ps1 must not persist test/ephemeral home launchers into User PATH.

Smoke harnesses install into temporary HERMES_HOMEs under %TEMP% (pytest tmp
roots land there too). ``Set-LauncherUserPath`` used to write every bin
directory into the persistent User PATH registry value, so a harness killed
before its ``finally`` cleanup left ``%TEMP%\\hermes_test_home_*\\bin`` behind
forever, and the stale ``hermes.exe`` shim shadowed the production launcher
in new shells (#125614). The guard must live in the installer, not in test
cleanup, because process-level ``finally`` cannot survive kill/timeout.
"""
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.platforms("windows")
INSTALLER = Path(__file__).resolve().parents[3] / "scripts" / "install.ps1"


def _ps_quote(value):
    return str(value).replace("'", "''")


def _run_snippet(tmp_path, body):
    powershell = shutil.which("powershell")
    assert powershell
    command = ". '" + _ps_quote(INSTALLER) + "'; " + body
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    markers = {}
    for line in result.stdout.splitlines():
        if "=" in line and line.split("=", 1)[0].isupper():
            key, value = line.split("=", 1)
            markers[key] = value
    return markers, result.stdout + result.stderr


def test_ephemeral_temp_home_leaves_user_path_untouched(tmp_path):
    """A TEMP-rooted home skips the persistent write but prepends process PATH."""
    body = (
        '$bin = "$env:TEMP\\hermes_test_home_125614\\bin"; '
        "$before = [Environment]::GetEnvironmentVariable('Path','User'); "
        'if ($before -like "*$bin*") { Write-Output \'PRECONDITION=polluted\'; exit 0 }; '
        "try { "
        "Set-LauncherUserPath $bin; "
        "$after = [Environment]::GetEnvironmentVariable('Path','User'); "
        'Write-Output ("USERPATH_SAME=" + ($before -eq $after)); '
        'Write-Output ("PROCESS_HAS_BIN=" + $env:Path.StartsWith($bin)); '
        "} finally { "
        "[Environment]::SetEnvironmentVariable('Path', $before, 'User'); "
        "}"
    )
    markers, output = _run_snippet(tmp_path, body)
    assert markers.get("PRECONDITION") != "polluted", (
        "User PATH already carries a hermes_test_home entry; clean it before running this test")
    assert markers.get("USERPATH_SAME") == "True", output
    assert markers.get("PROCESS_HAS_BIN") == "True", output


def test_production_home_prepends_once_and_is_idempotent(tmp_path):
    """A home outside TEMP keeps today's behavior: prepend exactly once."""
    body = (
        '$bin = "$env:USERPROFILE\\hermes_prod_like_home_125614\\bin"; '
        "$before = [Environment]::GetEnvironmentVariable('Path','User'); "
        'if ($before -like "*$bin*") { Write-Output \'PRECONDITION=polluted\'; exit 0 }; '
        "try { "
        "Set-LauncherUserPath $bin; "
        "$after = [Environment]::GetEnvironmentVariable('Path','User'); "
        'Write-Output ("PREPENDED=" + ($after -eq ($bin + \';\' + $before))); '
        "Set-LauncherUserPath $bin; "
        "$again = [Environment]::GetEnvironmentVariable('Path','User'); "
        'Write-Output ("IDEMPOTENT=" + ($again -eq $after)); '
        "} finally { "
        "[Environment]::SetEnvironmentVariable('Path', $before, 'User'); "
        "}"
    )
    markers, output = _run_snippet(tmp_path, body)
    assert markers.get("PRECONDITION") != "polluted", output
    assert markers.get("PREPENDED") == "True", output
    assert markers.get("IDEMPOTENT") == "True", output


def test_classifier_covers_ephemeral_and_production_roots(tmp_path):
    """Test-EphemeralLauncherHome classifies the homes the issue enumerates."""
    body = (
        '$t = "$env:TEMP"; $tm = "$env:TMP"; $la = "$env:LOCALAPPDATA"; $up = "$env:USERPROFILE"; '
        'Write-Output ("C0=" + (Test-EphemeralLauncherHome "$t\\hermes_test_home_abc\\bin")); '
        'Write-Output ("C1=" + (Test-EphemeralLauncherHome "$t\\pytest-of-user\\pytest-42\\homedir\\bin")); '
        'Write-Output ("C2=" + (Test-EphemeralLauncherHome "$tm\\arbitrary_home\\bin")); '
        'Write-Output ("C3=" + (Test-EphemeralLauncherHome "$la\\hermes\\bin")); '
        'Write-Output ("C4=" + (Test-EphemeralLauncherHome "$up\\custom\\hermes\\bin")); '
        'Write-Output ("C5=" + (Test-EphemeralLauncherHome "$la\\hermes_test_home_marker\\bin")); '
    )
    expected = {"C0": "True", "C1": "True", "C2": "True",
                "C3": "False", "C4": "False", "C5": "True"}
    markers, output = _run_snippet(tmp_path, body)
    for key, want in expected.items():
        assert markers.get(key) == want, f"{key}: expected {want}\n{output}"
