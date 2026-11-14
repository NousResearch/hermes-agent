"""The Windows update leg exempts App Installer from loopback isolation only for itself.

On the windows-2025 runner, App Installer 1.26 fetched the leg's 127.0.0.1 feed from inside its
AppContainer, failed with 0x80072efd, and showed "Cannot open app package. Reason: Error in parsing
the app package." (rehearsals rc.1, rc.2 and rc.4 of v0.23.2). Reproduced on Windows 11 with App
Installer 1.26.509; with the exemption the same descriptor opened the install prompt.
"""
import base64
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.platforms("windows")
HELPER = Path(__file__).resolve().parents[3] / "tests" / "install" / "e2e-assets" / "windows-appinstaller-loopback.ps1"
FAMILY = "microsoft.desktopappinstaller_8wekyb3d8bbwe"

# A function named like the executable shadows it, so the helper drives this fake exemption list.
FAKE_CHECKNETISOLATION = r"""
$global:exempt = [System.Collections.Generic.List[string]]::new()
$global:calls = [System.Collections.Generic.List[string]]::new()
$global:refuse = 'REFUSE'
function CheckNetIsolation.exe {
    $global:calls.Add($args -join ' ')
    switch ($args[1]) {
        '-s' { foreach ($name in $global:exempt) { '    Name: ' + $name } }
        '-a' { if ($global:refuse -ne '-a') { $global:exempt.Add($args[2].Substring(3)) } }
        '-d' { if ($global:refuse -ne '-d') { [void]$global:exempt.Remove($args[2].Substring(3)) } }
    }
}
"""


def _run(body: str, *, refuse: str = "") -> subprocess.CompletedProcess:
    powershell = shutil.which("powershell")
    assert powershell, "Windows PowerShell 5.1 is part of every supported Windows"
    script = (FAKE_CHECKNETISOLATION.replace("REFUSE", refuse)
              + body.replace("HELPER", str(HELPER)).replace("FAMILY", FAMILY))
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    return subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
                          capture_output=True, text=True, timeout=60, check=False)


def test_leg_adds_the_exemption_and_removes_only_its_own(tmp_path):
    result = _run(r"""
$added = & 'HELPER' -Action Add
"added=$added exempt=$($global:exempt -join ',')"
if ($added) { & 'HELPER' -Action Remove }
"after=$($global:exempt -join ',')"
$global:exempt.Add('FAMILY')
$global:calls.Clear()
$again = & 'HELPER' -Action Add
"preexisting added=$again calls=$($global:calls.Count) exempt=$($global:exempt -join ',')"
""")
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line.strip() for line in result.stdout.splitlines()]
    assert f"added=True exempt={FAMILY}" in lines
    assert "after=" in lines
    # Someone else's exemption is reported as not ours, so the leg's teardown leaves it in place.
    assert f"preexisting added=False calls=1 exempt={FAMILY}" in lines


def test_leg_fails_when_the_exemption_does_not_take(tmp_path):
    result = _run("& 'HELPER' -Action Add", refuse="-a")
    assert result.returncode != 0
    assert "Could not exempt App Installer from loopback isolation" in result.stdout + result.stderr


def test_leg_fails_when_the_exemption_is_not_removed(tmp_path):
    # Windows PowerShell 5.1 ignores a native command's exit code, so the helper re-reads the list.
    result = _run("$global:exempt.Add('FAMILY'); & 'HELPER' -Action Remove", refuse="-d")
    assert result.returncode != 0
    assert "Could not remove App Installer's loopback exemption" in result.stdout + result.stderr
