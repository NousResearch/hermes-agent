"""Native PowerShell prerequisite selection without downloading build tools."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.platforms("windows")
HELPER = Path(__file__).resolve().parents[2] / "scripts" / "windows-build-deps.ps1"


def test_openssl_installs_once_and_rejects_damaged_shared_install(tmp_path):
    script = tmp_path / "check.ps1"
    script.write_text(r'''
param([string]$Helper, [string]$Root)
$ErrorActionPreference = 'Stop'
. $Helper
$prefix = Join-Path $Root 'installed\arm64-windows-static-md'
function Invoke-HermesBuildCommand {
    param([string]$Command, [string[]]$Arguments)
    if ($Arguments[0] -ne 'install' -or $Arguments[1] -ne 'openssl:arm64-windows-static-md') {
        throw 'incorrect installation request'
    }
    if ($Arguments -notcontains '--classic') { throw 'manifest mode was not disabled' }
    $script:calls += 1
    if ($script:calls -gt 1) { return } # vcpkg trusts its installed database on subsequent requests.
    New-Item -ItemType Directory -Force (Join-Path $prefix 'lib'), (Join-Path $prefix 'include\openssl') | Out-Null
    foreach ($relative in @('lib\libcrypto.lib', 'lib\libssl.lib', 'include\openssl\ssl.h')) {
        [IO.File]::WriteAllText((Join-Path $prefix $relative), 'fixture')
    }
}
$calls = 0
$first = Install-HermesArm64OpenSSL -Vcpkg 'fixture-vcpkg' -Root $Root
$second = Install-HermesArm64OpenSSL -Vcpkg 'fixture-vcpkg' -Root $Root
if ($calls -ne 1 -or $first -ne $prefix -or $second -ne $prefix) { throw 'warm setup installed twice' }
Remove-Item (Join-Path $prefix 'include\openssl\ssl.h')
$rejected = $false
try { Install-HermesArm64OpenSSL -Vcpkg 'fixture-vcpkg' -Root $Root } catch {
    if ($_.Exception.Message -notmatch 'installation is damaged') { throw }
    $rejected = $true
}
if (-not $rejected -or $calls -ne 2) { throw 'damaged install was accepted' }
Write-Output 'PASS'
''', encoding="utf-8")
    env = dict(os.environ)
    env.setdefault("SystemRoot", r"C:\Windows")
    shell = shutil.which("powershell") or str(Path(env["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe")
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script), str(HELPER), str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_native_build_command_preserves_failures_and_spaces(tmp_path):
    script = tmp_path / "native.ps1"
    script.write_text(r'''
param([string]$Helper, [string]$Root)
$ErrorActionPreference = 'Stop'
. $Helper
$command = Join-Path $Root 'child with spaces.cmd'
[IO.File]::WriteAllText($command, "@echo off`r`necho native-progress 1>&2`r`nexit /b 19`r`n")
$failed = $false
try { Invoke-HermesBuildCommand $command @() } catch {
    if ($_.Exception.Message -notmatch 'exit code 19') { throw }
    $failed = $true
}
if (-not $failed -or $ErrorActionPreference -ne 'Stop') { throw 'failure or shell preference was lost' }
[IO.File]::WriteAllText($command, "@echo off`r`necho native-progress 1>&2`r`nexit /b 0`r`n")
Invoke-HermesBuildCommand $command @()
$failed = $false
try { Invoke-HermesBuildCommand (Join-Path $Root 'absent.exe') @() } catch { $failed = $true }
if (-not $failed) { throw 'missing executable was accepted after a successful command' }
Write-Output 'PASS'
''', encoding="utf-8")
    env = dict(os.environ)
    env.setdefault("SystemRoot", r"C:\Windows")
    env.setdefault("ComSpec", str(Path(env["SystemRoot"]) / "System32/cmd.exe"))
    env.setdefault("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    shell = shutil.which("powershell") or str(Path(env["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe")
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script), str(HELPER), str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
