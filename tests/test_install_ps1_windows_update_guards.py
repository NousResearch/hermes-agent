"""Behavioral regression coverage for Windows install/update guards."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
POWERSHELL = shutil.which("powershell")
UV = shutil.which("uv")
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


def _run_powershell(
    args: list[str],
    *,
    timeout: int = 60,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    assert POWERSHELL is not None
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", *args],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        cwd=cwd,
    )


@pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell unavailable")
def test_manifest_executes_under_windows_powershell_51_file_mode() -> None:
    result = _run_powershell(["-File", str(INSTALL_PS1), "-Manifest"])

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["protocol_version"] >= 1
    assert payload["stages"][-1]["name"] == "bootstrap-cache"


@pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell unavailable")
def test_mutable_cache_stage_can_evict_the_current_script(tmp_path: Path) -> None:
    cache_dir = tmp_path / "bootstrap-cache"
    cache_dir.mkdir()
    cached_script = cache_dir / "install-main.ps1"
    shutil.copy2(INSTALL_PS1, cached_script)

    result = _run_powershell(
        [
            "-File",
            str(cached_script),
            "-Stage",
            "bootstrap-cache",
            "-Json",
            "-NonInteractive",
            "-HermesHome",
            str(tmp_path),
            "-InstallDir",
            str(tmp_path / "hermes-agent"),
            "-Branch",
            "main",
        ]
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not cached_script.exists()


@pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell unavailable")
def test_commit_pinned_cache_is_retained(tmp_path: Path) -> None:
    cache_dir = tmp_path / "bootstrap-cache"
    cache_dir.mkdir()
    cached_script = cache_dir / "install-main.ps1"
    shutil.copy2(INSTALL_PS1, cached_script)

    result = _run_powershell(
        [
            "-File",
            str(cached_script),
            "-Stage",
            "bootstrap-cache",
            "-Json",
            "-NonInteractive",
            "-HermesHome",
            str(tmp_path),
            "-InstallDir",
            str(tmp_path / "hermes-agent"),
            "-Branch",
            "main",
            "-Commit",
            "02d26981d3d4",
        ]
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert cached_script.exists()


@pytest.mark.skipif(
    POWERSHELL is None or UV is None,
    reason="Windows PowerShell or managed uv unavailable",
)
def test_non_admin_task_fallback_and_uv_trampoline_descendant(tmp_path: Path) -> None:
    hermes_home = tmp_path / "hermes-home"
    install_dir = hermes_home / "hermes-agent"
    venv_scripts = install_dir / "venv" / "Scripts"
    service_dir = hermes_home / "gateway-service"
    bin_dir = hermes_home / "bin"
    for directory in (venv_scripts, service_dir, bin_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # The venv only needs to exist so Install-Venv enters its teardown path.
    (venv_scripts / "placeholder.txt").write_text("old", encoding="ascii")
    legacy_wrapper = service_dir / "Hermes_Gateway_Legacy.cmd"
    modern_wrapper = service_dir / "Hermes_Gateway.vbs"
    legacy_wrapper.write_text("@echo off\r\nexit /b 0\r\n", encoding="ascii")
    modern_wrapper.write_text("WScript.Quit 0\r\n", encoding="ascii")
    shutil.copy2(UV, bin_dir / "uv.exe")

    runtime_root = tmp_path / "runtime"
    temp_dir = runtime_root / "temp"
    local_app_data = runtime_root / "local-app-data"
    user_profile = runtime_root / "user-profile"
    uv_python_dir = runtime_root / "uv-python"
    uv_cache_dir = runtime_root / "uv-cache"
    for directory in (
        runtime_root,
        temp_dir,
        local_app_data,
        user_profile,
        uv_python_dir,
        uv_cache_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    repo_artifacts_before = {
        path.name
        for pattern in ("%SystemDrive%", "Python", "python_install_*.log")
        for path in REPO_ROOT.glob(pattern)
    }
    stopped_log = tmp_path / "stopped.txt"
    containment_log = tmp_path / "contained.txt"
    harness = tmp_path / "harness.ps1"
    harness.write_text(
        f"""
$env:OS = 'Windows_NT'
$env:SystemDrive = '{runtime_root}'
$env:TEMP = '{temp_dir}'
$env:TMP = '{temp_dir}'
$env:LOCALAPPDATA = '{local_app_data}'
$env:USERPROFILE = '{user_profile}'
$env:UV_PYTHON_INSTALL_DIR = '{uv_python_dir}'
$env:UV_CACHE_DIR = '{uv_cache_dir}'
$env:UV_PYTHON = '{Path(sys.executable)}'
$env:UV_PYTHON_DOWNLOADS = 'never'
$global:alive = @{{ 100 = $true; 101 = $true; 102 = $true; 103 = $true; 104 = $true }}
$global:legacyWrapper = '{legacy_wrapper}'
$global:modernWrapper = '{modern_wrapper}'
$global:stoppedLog = '{stopped_log}'
$global:containmentLog = '{containment_log}'

function global:Get-ScheduledTask {{
    @(
        [pscustomobject]@{{
            TaskName = 'Hermes_Gateway_Legacy'
            TaskPath = '\\'
            State = 'Running'
            Settings = [pscustomobject]@{{ Enabled = $true }}
            Actions = @([pscustomobject]@{{
                Execute = $global:legacyWrapper
                Arguments = ''
            }})
        }},
        [pscustomobject]@{{
            TaskName = 'Hermes_Gateway'
            TaskPath = '\\'
            State = 'Ready'
            Settings = [pscustomobject]@{{ Enabled = $true }}
            Actions = @([pscustomobject]@{{
                Execute = 'wscript.exe'
                Arguments = '//B //Nologo "' + $global:modernWrapper + '"'
            }})
        }}
    )
}}
function global:Stop-ScheduledTask {{ }}
function global:Disable-ScheduledTask {{ throw 'Access is denied' }}
function global:Enable-ScheduledTask {{ }}
function global:taskkill {{ }}
function global:Get-Process {{ throw 'module enumeration denied' }}
function global:Stop-Process {{
    param([int]$Id)
    Add-Content -LiteralPath $global:stoppedLog -Value $Id
    $global:alive.Remove($Id)
}}
function global:New-MockProcess {{
    param([int]$MockId, [switch]$Reused)
    switch ($MockId) {{
        100 {{ return [pscustomobject]@{{
            ProcessId = 100; ParentProcessId = 1; CreationDate = [datetime]'2026-07-18T12:00:00Z'
            Name = 'uv.exe'; ExecutablePath = '{venv_scripts / "uv.exe"}'; CommandLine = 'uv run gateway'
        }} }}
        101 {{ return [pscustomobject]@{{
            ProcessId = 101; ParentProcessId = 100; CreationDate = [datetime]'2026-07-18T12:00:01Z'
            Name = 'python.exe'; ExecutablePath = 'C:\\Python311\\python.exe'
            CommandLine = 'python.exe -m hermes_cli.main gateway run'
        }} }}
        102 {{ return [pscustomobject]@{{
            ProcessId = 102; ParentProcessId = 101; CreationDate = [datetime]'2026-07-18T12:00:02Z'
            Name = 'helper.exe'; ExecutablePath = 'C:\\Tools\\helper.exe'; CommandLine = 'helper.exe'
        }} }}
        103 {{ return [pscustomobject]@{{
            ProcessId = 103; ParentProcessId = 100
            CreationDate = $(if ($Reused) {{ [datetime]'2026-07-18T13:00:03Z' }} else {{ [datetime]'2026-07-18T12:00:03Z' }})
            Name = 'pythonw.exe'; ExecutablePath = 'C:\\Python311\\pythonw.exe'
            CommandLine = 'pythonw.exe -m hermes_cli.main gateway run'
        }} }}
        104 {{ return [pscustomobject]@{{
            ProcessId = 104; ParentProcessId = 100; CreationDate = [datetime]'2026-07-18T11:59:59Z'
            Name = 'python.exe'; ExecutablePath = 'C:\\Python311\\python.exe'
            CommandLine = 'python.exe -m hermes_cli.main gateway run'
        }} }}
    }}
}}
function global:Get-CimInstance {{
    param([string]$ClassName, [string]$Filter)
    if (
        -not (Test-Path -LiteralPath $global:legacyWrapper) -and
        -not (Test-Path -LiteralPath $global:modernWrapper)
    ) {{
        Set-Content -LiteralPath $global:containmentLog -Value 'wrapper-quarantined'
    }}
    if ($Filter -match 'ProcessId = (\d+)') {{
        $requested = [int]$matches[1]
        if (-not $global:alive.ContainsKey($requested)) {{ return $null }}
        if ($requested -eq 103) {{ return New-MockProcess -MockId 103 -Reused }}
        return New-MockProcess -MockId $requested
    }}
    $items = @()
    foreach ($mockId in @(100, 101, 102, 103, 104)) {{
        if ($global:alive.ContainsKey($mockId)) {{
            $items += New-MockProcess -MockId $mockId
        }}
    }}
    return $items
}}

& '{INSTALL_PS1}' -Stage venv -Json -NonInteractive `
    -HermesHome '{hermes_home}' -InstallDir '{install_dir}' -Branch main `
    -PythonVersion '{PYTHON_VERSION}'
""",
        encoding="ascii",
    )

    result = _run_powershell(["-File", str(harness)], timeout=120, cwd=tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert legacy_wrapper.exists(), "the legacy CMD wrapper must be restored"
    assert modern_wrapper.exists(), "the modern VBS launcher must be restored"
    assert containment_log.read_text(encoding="ascii").strip() == "wrapper-quarantined"
    assert stopped_log.exists(), result.stdout + result.stderr
    stopped = {int(line) for line in stopped_log.read_text(encoding="ascii").splitlines()}
    assert stopped == {100, 101}, (
        "only the proven uv seed and unchanged python trampoline may be stopped; "
        "a non-runtime child, a reused PID, and a child older than its parent "
        "must be preserved"
    )
    repo_artifacts_after = {
        path.name
        for pattern in ("%SystemDrive%", "Python", "python_install_*.log")
        for path in REPO_ROOT.glob(pattern)
    }
    assert repo_artifacts_after == repo_artifacts_before
