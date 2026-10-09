"""The real installer waits for bootstrap uv before it delegates to PM."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.platforms("windows")
INSTALLER = Path(__file__).resolve().parents[3] / "scripts" / "install.ps1"


@pytest.mark.parametrize("stage,exit_code,skip_browser", [
    ("venv", 0, False), ("python-deps", 0, False), ("python-deps", 17, False),
    # -SkipBrowser is PM's persisted opt-out, handed over as `--without`.
    ("python-deps", 0, True),
])
def test_python_stage_delegates_to_pin_without_touching_existing_env(tmp_path, stage, exit_code, skip_browser):
    powershell = shutil.which("powershell")
    assert powershell
    install = tmp_path / "install with spaces"
    package = install / "pm"
    package.mkdir(parents=True)
    (package / "lock.json").write_text(json.dumps({"packages": {"python": {"version": "3.12.8+fixture"}}}), encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(
        "import json, os, pathlib, sys\n"
        "assert not pathlib.Path(os.environ['UV_BUSY']).exists()\n"
        "pathlib.Path(os.environ['PM_LOG']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(int(os.environ['PROBE_EXIT']))\n", encoding="utf-8",
    )
    existing = install / "venv" / "sentinel"
    existing.parent.mkdir()
    existing.write_text("previous generation", encoding="utf-8")
    log = tmp_path / "uv-calls.jsonl"
    pm_log = tmp_path / "pm-call.json"
    wrapper = tmp_path / "boundary.ps1"
    wrapper.write_text(r'''
param([string]$Installer, [string]$InstallDir, [string]$HomeDir, [string]$Log)
$ErrorActionPreference = 'Stop'
function uv {
    Set-Content -Path $env:UV_BUSY -Value 'running'
    try {
        ConvertTo-Json -Compress -InputObject @($args) | Add-Content -Encoding UTF8 $Log
        switch ($args[1]) {
            'install' { }
            'find' { Write-Output $env:BOOTSTRAP_PYTHON }
            default { throw 'PM must not be a child of the bootstrap uv' }
        }
        $global:LASTEXITCODE = 0
    } finally { Remove-Item $env:UV_BUSY }
}
function Invoke-WebRequest { throw 'network access outside test boundary' }
$installerArgs = @{ InstallDir = $InstallDir; HermesHome = $HomeDir }
if ($env:PROBE_SKIP_BROWSER) { $installerArgs.SkipBrowser = $true }
. $Installer @installerArgs
# Dot-sourcing defines Get-Uv, so override only that acquisition boundary.
function Get-Uv { return 'uv' }
Invoke-StageByName $env:PROBE_STAGE
exit $LASTEXITCODE
''', encoding="utf-8-sig")
    # uv python find returns the distribution interpreter, not a venv launcher.
    env = dict(os.environ, PROBE_STAGE=stage, PROBE_EXIT=str(exit_code), BOOTSTRAP_PYTHON=sys._base_executable,
               PM_LOG=str(pm_log), UV_BUSY=str(tmp_path / "uv-busy"), PATHEXT=".COM;.EXE;.BAT;.CMD")
    env.pop("PYTHONPATH", None)
    env.pop("PROBE_SKIP_BROWSER", None)
    if skip_browser:
        env["PROBE_SKIP_BROWSER"] = "1"
    run = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(wrapper),
                          "-Installer", str(INSTALLER), "-InstallDir", str(install), "-HomeDir", str(tmp_path / "home"), "-Log", str(log)],
                         cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
                         capture_output=True, text=True, timeout=120)
    assert (run.returncode == 0) == (exit_code == 0), run.stdout + run.stderr
    from hermes_platform.host.facts import native_arch

    # The request names the machine's architecture: a bare version lets uv
    # pick an emulated x86_64 build on Windows-on-ARM.
    uv_arch = "aarch64" if native_arch() == "arm64" else "x86_64"
    assert [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()] == [
        ["python", "find", "--managed-python", "--no-project", f"cpython-3.12-windows-{uv_arch}-none"],
    ]
    if stage == "python-deps":
        assert json.loads(pm_log.read_text(encoding="utf-8-sig")) == (
            ["install", "--without", "agent-browser"] if skip_browser else ["install"])
    else:
        assert not pm_log.exists()
    assert existing.read_text(encoding="utf-8-sig") == "previous generation"
