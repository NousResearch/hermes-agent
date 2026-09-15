"""install.ps1 desktop stage + single authoritative stage list (audit C27).

The retained desktop stage used to call helpers deleted by the PM
consolidation (Resolve-UvCmd, Test-Node, the electron-dist recovery set) —
runtime failures under `-Stage desktop`. The stage must use the CURRENT
paths: pm's venv sync for the wake/voice extras and the product CLI
(`hermes desktop --build-only`) for the build. The $Stages table must be
the ONE list: -Manifest prints it and the no-flag ladder runs it, so
-IncludeDesktop affects the real loop exactly as the manifest advertises.
`-Stage desktop` stays directly dispatchable without the flag (the
bootstrap frontend iterates manifest stages and always pairs the flag;
standalone dispatch is the long-standing contract).

Boundary: the test generates a PowerShell wrapper that defines stub
functions (New-Object intercepting WScript.Shell, icacls, ie4uinit.exe)
and then DOT-SOURCES the real install.ps1 with -Stage desktop — the full
stage runs in one real PowerShell process against a temp home/install
dir, with every external effect either fake (compiled external bootstrap python)
or logged instead of written. Nothing touches the user's known folders.
If the artifact check fails, the assertion message carries the fake
python's actual logged arguments for debugging.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.installation_launcher_fixture import publish_fixture_launcher

pytestmark = pytest.mark.platforms("windows")

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"

_FAKE_PY = r'''
using System;
using System.IO;

public static class FakePy {
    public static int Main(string[] args) {
        string log = Environment.GetEnvironmentVariable("FAKE_PY_LOG");
        File.AppendAllText(log, string.Join("\u0001", args) + Environment.NewLine);
        if (Array.IndexOf(args, "-c") >= 0) { return 0; }
        if (args.Length >= 2
                && args[args.Length - 2] == "desktop"
                && args[args.Length - 1] == "--build-only") {
            string dir = Path.Combine(
                Environment.GetEnvironmentVariable("FAKE_INSTALL_DIR"),
                "apps", "desktop", "release", "win-unpacked");
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, "Hermes.exe"), "fake");
        }
        return 0;
    }
}
'''

# The test boundary: stub the three external-effect primitives, then run the
# REAL installer. Dot-sourcing binds -Stage desktop and exits this wrapper
# with the installer's own exit code. Functions win over external commands in
# PowerShell's name resolution, so `& icacls` / `& ie4uinit.exe` hit the
# stubs; other New-Object callers go through the module-qualified real one.
_WRAPPER = r'''
param(
    [Parameter(Mandatory = $true)][string]$InstallerPath,
    [Parameter(Mandatory = $true)][string]$HermesHome,
    [Parameter(Mandatory = $true)][string]$InstallDir
)
$ErrorActionPreference = "Stop"

function New-StubShortcut {
    $sc = [pscustomobject]@{
        TargetPath       = ""
        WorkingDirectory = ""
        IconLocation     = ""
        Description      = ""
    }
    $sc | Add-Member -MemberType ScriptMethod -Name Save -Value {
        Add-Content -Path $env:WSH_LOG `
            -Value "SAVED:$($this.TargetPath)|$($this.WorkingDirectory)|$($this.IconLocation)"
    }
    return $sc
}

function New-StubShell {
    $shell = [pscustomobject]@{}
    $shell | Add-Member -MemberType ScriptMethod -Name CreateShortcut -Value {
        return New-StubShortcut
    }
    return $shell
}

function New-Object {
    param([string]$ComObject, [string]$TypeName, [object[]]$ArgumentList)
    if ($ComObject -eq "WScript.Shell") {
        return New-StubShell
    }
    if ($TypeName) {
        return Microsoft.PowerShell.Utility\New-Object -TypeName $TypeName -ArgumentList $ArgumentList
    }
    return Microsoft.PowerShell.Utility\New-Object -ComObject $ComObject -ArgumentList $ArgumentList
}

function New-Item {
    param([string]$ItemType, [switch]$Force, [string]$Path)
    if (-not [IO.Path]::GetFullPath($Path).StartsWith(
        [IO.Path]::GetFullPath($env:FAKE_INSTALL_DIR), [StringComparison]::OrdinalIgnoreCase)) {
        throw "test blocked directory creation outside temporary install: $Path"
    }
    Microsoft.PowerShell.Management\New-Item -ItemType $ItemType -Force:$Force -Path $Path
}

function icacls {
    Add-Content -Path $env:ICACLS_LOG -Value ("icacls " + ($args -join " "))
    $global:LASTEXITCODE = 0
}

function ie4uinit.exe {
    Add-Content -Path $env:ICACLS_LOG -Value ("ie4uinit.exe " + ($args -join " "))
    $global:LASTEXITCODE = 0
}

# Load the definitions, then execute the real stage dispatcher.
. $InstallerPath -InstallDir $InstallDir -HermesHome $HermesHome
function Get-BootstrapPython { return $env:FAKE_BOOT_PY }
Invoke-StageByName 'desktop'
exit $LASTEXITCODE
'''


def _compile_fake_python(powershell: str, output: Path) -> None:
    source = output.with_suffix(".cs")
    source.write_text(_FAKE_PY, encoding="utf-8")
    compile_script = output.with_name("compile-fake-py.ps1")
    compile_script.write_text(
        "param([string]$Source, [string]$Output)\n"
        "Add-Type -Path $Source -OutputAssembly $Output "
        "-OutputType ConsoleApplication\n",
        encoding="utf-8",
    )
    subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(compile_script), "-Source", str(source), "-Output", str(output)],
        check=True, capture_output=True, text=True, timeout=120,
        stdin=subprocess.DEVNULL,
    )


def _run(powershell: str, tmp_path: Path, args: list[str], env: dict[str, str] | None = None):
    return subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(INSTALL_PS1), *args,
         "-HermesHome", str(tmp_path / "hermes-home"),
         "-InstallDir", str(tmp_path / "install")],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def _manifest_stages(run) -> list[str]:
    assert run.returncode == 0, run.stdout + run.stderr
    return [s["name"] for s in json.loads(run.stdout)["stages"]]


def test_manifest_without_flag_lists_no_desktop(tmp_path: Path) -> None:
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")
    assert "desktop" not in _manifest_stages(_run(powershell, tmp_path, ["-Manifest"]))


def test_manifest_with_include_desktop_orders_desktop_before_complete(tmp_path: Path) -> None:
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")
    names = _manifest_stages(_run(powershell, tmp_path, ["-Manifest", "-IncludeDesktop"]))
    assert names.index("desktop") < names.index("complete")
    assert names[-1] == "complete"


def test_unknown_stage_is_rejected(tmp_path: Path) -> None:
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")
    run = _run(powershell, tmp_path, ["-Stage", "bogus", "-Json"])
    assert run.returncode == 2
    frame = json.loads(run.stdout.splitlines()[-1])
    assert frame["ok"] is False and frame["stage"] == "bogus"


def test_complete_stage_writes_pinned_install_marker(tmp_path: Path) -> None:
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")
    install = tmp_path / "install"
    install.mkdir()
    commit = "a" * 40
    run = _run(powershell, tmp_path, ["-Stage", "complete", "-Commit", commit, "-Json"])
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout.splitlines()[-1])["ok"] is True
    marker = json.loads((install / ".hermes-bootstrap-complete").read_text(encoding="utf-8-sig"))
    assert marker["pinnedCommit"] == commit
    assert marker["pinnedBranch"] == "main"


def test_desktop_stage_uses_pm_sync_and_product_cli(tmp_path: Path) -> None:
    """-Stage desktop (without -IncludeDesktop — the standalone contract)
    runs the CURRENT paths: pm's venv sync for wake/voice, then
    `hermes desktop --build-only` through the published installation launcher; the produced
    artifact is probed, ACL-granted, and shortcut-ed — with icacls,
    ie4uinit.exe, and WScript.Shell intercepted in the wrapper boundary so
    nothing outside the temp dirs is touched."""
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")

    install_dir = tmp_path / "install"
    scripts = tmp_path / "store" / "python"
    scripts.mkdir(parents=True)
    py_log = tmp_path / "fake-python.log"
    wsh_log = tmp_path / "wsh.log"
    icacls_log = tmp_path / "icacls.log"
    fake_python = scripts / "python.exe"
    _compile_fake_python(powershell, fake_python)

    publish_fixture_launcher(install_dir, "import os, subprocess, sys\ndef main():\n    return subprocess.call([os.environ['FAKE_BOOT_PY'], *sys.argv[1:]])\nif __name__ == '__main__': sys.exit(main())\n")
    runtime_dir = install_dir / "scripts" / "desktop-update"
    runtime_dir.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / "scripts/desktop-update/runtime.ps1", runtime_dir / "runtime.ps1")

    wrapper = tmp_path / "boundary-wrapper.ps1"
    wrapper.write_text(_WRAPPER, encoding="utf-8-sig")
    env = {
        **os.environ,
        "PATHEXT": ";".join(dict.fromkeys([*os.environ.get("PATHEXT", "").split(";"), ".EXE"])),
        "FAKE_PY_LOG": str(py_log),
        "FAKE_BOOT_PY": str(fake_python),
        "HERMES_RUNTIME_DIR": str(tmp_path / "empty-store"),
        "FAKE_INSTALL_DIR": str(install_dir),
        "WSH_LOG": str(wsh_log),
        "ICACLS_LOG": str(icacls_log),
    }
    run = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(wrapper),
         "-InstallerPath", str(INSTALL_PS1),
         "-HermesHome", str(tmp_path / "hermes-home"),
         "-InstallDir", str(install_dir)],
        cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, check=False, timeout=180,
    )

    calls = [l.split("\u0001") for l in py_log.read_text().splitlines()] if py_log.exists() else "NO LOG"
    assert run.returncode == 0, f"{run.stdout}{run.stderr}\nFAKE LOG:\n{calls}"

    assert isinstance(calls, list)
    # 1. wake/voice extras via pm's venv sync (the pm-owned path).
    assert any(
        len(c) >= 3 and c[:2] == ["-I", "-c"] and "from pm import sync_venv" in c[2]
        and "'wake'" in c[2] and "'voice'" in c[2] and "explicit=True" in c[2]
        for c in calls
    ), calls
    # 2. the build through the parsed product CLI (never a `build` subcommand
    #    or any deleted helper).
    assert any(c[-2:] == ["desktop", "--build-only"] for c in calls), calls
    # 3. the stage probed the artifact the fake build produced.
    exe = install_dir / "apps" / "desktop" / "release" / "win-unpacked" / "Hermes.exe"
    assert exe.is_file(), calls
    # 4. ACL grant hit the intercepted icacls with the produced exe's dir.
    icacls_lines = icacls_log.read_text().splitlines()
    assert any(
        line.startswith("icacls ") and str(exe.parent) in line
        and "*S-1-15-2-2:(OI)(CI)(RX)" in line
        for line in icacls_lines
    ), icacls_lines
    # 5. icon-cache bust hit the intercepted ie4uinit.exe stub.
    assert any(line.startswith("ie4uinit.exe") for line in icacls_lines), icacls_lines
    # 6. shortcut creation went through the intercepted WScript.Shell stub:
    #    logged, pointing at the produced exe, and NOT written to any real
    #    known folder.
    shortcuts = wsh_log.read_text().splitlines()
    assert len(shortcuts) == 2, shortcuts
    for line in shortcuts:
        assert line.startswith("SAVED:"), shortcuts
        target = line.split("|")[0][len("SAVED:"):]
        assert target == str(exe), shortcuts
