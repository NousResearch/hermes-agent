"""Regression: Install-Uv must surface installer errors and have fallbacks.

Issue #69216: Windows installs died with only the generic message
``uv installed but not found at ...\\bin\\uv.exe``.  Two root causes:

1. ``Install-Uv`` piped the astral installer's entire output straight into
   ``Out-Null`` (``2>&1 | Out-Null``), so any real failure -- download error,
   corporate proxy block, AV quarantine, permissions -- was swallowed and
   the user only ever saw the generic post-condition failure (first
   identified in #69366).
2. There was exactly one install source, ``astral.sh``.  Corporate proxies
   commonly block astral.sh while the byte-identical installer published at
   GitHub releases downloads fine (diagnosed by @gakugaku on #69216).

The fix installs a three-rung ladder inside ``Install-Uv``:

- Rung 1: astral.sh installer, output captured via ``Tee-Object``.
- Rung 2: GitHub releases installer mirror, same ``UV_INSTALL_DIR``.
- Rung 3: salvage an existing ``uv.exe`` (``Get-Command uv`` or
  ``%USERPROFILE%\\.local\\bin\\uv.exe``) by copying it into
  ``$HermesHome\\bin\\uv.exe`` so the managed-first invariant holds.

Only after all three rungs fail does it error out -- and then it prints the
tail of the captured installer output so the real cause reaches the user.

install.ps1 only runs on Windows, so the harness tests dot-source it and run
under ``windows_only``; source-text assertions were dropped (they tested the
shape of the script, not behaviour).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_INSTALL_PS1 = Path(__file__).resolve().parents[3] / "scripts" / "install.ps1"

@pytest.mark.windows_only
def test_windows_rerun_self_heals_when_managed_uv_is_broken(tmp_path: Path):
    """When $HermesHome/bin/uv.exe is already a broken shim, Install-Uv must purge it and self-heal."""
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("powershell not found on PATH")

    hermes_home = tmp_path / "hermes_home"
    bin_dir = hermes_home / "bin"
    bin_dir.mkdir(parents=True)
    broken_managed_uv = bin_dir / "uv.exe"
    # Compiled outside bin/ so it survives Install-Uv purging the managed copy;
    # it doubles as the fail-fast stand-in for the installer host exe.
    broken_uv = tmp_path / "BrokenUv.exe"

    # Create a broken executable that fails uv --version (exit code 1)
    source_cs = tmp_path / "BrokenUv.cs"
    source_cs.write_text(
        "using System;\n"
        "public class BrokenUv {\n"
        "  public static int Main(string[] args) {\n"
        "    Console.Error.WriteLine(\"Cannot find file at '..\\\\lib\\\\uv\\\\tools\\\\uv.exe'\");\n"
        "    return 1;\n"
        "  }\n"
        "}\n",
        encoding="ascii",
    )
    compile_ps1 = tmp_path / "compile.ps1"
    compile_ps1.write_text(
        f"Add-Type -Path '{source_cs}' -OutputAssembly '{broken_uv}' -OutputType ConsoleApplication\n",
        encoding="ascii",
    )
    subprocess.run([powershell, "-ExecutionPolicy", "Bypass", "-File", str(compile_ps1)], check=True)
    shutil.copy(broken_uv, broken_managed_uv)
    assert broken_managed_uv.exists()

    # Also compile a valid uv that exits 0 with "uv 0.5.0"
    valid_uv = tmp_path / "valid_uv.exe"
    valid_cs = tmp_path / "ValidUv.cs"
    valid_cs.write_text(
        "using System;\n"
        "public class ValidUv {\n"
        "  public static int Main(string[] args) {\n"
        "    Console.WriteLine(\"uv 0.5.0\");\n"
        "    return 0;\n"
        "  }\n"
        "}\n",
        encoding="ascii",
    )
    compile_valid_ps1 = tmp_path / "compile_valid.ps1"
    compile_valid_ps1.write_text(
        f"Add-Type -Path '{valid_cs}' -OutputAssembly '{valid_uv}' -OutputType ConsoleApplication\n",
        encoding="ascii",
    )
    subprocess.run([powershell, "-ExecutionPolicy", "Bypass", "-File", str(compile_valid_ps1)], check=True)
    assert valid_uv.exists()

    # Run a test script that dot-sources install.ps1, mocks network installers to fail,
    # mocks Get-Command uv to return the valid uv, and verifies Install-Uv purges the broken managed uv
    # and replaces it with the valid candidate.
    test_harness = tmp_path / "test_harness.ps1"
    test_harness.write_text(
        r'''param([string]$InstallPs1, [string]$HomeDir, [string]$ValidCandidate, [string]$HostExe)
$env:HERMES_HOME = $HomeDir
. $InstallPs1 -HermesHome $HomeDir -InstallDir (Join-Path $HomeDir 'install')

# Network installer rungs must fail fast: the host exe is an exit-1 stub, so
# `& $psHostExe ...` returns immediately instead of opening a shell on stdin.
function Get-PowerShellHostExe { return $HostExe }
# Install-Uv calls Get-Command with -CommandType; the mock must bind it.
function Get-Command {
    [CmdletBinding()]
    param([Parameter(Position=0)][string]$Name, [string]$CommandType, [Parameter(ValueFromRemainingArguments=$true)][object[]]$Rest)
    if ($Name -eq 'uv') {
        return [pscustomobject]@{ Source = $ValidCandidate }
    }
    return $null
}

$res = Install-Uv
if ($res) { exit 0 } else { exit 1 }
''',
        encoding="ascii",
    )

    proc = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(test_harness),
            "-InstallPs1",
            str(_INSTALL_PS1),
            "-HomeDir",
            str(hermes_home),
            "-ValidCandidate",
            str(valid_uv),
            "-HostExe",
            str(broken_uv),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    # Verify managed uv now runs and is valid
    managed_version = subprocess.run([str(broken_managed_uv), "--version"], capture_output=True, text=True)
    assert managed_version.returncode == 0
    assert "uv 0.5.0" in managed_version.stdout


@pytest.mark.windows_only
def test_windows_salvage_resolves_chocolatey_shim_to_real_executable(tmp_path: Path):
    """Chocolatey PATH shims (bin/uv.exe) must be resolved to the real binary under lib/uv/tools/uv.exe."""
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("powershell not found on PATH")

    choco_bin = tmp_path / "chocolatey" / "bin"
    choco_tools = tmp_path / "chocolatey" / "lib" / "uv" / "tools"
    choco_bin.mkdir(parents=True)
    choco_tools.mkdir(parents=True)

    choco_shim = choco_bin / "uv.exe"
    real_uv = choco_tools / "uv.exe"

    # Chocolatey shim stub fails outside of its directory
    shim_cs = tmp_path / "ChocoShim.cs"
    shim_cs.write_text(
        "using System;\n"
        "public class ChocoShim {\n"
        "  public static int Main(string[] args) {\n"
        "    Console.Error.WriteLine(\"Cannot find file at '..\\\\lib\\\\uv\\\\tools\\\\uv.exe'\");\n"
        "    return 1;\n"
        "  }\n"
        "}\n",
        encoding="ascii",
    )
    compile_shim = tmp_path / "compile_shim.ps1"
    compile_shim.write_text(
        f"Add-Type -Path '{shim_cs}' -OutputAssembly '{choco_shim}' -OutputType ConsoleApplication\n",
        encoding="ascii",
    )
    subprocess.run([powershell, "-ExecutionPolicy", "Bypass", "-File", str(compile_shim)], check=True)

    # Real uv binary in lib/uv/tools/uv.exe works
    real_cs = tmp_path / "RealUv.cs"
    real_cs.write_text(
        "using System;\n"
        "public class RealUv {\n"
        "  public static int Main(string[] args) {\n"
        "    Console.WriteLine(\"uv 0.11.7\");\n"
        "    return 0;\n"
        "  }\n"
        "}\n",
        encoding="ascii",
    )
    compile_real = tmp_path / "compile_real.ps1"
    compile_real.write_text(
        f"Add-Type -Path '{real_cs}' -OutputAssembly '{real_uv}' -OutputType ConsoleApplication\n",
        encoding="ascii",
    )
    subprocess.run([powershell, "-ExecutionPolicy", "Bypass", "-File", str(compile_real)], check=True)

    hermes_home = tmp_path / "hermes_home"
    test_harness = tmp_path / "test_choco_harness.ps1"
    test_harness.write_text(
        r'''param([string]$InstallPs1, [string]$HomeDir, [string]$ChocoShim)
$env:HERMES_HOME = $HomeDir
. $InstallPs1 -HermesHome $HomeDir -InstallDir (Join-Path $HomeDir 'install')

# Network installer rungs must fail fast: the exit-1 shim stands in for the
# host exe so `& $psHostExe ...` returns instead of opening a shell on stdin.
function Get-PowerShellHostExe { return $ChocoShim }
# Install-Uv calls Get-Command with -CommandType; the mock must bind it.
function Get-Command {
    [CmdletBinding()]
    param([Parameter(Position=0)][string]$Name, [string]$CommandType, [Parameter(ValueFromRemainingArguments=$true)][object[]]$Rest)
    if ($Name -eq 'uv') {
        return [pscustomobject]@{ Source = $ChocoShim }
    }
    return $null
}

$res = Install-Uv
if ($res) { exit 0 } else { exit 1 }
''',
        encoding="ascii",
    )

    proc = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(test_harness),
            "-InstallPs1",
            str(_INSTALL_PS1),
            "-HomeDir",
            str(hermes_home),
            "-ChocoShim",
            str(choco_shim),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    managed_uv = hermes_home / "bin" / "uv.exe"
    assert managed_uv.exists()
    managed_version = subprocess.run([str(managed_uv), "--version"], capture_output=True, text=True)
    assert managed_version.returncode == 0
    assert "uv 0.11.7" in managed_version.stdout
