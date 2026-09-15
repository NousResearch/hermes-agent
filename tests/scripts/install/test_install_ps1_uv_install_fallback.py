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

install.ps1 only runs on Windows, so these tests lock the contract at the
source-text level (same style as test_install_ps1_uv_powershell_host.py).
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_INSTALL_PS1 = Path(__file__).resolve().parents[3] / "scripts" / "install.ps1"

_GITHUB_INSTALLER_URL = (
    "https://github.com/astral-sh/uv/releases/latest/download/uv-installer.ps1"
)


@pytest.fixture(scope="module")
def source() -> str:
    return _INSTALL_PS1.read_text(encoding="utf-8")


def _install_uv_body(source: str) -> str:
    """Extract the text of Install-Uv up to the next top-level function."""
    start = source.index("function Install-Uv")
    tail = source[start + 1 :]
    match = re.search(r"^function ", tail, flags=re.MULTILINE)
    end = start + 1 + (match.start() if match else len(tail))
    return source[start:end]


def test_astral_installer_output_not_swallowed_by_out_null(source: str):
    """Regression pin for the suppression bug (#69366 / #69216).

    The astral invocation must not discard the installer's merged output
    stream; a failed download/AV block has to reach the user.
    """
    forbidden = 'irm https://astral.sh/uv/install.ps1 | iex" 2>&1 | Out-Null'
    assert forbidden not in source, (
        "Install-Uv pipes the astral uv installer's output straight to "
        "Out-Null again -- failures become the generic 'uv installed but "
        "not found' message. Capture the output (e.g. Tee-Object) instead."
    )


def test_astral_installer_output_is_captured(source: str):
    body = _install_uv_body(source)
    astral_lines = [
        ln
        for ln in body.splitlines()
        if "irm https://astral.sh/uv/install.ps1 | iex" in ln
    ]
    assert astral_lines, "astral uv installer invocation not found in Install-Uv"
    for ln in astral_lines:
        assert "Tee-Object" in ln, (
            "astral uv installer output must be captured (Tee-Object) so the "
            f"failure path can show it to the user, got: {ln.strip()!r}"
        )


def test_github_releases_fallback_installer_present(source: str):
    """Rung 2: the GitHub releases mirror of the installer must be tried."""
    body = _install_uv_body(source)
    assert _GITHUB_INSTALLER_URL in body, (
        "Install-Uv must fall back to the GitHub releases uv installer "
        f"({_GITHUB_INSTALLER_URL}) when astral.sh is blocked "
        "(corporate proxies, #69216)."
    )
    fallback_lines = [ln for ln in body.splitlines() if _GITHUB_INSTALLER_URL in ln and "irm " in ln]
    for ln in fallback_lines:
        stripped = ln.strip()
        assert stripped.startswith("& $"), (
            "GitHub fallback installer must be invoked via the resolved "
            f"PowerShell host variable (`& $...`), got: {stripped!r}"
        )
        assert "Tee-Object" in ln, (
            f"GitHub fallback installer output must be captured too: {stripped!r}"
        )


def test_existing_uv_salvage_rung_present(source: str):
    """Rung 3: probe PATH and the astral default dir, copy into managed bin."""
    body = _install_uv_body(source)
    assert "Get-Command uv" in body, (
        "Install-Uv must probe for an existing uv on PATH (Get-Command uv) "
        "before failing."
    )
    assert '".local\\bin\\uv.exe"' in body and "$env:USERPROFILE" in body, (
        "Install-Uv must probe the astral default install location "
        "(%USERPROFILE%\\.local\\bin\\uv.exe)."
    )
    assert "Copy-Item" in body and "$managedUv" in body, (
        "A salvaged uv.exe must be copied into the managed location "
        "($HermesHome\\bin\\uv.exe) so managed-first resolution holds."
    )


def test_every_managed_uv_candidate_requires_successful_version_check(source: str):
    """A copied Chocolatey shim must not become the managed uv binary."""
    body = _install_uv_body(source)
    assert "function Get-UsableUvVersion" in body
    assert body.count("Get-UsableUvVersion") >= 4, (
        "initial, salvaged, and final managed candidates must all be "
        "validated through the same exit-code-aware check"
    )
    assert "$exitCode -eq 0" in body
    assert "Existing managed uv" in body and "Remove-Item $managedUv" in body


def test_no_unvalidated_managed_uv_version_invocations_remain(source: str):
    body = _install_uv_body(source)
    assert not re.search(r"\$version\s*=\s*&\s*\$managedUv\s+--version", body)


def test_unusable_installer_output_falls_through_to_mirror(source: str):
    body = _install_uv_body(source)
    assert "astral.sh produced an unusable uv" in body
    assert "GitHub uv installer produced an unusable binary" in body
    assert body.count("Get-UsableUvVersion $managedUv") >= 3


def test_broken_path_candidate_does_not_hide_default_uv_candidate(source: str):
    body = _install_uv_body(source)
    assert "$salvageCandidates" in body
    assert "Select-Object -Unique" in body
    assert '".local\\bin\\uv.exe"' in body


def test_failure_path_keeps_manual_install_pointer_and_shows_output(source: str):
    body = _install_uv_body(source)
    assert "https://docs.astral.sh/uv/getting-started/installation/" in body, (
        "the manual-install pointer must survive in the failure path"
    )
    assert "$installerOutput" in body and "Select-Object -Last" in body, (
        "the failure path must print the tail of the captured installer "
        "output so the real error reaches the user"
    )


def test_resolve_executable_target_resolves_package_manager_shims(source: str):
    """Chocolatey and Scoop shims must resolve to the real underlying binary."""
    body = _install_uv_body(source)
    assert "function Resolve-ExecutableTarget" in body
    assert 'lib\\$name\\tools\\$name.exe' in body, (
        "Resolve-ExecutableTarget must inspect Chocolatey lib directory for the real tool"
    )
    assert ".shim" in body and 'path\\s*=\\s*' in body, (
        "Resolve-ExecutableTarget must inspect Scoop .shim files for the target path"
    )
    assert "LinkType" in body and "Target" in body, (
        "Resolve-ExecutableTarget must resolve symlinks and reparse points"
    )


def test_salvage_rung_prioritizes_resolved_shim_target(source: str):
    """The salvage candidates list must place the resolved real target before raw candidate."""
    body = _install_uv_body(source)
    assert "Resolve-ExecutableTarget $uvOnPath.Source" in body
    assert "$salvageCandidates += $resolved" in body
    assert "$salvageCandidates += $uvOnPath.Source" in body


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell required")
def test_windows_rerun_self_heals_when_managed_uv_is_broken(tmp_path: Path):
    """When $HermesHome/bin/uv.exe is already a broken shim, Install-Uv must purge it and self-heal."""
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("powershell not found on PATH")

    hermes_home = tmp_path / "hermes_home"
    bin_dir = hermes_home / "bin"
    bin_dir.mkdir(parents=True)
    broken_managed_uv = bin_dir / "uv.exe"

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
        f"Add-Type -Path '{source_cs}' -OutputAssembly '{broken_managed_uv}' -OutputType ConsoleApplication\n",
        encoding="ascii",
    )
    subprocess.run([powershell, "-ExecutionPolicy", "Bypass", "-File", str(compile_ps1)], check=True)
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
        r'''param([string]$InstallPs1, [string]$HomeDir, [string]$ValidCandidate)
$env:HERMES_HOME = $HomeDir
. $InstallPs1 -HermesHome $HomeDir -InstallDir (Join-Path $HomeDir 'install')

# Mock network installers by redefining Get-PowerShellHostExe to return a non-existent exe or dummy
function Get-PowerShellHostExe { return "cmd.exe" }
function Get-Command {
    [CmdletBinding()]
    param([Parameter(Position=0)][string]$Name, [Parameter(ValueFromRemainingArguments=$true)][object[]]$Rest)
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
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    # Verify managed uv now runs and is valid
    managed_version = subprocess.run([str(broken_managed_uv), "--version"], capture_output=True, text=True)
    assert managed_version.returncode == 0
    assert "uv 0.5.0" in managed_version.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell required")
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

# Mock network installers
function Get-PowerShellHostExe { return "cmd.exe" }
function Get-Command {
    [CmdletBinding()]
    param([Parameter(Position=0)][string]$Name, [Parameter(ValueFromRemainingArguments=$true)][object[]]$Rest)
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
