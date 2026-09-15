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


def test_salvaged_path_uv_is_retained_only_after_version_succeeds(source: str):
    """A PATH uv.exe becomes managed only after its copied binary runs."""
    body = _install_uv_body(source)
    copied = body.index("Copy-Item $existingUv $managedUv -Force")
    version_probe = body.index("$salvagedVersion = Test-ManagedUvBinary $managedUv", copied)
    exit_check = body.index("if (-not $salvagedVersion)", version_probe)
    assert copied < version_probe < exit_check, (
        "the copied PATH uv.exe must be retained only when the managed copy "
        "returns success from `uv --version`"
    )


def test_broken_chocolatey_relative_shim_is_removed_after_copy(source: str):
    """Chocolatey relative shims must be rejected once copied out of PATH."""
    body = _install_uv_body(source)
    rejected_copy = re.search(
        r"Copy-Item \$existingUv \$managedUv -Force\s+"
        r"\$salvagedVersion = Test-ManagedUvBinary \$managedUv\s+"
        r"if \(-not \$salvagedVersion\) \{\s+"
        r"Write-Info .+?\s+"
        r"Remove-Item \$managedUv -Force -ErrorAction SilentlyContinue\s+\}",
        body,
        flags=re.DOTALL,
    )
    assert rejected_copy, (
        "a copied Chocolatey relative shim that cannot run from Hermes' "
        "managed bin must be removed before fallback continues"
    )
    assert "Remove-Item $managedUv -Force -ErrorAction SilentlyContinue" in body, (
        "the rejected managed copy must be removed"
    )


def test_existing_managed_uv_is_validated_before_early_return(source: str):
    """Rerun must not trust a leftover broken shim at the managed path."""
    body = _install_uv_body(source)
    early = body.index("if (Test-Path $managedUv)")
    probe = body.index("$existingVersion = Test-ManagedUvBinary $managedUv", early)
    remove = body.index("Remove-Item $managedUv -Force -ErrorAction SilentlyContinue", probe)
    assert early < probe < remove, (
        "an existing managed uv.exe must be validated and removed on failure "
        "before Install-Uv continues the install ladder"
    )


def test_failure_path_keeps_manual_install_pointer_and_shows_output(source: str):
    body = _install_uv_body(source)
    assert "https://docs.astral.sh/uv/getting-started/installation/" in body, (
        "the manual-install pointer must survive in the failure path"
    )
    assert "$installerOutput" in body and "Select-Object -Last" in body, (
        "the failure path must print the tail of the captured installer "
        "output so the real error reaches the user"
    )


pytestmark_windows = pytest.mark.windows_only


def _compile_uv_candidate(
    powershell: str, path: Path, version_exit: int, *, relative_target: bool = False
) -> None:
    """Build a minimal executable whose ``uv --version`` outcome is controlled."""
    path.parent.mkdir(parents=True, exist_ok=True)
    source = path.with_suffix(".cs")
    version_result = (
        "return System.IO.File.Exists(System.IO.Path.GetFullPath("
        "System.IO.Path.Combine(System.IO.Path.GetDirectoryName("
        "Environment.GetCommandLineArgs()[0]), \"..\", \"lib\", \"uv.exe\"))) "
        "? 0 : 1;"
        if relative_target
        else f"return {version_exit};"
    )
    source.write_text(
        "using System;\n"
        "public static class CandidateUv {\n"
        "  public static int Main(string[] args) {\n"
        "    if (args.Length == 1 && args[0] == \"--version\") "
        "{ Console.WriteLine(\"uv 0.0.0-test\"); "
        f"{version_result} }}\n"
        f"    return {version_exit};\n"
        "  }\n"
        "}\n",
        encoding="ascii",
    )
    compile_script = path.with_name(f"compile-{path.stem}.ps1")
    compile_script.write_text(
        "param([string]$Source, [string]$Output)\n"
        "Add-Type -Path $Source -OutputAssembly $Output "
        "-OutputType ConsoleApplication\n",
        encoding="ascii",
    )
    subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(compile_script),
            "-Source",
            str(source),
            "-Output",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytestmark_windows
def test_salvaged_path_uv_must_run_before_managed_copy_is_kept(tmp_path: Path) -> None:
    """A copied PATH shim is kept only when its managed copy runs ``--version``.

    The harness dot-sources the real installer. It substitutes only the
    external network-installer host and PATH command lookup, leaving the
    candidate copy, verification, cleanup, and final resolution paths real.
    """
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")

    runner = tmp_path / "installer-runner.exe"
    broken = tmp_path / "chocolatey" / "bin" / "uv.exe"
    valid = tmp_path / "valid-uv.exe"
    _compile_uv_candidate(powershell, runner, 0)
    _compile_uv_candidate(powershell, broken, 1, relative_target=True)
    (broken.parent.parent / "lib").mkdir()
    (broken.parent.parent / "lib" / "uv.exe").write_bytes(b"target present in shim home")
    _compile_uv_candidate(powershell, valid, 0)

    harness = tmp_path / "run-install-uv.ps1"
    harness.write_text(
        r'''param(
    [string]$InstallPs1,
    [string]$HermesHome,
    [string]$Candidate,
    [string]$Runner
)

. $InstallPs1 -HermesHome $HermesHome -InstallDir (Join-Path $HermesHome 'install')

function Get-PowerShellHostExe { return $Runner }
function Get-Command {
    [CmdletBinding()]
    param(
        [Parameter(Position=0)] [string]$Name,
        [Parameter(ValueFromRemainingArguments=$true)] [object[]]$Rest
    )
    if ($Name -eq 'uv') { return [pscustomobject]@{ Source = $Candidate } }
    return $null
}

if (Install-Uv) { exit 0 }
exit 1
''',
        encoding="ascii",
    )

    def run_candidate(name: str, candidate: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(harness),
                "-InstallPs1",
                str(_INSTALL_PS1),
                "-HermesHome",
                str(tmp_path / name),
                "-Candidate",
                str(candidate),
                "-Runner",
                str(runner),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )

    rejected = run_candidate("broken-home", broken)
    rejected_managed = tmp_path / "broken-home" / "bin" / "uv.exe"
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert not rejected_managed.exists(), "a failing copied shim must be removed"

    accepted = run_candidate("valid-home", valid)
    accepted_managed = tmp_path / "valid-home" / "bin" / "uv.exe"
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert accepted_managed.exists(), "a valid PATH uv must remain managed"


@pytestmark_windows
def test_rerun_rejects_broken_shim_already_at_managed_path(tmp_path: Path) -> None:
    """A leftover broken managed uv.exe must be removed so the ladder can continue."""
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")

    runner = tmp_path / "installer-runner.exe"
    broken = tmp_path / "chocolatey" / "bin" / "uv.exe"
    valid = tmp_path / "valid-uv.exe"
    _compile_uv_candidate(powershell, runner, 0)
    _compile_uv_candidate(powershell, broken, 1, relative_target=True)
    (broken.parent.parent / "lib").mkdir()
    (broken.parent.parent / "lib" / "uv.exe").write_bytes(b"target present in shim home")
    _compile_uv_candidate(powershell, valid, 0)

    hermes_home = tmp_path / "rerun-home"
    managed = hermes_home / "bin" / "uv.exe"
    managed.parent.mkdir(parents=True)
    shutil.copy2(broken, managed)

    harness = tmp_path / "run-install-uv-rerun.ps1"
    harness.write_text(
        r'''param(
    [string]$InstallPs1,
    [string]$HermesHome,
    [string]$Candidate,
    [string]$Runner
)

. $InstallPs1 -HermesHome $HermesHome -InstallDir (Join-Path $HermesHome 'install')

function Get-PowerShellHostExe { return $Runner }
function Get-Command {
    [CmdletBinding()]
    param(
        [Parameter(Position=0)] [string]$Name,
        [Parameter(ValueFromRemainingArguments=$true)] [object[]]$Rest
    )
    if ($Name -eq 'uv') { return [pscustomobject]@{ Source = $Candidate } }
    return $null
}

if (Install-Uv) { exit 0 }
exit 1
''',
        encoding="ascii",
    )

    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(harness),
            "-InstallPs1",
            str(_INSTALL_PS1),
            "-HermesHome",
            str(hermes_home),
            "-Candidate",
            str(valid),
            "-Runner",
            str(runner),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert managed.exists(), "rerun should replace the broken managed shim"
    assert managed.read_bytes() == valid.read_bytes()
