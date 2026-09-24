"""Return-value contract of the install.ps1 tree-swap preflight.

``Get-HermesTreeHolder`` answers one question for the caller that decides
whether the install tree may be renamed: is anything running out of it?  It
answers with three distinguishable states, and the caller branches on all
three:

* a list of holders      -> refuse (the App / backend is live in the tree)
* an EMPTY array         -> allow  (sweep succeeded, tree is free)
* ``$null``              -> refuse (the sweep itself failed; nothing proven)

The empty case and the ``$null`` case are one PowerShell footgun apart: a
returned array is enumerated into the pipeline, and an empty array enumerates
to *nothing*, so ``return @()`` silently degrades "sweep succeeded, no
holders" into the failure sentinel -- every tree swap then refuses, and the
refusal path looks perfectly healthy while it does.  This test pins the
contract, not the implementation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.windows_only

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INSTALL_PS1 = _REPO_ROOT / "scripts" / "install.ps1"

# Lifts the three functions under test out of install.ps1 via the AST (dot
# sourcing install.ps1 would run the installer) and drives both sweep
# outcomes.  Win32_Process is shadowed by a same-named function -- a function
# beats a cmdlet in PowerShell's command resolution -- so the empty sweep and
# the exploding sweep are both deterministic, and neither reads the host's
# real process table.
_HARNESS = r'''
param(
    [Parameter(Mandatory = $true)][string]$InstallPs1,
    [Parameter(Mandatory = $true)][string]$InstallDir
)

$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($InstallPs1, [ref]$tokens, [ref]$errors)
if ($errors.Count -gt 0) { throw "install.ps1 does not parse: $($errors[0].Message)" }

$wanted = @('ConvertTo-LongPath', 'ConvertTo-TreePathPattern', 'Get-HermesTreeHolder')
$defs = @($ast.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $wanted -contains $node.Name
}, $true))
if ($defs.Count -ne $wanted.Count) { throw "expected $($wanted.Count) functions, found $($defs.Count)" }

Invoke-Expression (($defs | ForEach-Object { $_.Extent.Text }) -join "`n`n")

function Get-CimInstance { }

$clean = Get-HermesTreeHolder -InstallDir $InstallDir

function Get-CimInstance { throw 'simulated WMI failure' }

$broken = Get-HermesTreeHolder -InstallDir $InstallDir

# The caller's gate, mirrored from the tree-replacement block that runs before
# the rename: $null means the sweep failed, an empty array means the tree is
# free.
$cleanFailed = $null -eq $clean
$cleanHolders = @($clean | Where-Object { $null -ne $_ })
$brokenFailed = $null -eq $broken

$result = [ordered]@{
    clean_is_null         = $cleanFailed
    clean_count           = $cleanHolders.Count
    caller_allows_clean   = -not ($cleanFailed -or $cleanHolders.Count -gt 0)
    broken_is_null        = $brokenFailed
    caller_refuses_broken = $brokenFailed
}
Write-Output ('HERMES_HOLDER_CONTRACT ' + ($result | ConvertTo-Json -Compress))
'''


def _run_contract(source: Path, tmp_path: Path) -> dict:
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")

    harness = tmp_path / "tree_holder_contract.ps1"
    harness.write_text(_HARNESS, encoding="ascii")

    run = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(harness),
            "-InstallPs1",
            str(source),
            "-InstallDir",
            str(tmp_path / "not-an-install-tree"),
        ],
        capture_output=True,
        text=True,
        # Windows PowerShell 5.1 writes the console codepage, not UTF-8; the
        # marker line this test reads is ASCII either way, and errors="replace"
        # keeps a stray byte from aborting the reader thread mid-stream.
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=120,
    )

    marked = [
        line for line in run.stdout.splitlines() if line.startswith("HERMES_HOLDER_CONTRACT ")
    ]
    assert marked, run.stdout + run.stderr
    return json.loads(marked[-1].split(" ", 1)[1])


def test_empty_sweep_allows_and_failed_sweep_refuses(tmp_path: Path) -> None:
    # HERMES_INSTALL_PS1_UNDER_TEST points the same assertions at another
    # revision of install.ps1 (used to prove this test catches the regression).
    source = Path(os.environ.get("HERMES_INSTALL_PS1_UNDER_TEST") or _INSTALL_PS1)
    result = _run_contract(source, tmp_path)

    assert result["clean_is_null"] is False, (
        "a successful sweep that finds no holders must not return $null: "
        "$null is the failed-sweep sentinel, so the caller refuses every "
        "tree swap"
    )
    assert result["clean_count"] == 0
    assert result["caller_allows_clean"] is True

    assert result["broken_is_null"] is True, (
        "a sweep that throws must keep returning the $null sentinel, or the "
        "caller proceeds on a probe that proved nothing"
    )
    assert result["caller_refuses_broken"] is True
