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

Scope limit, because it decides which mutations this test can catch.  What is
pinned is the *observable* three-state contract -- a list, a non-$null empty
array, $null -- not the source text of the return statements.  Under
PowerShell's return semantics ``return @()`` and ``return $null`` are
indistinguishable to the caller: both enumerate to nothing and arrive as
$null.  So rewriting the catch's ``return $null`` as ``return @()`` is not a
detectable regression and is deliberately not a mutation this test claims to
catch.  The two directions that do move observable state, and that the
mutation control exercises, are the success path losing its leading comma
(``,@(...)`` -> ``@(...)``: the empty sweep collapses into the failure
sentinel and every swap refuses) and the catch returning a real empty array
(``return ,@()``: the fail-OPEN direction, a broken sweep reported as a free
tree, which the ``caller_allows_broken`` assertion below refuses).

The second test here pins the other half of the same guard, which a passing
return contract alone does not buy: the gate has to actually run before each
of the three commands in ``Stage-Repository`` that rename or delete the
install tree.  A gate that returns all three states correctly but is not
called -- or is called after the move -- is the incident all over again, so
the call sites are asserted statically and positionally.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

# The retired linux_only / macos_only / windows_only marks are rejected at
# collection time; platforms(...) is the current spelling of the same gate.
pytestmark = pytest.mark.platforms("windows")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INSTALL_PS1 = _REPO_ROOT / "scripts" / "install.ps1"

# Lifts the functions under test out of install.ps1 via the AST (dot sourcing
# install.ps1 would run the installer) and drives both sweep outcomes.
# Win32_Process is shadowed by a same-named function -- a function beats a
# cmdlet in PowerShell's command resolution -- so the empty sweep and the
# exploding sweep are both deterministic, and neither reads the host's real
# process table.
#
# The extraction set is the transitive closure of what the tested functions
# reach, so a lifted function runs to completion instead of dying on an
# unresolved name.  That matters most in the failure path: the catch block's
# first statement is a Write-Warn call, and if Write-Warn is not lifted the
# catch dies on CommandNotFoundException *before* reaching its `return $null`
# -- the function then yields $null for the wrong reason, and the fail-closed
# assertion below passes without ever having exercised the sentinel it claims
# to pin.  Real lifted definitions, not stubs: these are self-contained in
# install.ps1 (no script state, no external commands) and lifting them is what
# keeps the harness honest about which line produced the result.
#   Get-HermesTreeHolder -> ConvertTo-LongPath, ConvertTo-TreePathPattern, Write-Warn
#   ConvertTo-LongPath   -> Expand-ShortProfileRoot (8.3 fallback)
#   Expand-ShortProfileRoot -> Get-LongProfileRoot
_HARNESS = r'''
param(
    [Parameter(Mandatory = $true)][string]$InstallPs1,
    [Parameter(Mandatory = $true)][string]$InstallDir
)

$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($InstallPs1, [ref]$tokens, [ref]$errors)
if ($errors.Count -gt 0) { throw "install.ps1 does not parse: $($errors[0].Message)" }

$wanted = @(
    # The installer's own line style; Get-HermesTreeHolder's catch calls
    # Write-Warn as its first statement, so the sentinel below is only reached
    # when these are defined.
    'Log', 'Write-Ok', 'Write-Warn', 'Write-Err',
    # ConvertTo-LongPath's last resolver, reached only on a host whose profile
    # root is 8.3-aliased (a short temp/profile path); lifted so the harness
    # resolves the same way the installer does on those machines.
    'Get-LongProfileRoot', 'Expand-ShortProfileRoot',
    # The functions actually under test.
    'ConvertTo-LongPath', 'ConvertTo-TreePathPattern', 'Get-HermesTreeHolder'
)
$defs = @($ast.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $wanted -contains $node.Name
}, $true))
if ($defs.Count -ne $wanted.Count) {
    throw "expected $($wanted.Count) functions, found $($defs.Count): $(@($defs | ForEach-Object { $_.Name }) -join ', ')"
}

Invoke-Expression (($defs | ForEach-Object { $_.Extent.Text }) -join "`n`n")

function Get-CimInstance { }

$clean = Get-HermesTreeHolder -InstallDir $InstallDir

function Get-CimInstance { throw 'simulated WMI failure' }

$broken = Get-HermesTreeHolder -InstallDir $InstallDir

# The caller's gate, mirrored from the tree-replacement block that runs before
# the rename: $null means the sweep failed, an empty array means the tree is
# free.  Reported for the failing sweep too, not just as a $null check: the
# fail-OPEN mutation (a real empty array out of the catch) leaves
# broken_is_null false *and* lets the caller proceed, and the allowance is the
# consequence worth naming.
$cleanFailed = $null -eq $clean
$cleanHolders = @($clean | Where-Object { $null -ne $_ })
$brokenFailed = $null -eq $broken
$brokenHolders = @($broken | Where-Object { $null -ne $_ })

$result = [ordered]@{
    clean_is_null         = $cleanFailed
    clean_count           = $cleanHolders.Count
    caller_allows_clean   = -not ($cleanFailed -or $cleanHolders.Count -gt 0)
    broken_is_null        = $brokenFailed
    broken_count          = $brokenHolders.Count
    caller_refuses_broken = $brokenFailed
    caller_allows_broken  = -not ($brokenFailed -or $brokenHolders.Count -gt 0)
}
Write-Output ('HERMES_HOLDER_CONTRACT ' + ($result | ConvertTo-Json -Compress))
'''

# Static half of the same guard: the three rename/delete commands inside
# Stage-Repository must each be preceded, in their own statement block, by the
# Assert-NoTreeHolders call.  Nothing here executes the installer -- the file
# is parsed, the commands are located by name and by argument text, and the
# positional relation is read off the AST.  Mutating any of the three call
# sites (dropping one, moving one after the command it guards) turns one of
# the booleans below false.
_GATE_ORDER_HARNESS = r'''
param(
    [Parameter(Mandatory = $true)][string]$InstallPs1
)

$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($InstallPs1, [ref]$tokens, [ref]$errors)
if ($errors.Count -gt 0) { throw "install.ps1 does not parse: $($errors[0].Message)" }

function Get-CommandNameOf {
    param($Command)
    return $Command.GetCommandName()
}

# A space-separated parameter is two CommandElements (the parameter, then its
# argument), so pair them positionally rather than reading
# CommandParameterAst.Argument, which is only filled in for `-Name:value`.
function Get-CommandArguments {
    param($Command)
    $found = @{}
    $elements = $Command.CommandElements
    for ($i = 0; $i -lt $elements.Count; $i++) {
        $element = $elements[$i]
        if ($element -is [System.Management.Automation.Language.CommandParameterAst]) {
            $value = $null
            $next = $null
            if ($i + 1 -lt $elements.Count) { $next = $elements[$i + 1] }
            if ($next -and -not ($next -is [System.Management.Automation.Language.CommandParameterAst])) {
                $value = $next.Extent.Text
            }
            $found[$element.ParameterName] = $value
        }
    }
    return $found
}

function Select-TargetCommand {
    param($Commands, [string]$Name, $Expected)
    $matched = @()
    foreach ($command in $Commands) {
        if ((Get-CommandNameOf $command) -ne $Name) { continue }
        $arguments = Get-CommandArguments $command
        $same = $true
        foreach ($key in $Expected.Keys) {
            if ($arguments[$key] -ne $Expected[$key]) { $same = $false; break }
        }
        if ($same) { $matched += $command }
    }
    return ,@($matched)
}

# True when the statement immediately before $Command in its own
# StatementBlockAst is an Assert-NoTreeHolders command.
function Test-GateImmediatelyBefore {
    param($Command)
    $node = $Command
    while ($node -and $node.Parent -and -not ($node.Parent -is [System.Management.Automation.Language.StatementBlockAst])) {
        $node = $node.Parent
    }
    if (-not $node -or -not $node.Parent) { return $false }
    $statements = $node.Parent.Statements
    for ($i = 1; $i -lt $statements.Count; $i++) {
        if ($statements[$i] -ne $node) { continue }
        $previous = $statements[$i - 1]
        $previousCommand = $null
        if ($previous -is [System.Management.Automation.Language.PipelineAst]) {
            if ($previous.PipelineElements.Count -gt 0) { $previousCommand = $previous.PipelineElements[0] }
        } elseif ($previous -is [System.Management.Automation.Language.CommandAst]) {
            $previousCommand = $previous
        }
        if ($previousCommand -is [System.Management.Automation.Language.CommandAst]) {
            return (Get-CommandNameOf $previousCommand) -eq 'Assert-NoTreeHolders'
        }
        return $false
    }
    return $false
}

$stage = @($ast.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Stage-Repository'
}, $true))
if ($stage.Count -ne 1) { throw "expected exactly 1 Stage-Repository definition, found $($stage.Count)" }

$commands = @($stage[0].Body.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.CommandAst]
}, $true))

# The three tree-replacing commands, identified by name plus argument text:
#   R1  Move-Item -LiteralPath $InstallDir -Destination $broken
#   R2  Remove-Item -LiteralPath $InstallDir -Force
#   R3  Move-Item -LiteralPath $tree -Destination $InstallDir
$r1 = Select-TargetCommand $commands 'Move-Item' @{ Destination = '$broken' }
$r2 = Select-TargetCommand $commands 'Remove-Item' @{ LiteralPath = '$InstallDir' }
$r3 = Select-TargetCommand $commands 'Move-Item' @{ LiteralPath = '$tree'; Destination = '$InstallDir' }

# Ambiguity is a failure, not a skip: a second match means the locator is no
# longer pinning what this test claims it pins.
if ($r1.Count -ne 1) { throw 'expected exactly 1 Move-Item -Destination $broken, found ' + $r1.Count }
if ($r2.Count -ne 1) { throw 'expected exactly 1 Remove-Item -LiteralPath $InstallDir, found ' + $r2.Count }
if ($r3.Count -ne 1) { throw 'expected exactly 1 Move-Item -LiteralPath $tree -Destination $InstallDir, found ' + $r3.Count }

$gates = @($commands | Where-Object { (Get-CommandNameOf $_) -eq 'Assert-NoTreeHolders' })

$result = [ordered]@{
    gate_count                 = $gates.Count
    r1_present                 = ($r1.Count -eq 1)
    r2_present                 = ($r2.Count -eq 1)
    r3_present                 = ($r3.Count -eq 1)
    r1_gate_immediately_before = (Test-GateImmediatelyBefore -Command $r1[0])
    r2_gate_immediately_before = (Test-GateImmediatelyBefore -Command $r2[0])
    r3_gate_immediately_before = (Test-GateImmediatelyBefore -Command $r3[0])
}
Write-Output ('HERMES_HOLDER_GATE_ORDER ' + ($result | ConvertTo-Json -Compress))
'''


def _run_harness(
    harness_text: str, marker: str, source: Path, tmp_path: Path, *, sweep: bool = True
) -> dict:
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")

    harness = tmp_path / "tree_holder_gate.ps1"
    harness.write_text(harness_text, encoding="ascii")

    args = [
        powershell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(harness),
        "-InstallPs1",
        str(source),
    ]
    if sweep:
        # Only the return-contract harness sweeps processes; the static
        # gate-order harness takes the install.ps1 path and nothing else.
        args += ["-InstallDir", str(tmp_path / "not-an-install-tree")]

    run = subprocess.run(
        args,
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
        line for line in run.stdout.splitlines() if line.startswith(marker + " ")
    ]
    assert marked, run.stdout + run.stderr
    return json.loads(marked[-1].split(" ", 1)[1])


def _source_under_test() -> Path:
    # HERMES_INSTALL_PS1_UNDER_TEST points the same assertions at another
    # revision of install.ps1 (used to prove these tests catch the regression).
    return Path(os.environ.get("HERMES_INSTALL_PS1_UNDER_TEST") or _INSTALL_PS1)


def _run_contract(source: Path, tmp_path: Path) -> dict:
    return _run_harness(_HARNESS, "HERMES_HOLDER_CONTRACT", source, tmp_path)


def test_empty_sweep_allows_and_failed_sweep_refuses(tmp_path: Path) -> None:
    result = _run_contract(_source_under_test(), tmp_path)

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
    # The fail-OPEN direction, stated as its own consequence: whatever the
    # catch hands back, a sweep that threw must never be readable by the
    # caller as "the tree is free".
    assert result["caller_allows_broken"] is False, (
        "a failed sweep reported as a free tree is the fail-open direction "
        "this guard exists to close"
    )


def test_holder_gate_runs_immediately_before_every_tree_swap(tmp_path: Path) -> None:
    result = _run_harness(
        _GATE_ORDER_HARNESS,
        "HERMES_HOLDER_GATE_ORDER",
        _source_under_test(),
        tmp_path,
        sweep=False,
    )

    assert result["gate_count"] == 3, (
        "Stage-Repository holds exactly three rename/delete commands (broken "
        ".git move-aside, stale directory delete, staging tree publish) and "
        "each must be gated; a missing call is the rename-under-a-live-app "
        "the gate exists to stop"
    )

    for key in ("r1_present", "r2_present", "r3_present"):
        assert result[key] is True, f"{key}: the gated command was not found"

    for key, what in (
        ("r1_gate_immediately_before", "the broken .git move-aside"),
        ("r2_gate_immediately_before", "the stale directory delete"),
        ("r3_gate_immediately_before", "the staging tree publish"),
    ):
        assert result[key] is True, (
            f"{what} must be preceded, in its own statement block, by "
            "Assert-NoTreeHolders -InstallDir $InstallDir: a gate that runs "
            "after the rename guards nothing"
        )
