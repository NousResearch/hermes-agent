# Behavioral tests for install.ps1 uv acceptance (issue #110350).
#
# A relocated package-manager shim (Chocolatey's uv.exe launcher resolves
# its real binary RELATIVE to its own location) must NEVER be accepted as
# the managed uv, and a working uv must still be accepted on every path.
#
# Design (dot-sourced without running the installer's entry point):
#   SECTION A -- REAL Test-UvBinary primitive, where a real executable is
#     cheap to fabricate: on POSIX a sh script (accept exit-0 / reject
#     exit-1); on Windows a non-executable text file (reject), since no
#     offline uv.exe that prints a version is available in CI.
#   SECTION B -- REAL Resolve-UvShimTarget: a native Chocolatey layout
#     (<root>\chocolatey\bin\uv.exe shim -> <root>\chocolatey\lib\uv\tools\
#     uv.exe target) is recognized cross-platform because the resolver keys
#     off the 'chocolatey' path segment.
#   FLOW F1-F6 -- the Install-Uv / Resolve-UvCmd decision flow, driven by a
#     content-marker stand-in for the native "does this uv run" check plus
#     stubbed installer/PATH boundaries, so the flow is deterministic and
#     offline on every platform.  'uv-healthy' marker content = accepted;
#     broken-shim text = rejected.  The real primitives above already prove
#     the marker->accept mapping mirrors the real exit-code contract.
#
# Scenarios:
#   F1  pre-existing managed uv that works        -> accepted, no installer
#   F2  pre-existing managed uv that is broken    -> removed, re-provisioned
#   F3  salvage a choco shim on PATH              -> target resolved + copied
#   F4  salvage a candidate that cannot run       -> stage fails honestly
#   F5  Resolve-UvCmd, broken managed leftover     -> removed, PATH uv takes over
#   F6  Resolve-UvCmd, healthy managed uv         -> used directly
#
# Run:
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/tests/test-install-ps1-uv-shim.ps1

$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot 'scripts\install.ps1'
if (-not (Test-Path $installScript)) { throw "Could not locate install.ps1 at $installScript" }

$testRoot = Join-Path ([IO.Path]::GetTempPath()) ("hermes-uv-shim-test-" + [Guid]::NewGuid().ToString('N'))
$HermesHome = Join-Path $testRoot 'home'
$InstallDir = Join-Path $testRoot 'missing-checkout'
New-Item -ItemType Directory -Force -Path $HermesHome | Out-Null

# install.ps1 is the Windows installer; $env:USERPROFILE is always set on
# Windows (the salvage rung probes $USERPROFILE\.local\bin\uv.exe). Set it
# on non-Windows test hosts so the flow runs identically.
if (-not $env:USERPROFILE) { $env:USERPROFILE = $testRoot }

# The managed uv path EXACTLY as the installer computes it:
#   $managedUv = Join-Path $HermesHome "bin\uv.exe"
# On Windows those are native separators; on POSIX the whole "bin\uv.exe"
# is a single literal leaf under $HermesHome. Mirror the expression so the
# test's path bookkeeping matches the installer's.
$managedUv = Join-Path $HermesHome "bin\uv.exe"

# Dot-source loads the installer's REAL functions.
. $installScript -HermesHome $HermesHome -InstallDir $InstallDir

$script:Failures = 0
function Assert-Equal {
    param($Expected, $Actual, [string]$Label)
    if ($Expected -ceq $Actual) {
        Write-Host "PASS: $Label"
    } else {
        Write-Host "FAIL: $Label"
        Write-Host "  expected: [$Expected]"
        Write-Host "  actual:   [$Actual]"
        $script:Failures++
    }
}
function Assert-True {
    param($Actual, [string]$Label)
    if ($Actual) { Write-Host "PASS: $Label" }
    else {
        Write-Host "FAIL: $Label"
        Write-Host "  expected: True"
        Write-Host "  actual:   $Actual"
        $script:Failures++
    }
}

# =========================================================================
# SECTION A: REAL Test-UvBinary primitive
# =========================================================================
Write-Host ''
Write-Host '-- A. real Test-UvBinary primitive --'
# Platform split that works on BOTH PowerShell 5.1 (no $IsWindows auto var)
# and pwsh 7+: use $PSVersionTable.Platform ('Win32NT' vs 'Unix').
$onWinHost = ($PSVersionTable.Platform -eq 'Win32NT')

if (-not $onWinHost) {
    $okExe = Join-Path $testRoot 'probe-ok.sh'
    [System.IO.File]::WriteAllText($okExe, "#!/bin/sh`necho 'uv 0.99.0'`nexit 0`n")
    chmod +x $okExe
    $badExe = Join-Path $testRoot 'probe-bad.sh'
    [System.IO.File]::WriteAllText($badExe, "#!/bin/sh`necho 'Cannot find file at ..'`nexit 1`n")
    chmod +x $badExe
    Assert-True (Test-UvBinary $okExe) 'real Test-UvBinary accepts a working exe (exit 0, uv version stdout)'
    Assert-True (-not (Test-UvBinary $badExe)) 'real Test-UvBinary rejects an exe that exits nonzero'
    Assert-True (-not (Test-UvBinary (Join-Path $testRoot 'missing-exe'))) 'real Test-UvBinary rejects a missing path'
} else {
    $text = Join-Path $testRoot 'probe-text.txt'
    "Cannot find file at '..\lib\uv\tools\uv.exe'" | Set-Content -LiteralPath $text -Encoding Ascii
    Assert-True (-not (Test-UvBinary $text)) 'real Test-UvBinary rejects a non-executable (Windows)'
    Write-Host 'INFO: Windows accept path (real uv.exe) is covered by the flow + windows e2e'
}

# =========================================================================
# SECTION B: REAL Resolve-UvShimTarget (shim -> standalone target)
# =========================================================================
Write-Host ''
Write-Host '-- B. real Resolve-UvShimTarget (shim -> standalone target) --'
function New-ChocoLayout {
    param([string]$Root)
    $base = [IO.Path]::Combine($Root, 'chocolatey')
    $shimPath = [IO.Path]::Combine($base, 'bin', 'uv.exe')
    $target = [IO.Path]::Combine($base, 'lib', 'uv', 'tools', 'uv.exe')
    New-Item -ItemType Directory -Force -Path (Split-Path $shimPath) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    'shim-text' | Set-Content -LiteralPath $shimPath -Encoding Ascii
    'target-text' | Set-Content -LiteralPath $target -Encoding Ascii
    return [pscustomobject]@{ ShimPath = $shimPath; Target = $target }
}
$chocoB = New-ChocoLayout (Join-Path $testRoot 'res-solve')
Assert-Equal $chocoB.Target (Resolve-UvShimTarget $chocoB.ShimPath) 'choco bin\ shim resolves to its standalone target'
Assert-Equal $chocoB.Target (Resolve-UvShimTarget $chocoB.Target) 'a standalone target passes through unchanged'
$plain = Join-Path $testRoot 'plain-uv'
New-Item -ItemType Directory -Force -Path (Split-Path $plain) | Out-Null
'x' | Set-Content -LiteralPath $plain -Encoding Ascii
Assert-Equal $plain (Resolve-UvShimTarget $plain) 'a non-choco standalone path passes through unchanged'

# =========================================================================
# FLOW: install the deterministic stand-ins for the native/installer/PATH
# boundaries (they take precedence over the dot-sourced originals from here
# on).  Section A above already ran against the REAL primitives.
# =========================================================================
# Content-marker stand-in for the native "does this uv run" check: uv marker
# files reading 'uv-healthy' are accepted; anything else is rejected.
function Test-UvBinary {
    param([string]$UvPath)
    if ([string]::IsNullOrWhiteSpace($UvPath) -or -not (Test-Path -LiteralPath $UvPath)) { return $false }
    try {
        $content = Get-Content -LiteralPath $UvPath -Raw -ErrorAction Stop
    } catch { return $false }
    return ($content -match 'uv-healthy')
}
# Write a uv marker (creates the parent dir when it is a real directory).
function New-UvMarker {
    param([string]$Path, [ValidateSet('healthy', 'broken')][string]$Kind)
    $parent = Split-Path $Path -Parent
    if ($parent -and $parent -ne '.' -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    $text = if ($Kind -eq 'healthy') { "uv-healthy marker (a working uv)" }
    else { "Cannot find file at '..\lib\uv\tools\uv.exe' (broken copy)" }
    [System.IO.File]::WriteAllText($Path, $text)
}
# The two official installer spawns (astral.sh + GitHub mirror): replaced by
# a deterministic fake. 'installed' drops a healthy managed uv; 'absent'
# simulates a fully blocked network. No real network is touched.
$script:InstallerAttempts = 0
$script:InstallRung = 'absent'
function Invoke-UvInstallerSpawns {
    param([string]$UvInstallDir)
    $script:InstallerAttempts++
    if ($script:InstallRung -eq 'installed') {
        New-UvMarker $UvInstallDir 'healthy'
    }
    @("--- uv installer source: fake (no network in test) ---")
}
# PATH discovery (salvage rung + Resolve-UvCmd fall-through): return the
# controlled Source; -ErrorAction stays the common parameter.
$script:SalvageSource = $null
function Get-Command {
    [CmdletBinding()]
    param([string]$Name, [string]$CommandType, [string]$CommandTypes)
    if ($Name -eq 'uv' -and $script:SalvageSource) {
        return [PSCustomObject]@{ Source = $script:SalvageSource; CommandName = 'uv' }
    }
    return $null
}
function Reset-Scenario {
    $script:UvCmd = $null
    $script:InstallerAttempts = 0
    $script:SalvageSource = $null
    $script:InstallRung = 'absent'
    Remove-Item -LiteralPath $managedUv -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $HermesHome 'bin') -Recurse -Force -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------------------
# F1: pre-existing managed uv that works -> accepted, no reinstall.
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '-- F1: pre-existing managed uv that works --'
Reset-Scenario
New-UvMarker $managedUv 'healthy'
$r = Install-Uv
Assert-Equal $true $r 'working managed uv is accepted'
Assert-Equal 0 $script:InstallerAttempts 'no installer spawn for a working managed uv'
Assert-Equal $managedUv $script:UvCmd 'UvCmd points at the managed copy'

# ---------------------------------------------------------------------------
# F2: pre-existing managed uv that is broken (the #110350 leftover) ->
# removed and re-provisioned by the installer rungs.
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '-- F2: pre-existing managed uv that is broken --'
Reset-Scenario
New-UvMarker $managedUv 'broken'
$script:InstallRung = 'installed'
$r = Install-Uv
Assert-Equal $true $r 'broken managed uv is replaced by a fresh managed install'
Assert-Equal 1 $script:InstallerAttempts 'broken managed uv triggers re-provisioning'
Assert-Equal $managedUv $script:UvCmd 'UvCmd re-pointed at the reinstalled managed uv'
Assert-True ((Get-Content -LiteralPath $managedUv -Raw -ErrorAction SilentlyContinue) -match 'uv-healthy') 'the managed copy now holds the healthy binary'

# ---------------------------------------------------------------------------
# F3: salvage rung -- a Chocolatey-style shim on PATH whose standalone
# target is resolvable -> the TARGET (not the shim) is copied and accepted.
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '-- F3: salvage resolves a choco shim on PATH to its target --'
Reset-Scenario
$script:InstallRung = 'absent'   # official installer blocked; only salvage works
$chocoF = New-ChocoLayout (Join-Path $testRoot 'res-salvage')
# Mark the standalone target healthy; the shim itself stays "broken".
[System.IO.File]::WriteAllText($chocoF.Target, "uv-healthy marker (a working uv)")
$script:SalvageSource = $chocoF.ShimPath
$r = Install-Uv
Assert-Equal $true $r 'salvage of a choco shim succeeds via its resolved target'
Assert-Equal $chocoF.Target (Resolve-UvShimTarget $chocoF.ShimPath) 'the shim was resolved to its standalone target'
Assert-Equal $managedUv $script:UvCmd 'UvCmd points at the salvaged target'
Assert-True ((Get-Content -LiteralPath $managedUv -Raw -ErrorAction SilentlyContinue) -match 'uv-healthy') 'the copied managed uv is the healthy target'
$script:SalvageSource = $null

# ---------------------------------------------------------------------------
# F4: salvage rung -- a standalone candidate that cannot run -> the stage
# fails honestly; no broken copy is left behind as "the" managed uv.
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '-- F4: salvage of an unrunnable candidate fails the stage --'
Reset-Scenario
$script:InstallRung = 'absent'
$brokenStandalone = Join-Path $testRoot 'broken-uv'
New-UvMarker $brokenStandalone 'broken'
$script:SalvageSource = $brokenStandalone   # standalone: nothing left to resolve to
$r = Install-Uv
Assert-Equal $false $r 'non-running salvage candidate fails Install-Uv'
Assert-Equal $null $script:UvCmd 'UvCmd stays unset when salvage fails'
Assert-Equal $false (Test-Path $managedUv) 'failed salvage leaves no broken managed copy'
$script:SalvageSource = $null

# ---------------------------------------------------------------------------
# F5: Resolve-UvCmd self-heals a broken managed leftover (the cross-process
# stage-driver path that would otherwise keep a dead shim), falling
# through to a PATH uv.
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '-- F5: Resolve-UvCmd removes a broken managed uv, PATH takes over --'
Reset-Scenario
New-UvMarker $managedUv 'broken'
$chocoG = New-ChocoLayout (Join-Path $testRoot 'res-heal')
$script:SalvageSource = $chocoG.Target
Resolve-UvCmd
Assert-Equal 'uv' $script:UvCmd 'broken managed uv removed; PATH uv takes over'
Assert-Equal $false (Test-Path $managedUv) 'broken managed uv removed on the ground'
$script:SalvageSource = $null

# ---------------------------------------------------------------------------
# F6: Resolve-UvCmd accepts a healthy managed uv without touching it.
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '-- F6: Resolve-UvCmd uses a healthy managed uv directly --'
Reset-Scenario
New-UvMarker $managedUv 'healthy'
Resolve-UvCmd
Assert-Equal $managedUv $script:UvCmd 'healthy managed uv is used directly'

Write-Host ''
if ($script:Failures -gt 0) {
    Write-Host "$script:Failures assertion(s) failed"
    if (Test-Path $testRoot) { Remove-Item -LiteralPath $testRoot -Recurse -Force }
    exit 1
}
Write-Host 'all assertions passed'

if (Test-Path $testRoot) {
    Remove-Item -LiteralPath $testRoot -Recurse -Force
}
