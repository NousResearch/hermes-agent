# Tests for install.ps1's install-directory selection.
#
# Run from a PowerShell prompt:
#
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/tests/test-install-ps1-install-dir-precedence.ps1
#
# Background: on Windows the code checkout + venv (~1 GB) and the user's
# data (config.yaml, .env, sessions, skills, memories) used to be inseparable.
# The only way to move the program off a small C: drive was to move
# HERMES_HOME -- which moves the data with it. The installer also ignored
# $HERMES_INSTALL_DIR, which install.sh has honored since it grew --dir, so a
# user installing through the documented one-liner (`irm ... | iex`) had no way
# to choose a directory at all: Invoke-Expression passes no arguments, so
# -InstallDir/-HermesHome are unreachable from that form.
#
# What this asserts (the selection contract):
#   1. $HERMES_INSTALL_DIR moves the checkout and leaves HERMES_HOME alone.
#      Case 1 invokes the installer with NO path arguments, which is exactly
#      what `irm ... | iex` does, so it covers the one-liner.
#   2. $HERMES_INSTALL_DIR and $HERMES_HOME are independent -- the program can
#      live on one volume and the data on another (GH #43868).
#   3. An explicit -InstallDir wins over the ambient $HERMES_INSTALL_DIR.
#   4. An explicit -HermesHome carries the checkout with it, so choosing a data
#      directory never strands the ~1 GB program on the default volume.
#   5. $HERMES_HOME alone still moves both (long-standing behavior; must not
#      regress when $HERMES_INSTALL_DIR is added above it in the precedence).
#
# HOW THIS RUNS THE CODE: by executing install.ps1 as a real subprocess with a
# crafted environment and reading what it reports back. -ShowResolvedPaths is a
# side-effect-free early exit that sits BELOW the path-resolution block, so the
# resolution runs exactly as it does during an install and nothing is written.
# Nothing here parses install.ps1's source (AGENTS.md bans source-reading
# tests: they pass on broken code and fail on correct refactors).
#
# HERMETIC ENVIRONMENT: every case sets all five profile variables plus both
# HERMES_* variables explicitly, so an inherited value is never the explanation
# for a pass or a failure.
#
# The directories named here are deliberately NOT created: the installer must
# report the caller's choice without probing the filesystem for it.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot "scripts/install.ps1"

if (-not (Test-Path $installScript)) {
    throw "Could not locate install.ps1 at $installScript"
}

$failures = 0
$script:lastRaw = ''
$script:lastEnv = ''

function Assert-Equal {
    param($Expected, $Actual, [Parameter(Mandatory = $true)][string]$Label)
    if ($Expected -ne $Actual) {
        Write-Host "FAIL: $Label" -ForegroundColor Red
        Write-Host "  expected: $Expected"
        Write-Host "  actual:   $Actual"
        if ($script:lastRaw) {
            # The installer's own account of what it resolved, plus the
            # environment it was handed: without both, a failure on a host you
            # cannot reach is pure guesswork.
            Write-Host "  installer reported: $script:lastRaw"
            Write-Host "  environment sent:   $script:lastEnv"
        }
        $script:failures++
    } else {
        Write-Host "OK: $Label" -ForegroundColor Green
    }
}

# --- Harness ---------------------------------------------------------------
$profileDir = [Environment]::GetFolderPath('UserProfile')

# Baseline profile root for every case's crafted environment: a real, long-form
# directory on this host. The probe below replaces it with whatever root the
# installer itself resolves, so the assertions do not depend on this guess.
$script:baseRoot = $profileDir

# Ask install.ps1 what paths it resolves under a given environment.
#
# -ShowResolvedPaths prints a JSON object on STDOUT and exits without touching
# anything. Stdout, deliberately: three separate stderr capture mechanisms were
# verified to come back EMPTY from the installer on a windows-latest runner
# while stdout arrived intact.
#
# Environment overrides are applied to this process and restored afterwards,
# since that is what the child inherits.
function Invoke-Resolve {
    param(
        [hashtable]$Environment = @{},
        [string[]]$ExtraArgs = @()
    )

    # Start from a self-consistent profile so nothing is inherited; callers
    # override only the variables their case is about. The probe below replaces
    # $script:baseRoot with the root the installer itself resolves, so these
    # assertions hold on any host and any account.
    $root = $script:baseRoot
    $localAppData = Join-Path (Join-Path $root 'AppData') 'Local'
    $env0 = @{
        TEMP               = (Join-Path $localAppData 'Temp')
        TMP                = (Join-Path $localAppData 'Temp')
        LOCALAPPDATA       = $localAppData
        APPDATA            = (Join-Path (Join-Path $root 'AppData') 'Roaming')
        USERPROFILE        = $root
        HERMES_HOME        = ''
        HERMES_INSTALL_DIR = ''
    }
    foreach ($key in $Environment.Keys) { $env0[$key] = $Environment[$key] }

    $psExe = (Get-Process -Id $PID).Path
    $outFile = [System.IO.Path]::GetTempFileName()
    $saved = @{}
    foreach ($key in $env0.Keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key) }

    try {
        foreach ($key in $env0.Keys) { Set-Item -Path "Env:$key" -Value $env0[$key] }
        $callArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $installScript) + $ExtraArgs + @('-ShowResolvedPaths')
        # The call operator, not Start-Process: on Windows Start-Process does
        # not hand the parent's modified environment block to the child, so the
        # installer would see the real user profile instead of the crafted one
        # this case sets. `&` inherits the environment on every host.
        #
        # stderr is merged into the same file rather than redirected separately:
        # Windows PowerShell 5.1 wraps ANY stderr from a native command in a
        # NativeCommandError record, and a bare `2>$file` still emits that
        # record into this script's error stream, which fails the 5.1 lane even
        # under 'Continue'. Merging with 2>&1 keeps the bytes and produces no
        # error record. The installer's stdout here is a single JSON object and
        # its diagnostics are all `[hermes] `-prefixed, so the two separate
        # cleanly on the way back out.
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $global:LASTEXITCODE = 0
        try {
            & $psExe @callArgs *> $outFile
        } finally {
            $ErrorActionPreference = $prevEAP
        }
        $exitCode = $LASTEXITCODE
        $raw = @(Get-Content -LiteralPath $outFile -ErrorAction SilentlyContinue)
        $stdout = ($raw | Where-Object { $_ -notlike '`[hermes`]*' }) -join "`n"
    } finally {
        foreach ($key in $saved.Keys) {
            if ($null -eq $saved[$key]) {
                Remove-Item -LiteralPath "Env:$key" -ErrorAction SilentlyContinue
            } else {
                Set-Item -Path "Env:$key" -Value $saved[$key]
            }
        }
        Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue
    }

    if ($null -eq $stdout) { $stdout = '' }
    $stdout = $stdout.Trim()
    $script:lastRaw = if ($stdout) { $stdout } else { '(child produced no stdout)' }
    $script:lastEnv = ($env0.Keys | Sort-Object | ForEach-Object { "$_=$($env0[$_])" }) -join '; '

    $paths = $null
    if ($stdout) {
        try { $paths = $stdout | ConvertFrom-Json } catch { $paths = $null }
    }

    return @{
        ExitCode   = $exitCode
        Stdout     = $stdout
        InstallDir = $(if ($paths) { $paths.install_dir } else { $null })
        HermesHome = $(if ($paths) { $paths.hermes_home } else { $null })
    }
}

# Ask the installer once, up front, where it puts things with a clean
# environment, and express every expectation relative to that. Deriving the
# default independently in the test would only prove the two derivations agree.
$probe = Invoke-Resolve
if ([string]::IsNullOrEmpty($probe.InstallDir) -or [string]::IsNullOrEmpty($probe.HermesHome)) {
    Write-Host "FAIL: the installer reports its default paths" -ForegroundColor Red
    Write-Host "  probe exit code: $($probe.ExitCode)"
    Write-Host "  probe env:       $script:lastEnv"
    Write-Host "  probe output:"
    foreach ($line in ($script:lastRaw -split "`r?`n")) {
        if ($line.Trim()) { Write-Host "    $line" }
    }
    Write-Host "FAILED: cannot continue without a baseline" -ForegroundColor Red
    exit 1
}
$defaultHome = $probe.HermesHome
$defaultInstallDir = $probe.InstallDir
Write-Host ""
Write-Host "baseline: HERMES_HOME=$defaultHome InstallDir=$defaultInstallDir"

# Paths on volumes that need not exist, so the assertion is about the choice
# the installer reports and not about a directory this test happened to create.
$programDir = 'D:\hermes-program'
$dataDir = 'E:\hermes-data'

Write-Host ""
Write-Host "-- `$HERMES_INSTALL_DIR relocates the checkout, not the data --"

# No path arguments at all: this is the `irm ... | iex` invocation, where the
# environment is the only channel available.
$result = Invoke-Resolve @{ HERMES_INSTALL_DIR = $programDir }
Assert-Equal -Expected 0 -Actual $result.ExitCode -Label "HERMES_INSTALL_DIR: install.ps1 reaches its early exit"
Assert-Equal -Expected $programDir -Actual $result.InstallDir -Label "HERMES_INSTALL_DIR is the checkout directory"
Assert-Equal -Expected $defaultHome -Actual $result.HermesHome -Label "HERMES_INSTALL_DIR leaves HERMES_HOME at its default"

Write-Host ""
Write-Host "-- program and data directories are independently choosable --"

$result = Invoke-Resolve @{ HERMES_INSTALL_DIR = $programDir; HERMES_HOME = $dataDir }
Assert-Equal -Expected $programDir -Actual $result.InstallDir -Label "program on one volume"
Assert-Equal -Expected $dataDir -Actual $result.HermesHome -Label "data on another volume"

Write-Host ""
Write-Host "-- an explicit -InstallDir beats the ambient `$HERMES_INSTALL_DIR --"

$result = Invoke-Resolve -Environment @{ HERMES_INSTALL_DIR = $programDir } -ExtraArgs @('-InstallDir', $defaultInstallDir)
Assert-Equal -Expected $defaultInstallDir -Actual $result.InstallDir -Label "-InstallDir wins over HERMES_INSTALL_DIR"

Write-Host ""
Write-Host "-- an explicit -HermesHome carries the checkout with it --"

$result = Invoke-Resolve -ExtraArgs @('-HermesHome', $dataDir)
Assert-Equal -Expected $dataDir -Actual $result.HermesHome -Label "-HermesHome is the data directory"
Assert-Equal -Expected (Join-Path $dataDir 'hermes-agent') -Actual $result.InstallDir -Label "-HermesHome moves the checkout too"

Write-Host ""
Write-Host "-- `$HERMES_HOME alone still moves both --"

$result = Invoke-Resolve @{ HERMES_HOME = $dataDir }
Assert-Equal -Expected $dataDir -Actual $result.HermesHome -Label "HERMES_HOME is the data directory"
Assert-Equal -Expected (Join-Path $dataDir 'hermes-agent') -Actual $result.InstallDir -Label "HERMES_HOME carries the checkout"

# --- Summary ---------------------------------------------------------------
Write-Host ""
if ($failures -gt 0) {
    Write-Host "FAILED: $failures assertion(s) failed" -ForegroundColor Red
    exit 1
} else {
    Write-Host "All install-directory selection tests passed." -ForegroundColor Green
    exit 0
}
