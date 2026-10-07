# Tests for install.ps1's RabbitHome/InstallDir separation guard (#124526).
#
# Run from a PowerShell prompt:
#
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/tests/test-install-ps1-store-guard.ps1
#
# Background: passing -RabbitHome and -InstallDir as the same path (or a
# RabbitHome nested inside -InstallDir) makes the pm tool store land at
# <InstallDir>\tools, INSIDE the checkout. Two failures follow: the
# repository stage's occupied-directory preflight refuses every retry after
# the first run populated the folder ("exists and is not a Rabbit git
# checkout"), and `git stash --include-untracked` would sweep the toolchain
# into the stash. The installer now refuses the combination up front, before
# any download, unless RABBIT_RUNTIME_DIR parks the store elsewhere.
#
# HOW THIS RUNS THE CODE: by executing install.ps1 as a real subprocess with
# crafted -RabbitHome / -InstallDir arguments. -ShowResolvedPaths is a
# side-effect-free early exit that runs AFTER Initialize-ResolvedPaths, so
# the guard executes exactly as during an install. The refusal is a `throw`,
# which under -File exits the child non-zero before the JSON is printed; the
# accepted configurations print the resolved-path JSON with the requested
# paths intact. Nothing here parses install.ps1's source (AGENTS.md bans
# source-reading tests: they pass on broken code and fail on correct
# refactors).

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot "scripts/install.ps1"

if (-not (Test-Path $installScript)) {
    throw "Could not locate install.ps1 at $installScript"
}

$failures = 0

function Assert-True {
    param([bool]$Condition, [Parameter(Mandatory = $true)][string]$Label)
    if (-not $Condition) {
        Write-Host "FAIL: $Label" -ForegroundColor Red
        $script:failures++
    } else {
        Write-Host "OK: $Label" -ForegroundColor Green
    }
}

# Run install.ps1 -ShowResolvedPaths with explicit -RabbitHome/-InstallDir and
# capture (exit code, stdout, stderr). The call operator, not Start-Process:
# `&` inherits this process's environment on every host, and stderr is merged
# so Windows PowerShell 5.1's NativeCommandError records cannot fail the lane.
function Invoke-ResolvedPaths {
    param([string[]]$ExtraArgs = @())

    $psExe = (Get-Process -Id $PID).Path
    $outFile = [System.IO.Path]::GetTempFileName()
    $savedRuntimeDir = [Environment]::GetEnvironmentVariable('RABBIT_RUNTIME_DIR')
    $savedRabbitHome = [Environment]::GetEnvironmentVariable('RABBIT_HOME')
    try {
        if ($null -eq $savedRuntimeDir) {
            Remove-Item Env:RABBIT_RUNTIME_DIR -ErrorAction SilentlyContinue
        } else {
            $env:RABBIT_RUNTIME_DIR = $savedRuntimeDir
        }
        if ($null -eq $savedRabbitHome) {
            Remove-Item Env:RABBIT_HOME -ErrorAction SilentlyContinue
        } else {
            $env:RABBIT_HOME = $savedRabbitHome
        }
        $callArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $installScript) + $ExtraArgs + @('-ShowResolvedPaths')
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
        $stdout = ($raw | Where-Object { $_ -notlike '`[rabbit`]*' }) -join "`n"
        $stderr = ($raw | Where-Object { $_ -like '`[rabbit`]*' -or $_ -match 'RabbitHome|install directory|tool store' }) -join "`n"
        return @{ ExitCode = $exitCode; Stdout = $stdout; All = ($raw -join "`n") }
    } finally {
        if ($null -eq $savedRuntimeDir) {
            Remove-Item Env:RABBIT_RUNTIME_DIR -ErrorAction SilentlyContinue
        } else {
            $env:RABBIT_RUNTIME_DIR = $savedRuntimeDir
        }
        if ($null -eq $savedRabbitHome) {
            Remove-Item Env:RABBIT_HOME -ErrorAction SilentlyContinue
        } else {
            $env:RABBIT_HOME = $savedRabbitHome
        }
        Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue
    }
}

$base = Join-Path ([System.IO.Path]::GetTempPath()) ("rabbit-store-guard-" + [guid]::NewGuid())
$home1 = Join-Path $base 'home1'
$dir1 = Join-Path $base 'checkout1'
$insideHome = Join-Path $base 'checkout2'      # RabbitHome nested inside InstallDir
$insideDir = Join-Path $base 'checkout2'       # the InstallDir that contains it

try {
    # 1. Same path for both: refused, before any download.
    $r = Invoke-ResolvedPaths @('-RabbitHome', $home1, '-InstallDir', $home1)
    Assert-True ($r.ExitCode -ne 0) 'same RabbitHome and InstallDir is refused (non-zero exit)'
    Assert-True ($r.All -match 'tool store would land inside the checkout') 'refusal names the tool-store cause'

    # 2. RabbitHome nested inside InstallDir: refused for the same reason.
    $r = Invoke-ResolvedPaths @('-RabbitHome', $insideHome, '-InstallDir', $insideDir)
    Assert-True ($r.ExitCode -ne 0) 'RabbitHome inside InstallDir is refused (non-zero exit)'

    # 3. Distinct paths: accepted, and the caller's paths survive verbatim.
    $r = Invoke-ResolvedPaths @('-RabbitHome', $home1, '-InstallDir', $dir1)
    Assert-True ($r.ExitCode -eq 0) "distinct paths are accepted (exit $($r.ExitCode))"
    $paths = $null
    try { $paths = $r.Stdout | ConvertFrom-Json } catch { $paths = $null }
    Assert-True ($null -ne $paths) 'accepted run prints the resolved-path JSON'
    if ($paths) {
        Assert-True ("$($paths.rabbit_home)" -eq $home1) 'rabbit_home echoes the requested -RabbitHome'
        Assert-True ("$($paths.install_dir)" -eq $dir1) 'install_dir echoes the requested -InstallDir'
    }

    # 4. RABBIT_RUNTIME_DIR outside the checkout lifts the refusal: the store
    #    no longer lands inside InstallDir, so the configuration is legitimate.
    $storeElsewhere = Join-Path $base 'store'
    $saved = $env:RABBIT_RUNTIME_DIR
    try {
        $env:RABBIT_RUNTIME_DIR = $storeElsewhere
        $r = Invoke-ResolvedPaths @('-RabbitHome', $home1, '-InstallDir', $home1)
        Assert-True ($r.ExitCode -eq 0) "RABBIT_RUNTIME_DIR outside lifts the refusal (exit $($r.ExitCode))"
    } finally {
        if ($null -eq $saved) {
            Remove-Item Env:RABBIT_RUNTIME_DIR -ErrorAction SilentlyContinue
        } else {
            $env:RABBIT_RUNTIME_DIR = $saved
        }
    }

    # 5. Default derivation never trips the guard: InstallDir defaults to
    #    <RabbitHome>\rabbit-agent, which is a child, not a parent.
    $r = Invoke-ResolvedPaths @('-RabbitHome', $home1)
    Assert-True ($r.ExitCode -eq 0) "default -InstallDir under -RabbitHome stays accepted (exit $($r.ExitCode))"
    $paths = $null
    try { $paths = $r.Stdout | ConvertFrom-Json } catch { $paths = $null }
    if ($paths) {
        Assert-True ("$($paths.install_dir)" -eq (Join-Path $home1 'rabbit-agent')) 'default install_dir derives under rabbit_home'
    }
} finally {
    Remove-Item -LiteralPath $base -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures -gt 0) {
    Write-Host "$failures assertion(s) failed." -ForegroundColor Red
    exit 1
}
Write-Host 'RabbitHome/InstallDir separation guard tests passed.' -ForegroundColor Green
exit 0
