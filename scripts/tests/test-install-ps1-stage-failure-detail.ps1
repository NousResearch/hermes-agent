# Tests for install.ps1's stage-failure diagnostics.
#
# Run from a PowerShell prompt:
#
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/tests/test-install-ps1-stage-failure-detail.ps1
#
# Background: the stage dispatcher's catch reported only "$_" -- for a real
# exception that is the localized message alone ("Odmowa dostępu"), which
# cannot tell an inherited ACL from an elevated process's leftover file, AV
# interference, or a bootstrap bug (#124052). Get-StageFailureReason appends
# the exception type, HResult and the failing operation; a Fail() string is
# already actionable and must pass through untouched.
#
# HOW THIS RUNS THE CODE: install.ps1 runs as a real subprocess with crafted
# parameters, exactly like the -Json driver desktop's bootstrap uses; the
# assertions read only the frames it prints. Nothing parses install.ps1's
# source (AGENTS.md bans source-reading tests: they pass on broken code and
# fail on correct refactors).
#
# Both cases are hermetic: they stop inside their stage before any download
# or git use (config's first directory write; repository's occupied-check
# refusal), so no network and no pinned tooling is involved.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot "scripts/install.ps1"

if (-not (Test-Path $installScript)) {
    throw "Could not locate install.ps1 at $installScript"
}

$failures = 0

function Assert-True {
    param([bool]$Condition, [string]$Label, [string]$Detail = '')
    if ($Condition) {
        Write-Host "OK: $Label" -ForegroundColor Green
    } else {
        Write-Host "FAIL: $Label" -ForegroundColor Red
        if ($Detail) { Write-Host $Detail }
        $script:failures++
    }
}

# Run the installer as a child process and capture everything it printed
# (stdout merged with stderr into one file: Windows PowerShell 5.1 wraps any
# separately-redirected native stderr in a NativeCommandError record, which
# fails the 5.1 lane even under 'Continue'; merging keeps the bytes without
# the record). The -Json frames are the stdout lines that start with '{'.
function Invoke-InstallerStage {
    param([string[]]$ExtraArgs)
    $psExe = (Get-Process -Id $PID).Path
    $outFile = [System.IO.Path]::GetTempFileName()
    try {
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try { & $psExe -NoProfile -ExecutionPolicy Bypass -File $installScript @ExtraArgs *> $outFile }
        finally { $ErrorActionPreference = $prevEAP }
        $exitCode = $LASTEXITCODE
    } finally {
        $raw = @(Get-Content -LiteralPath $outFile -ErrorAction SilentlyContinue)
        Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue
    }
    return @{ ExitCode = $exitCode; Output = ($raw -join "`n") }
}

function Get-Frames {
    param([string]$Output)
    return @($Output -split "`n" | Where-Object { $_.StartsWith('{') } |
        ForEach-Object { $_ | ConvertFrom-Json })
}

Write-Host ""
Write-Host "-- a real exception reports type, HResult and where it died --"

# $HermesHome exists as a plain file, so Stage-Config's first directory
# create (cron under the home) fails with a genuine filesystem IOException
# -- not a Fail() string -- which is the class of error whose context the
# old catch collapsed to the localized message alone.
$homeFile = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-detail-home-" + [guid]::NewGuid().ToString('N'))
Set-Content -LiteralPath $homeFile -Value 'not a directory'
try {
    $result = Invoke-InstallerStage @('-Stage', 'config', '-Json', '-HermesHome', $homeFile)
    $frames = Get-Frames $result.Output
    Assert-True ($result.ExitCode -eq 1) "file home: the stage exits 1" $result.Output
    Assert-True ($frames.Count -eq 1) "file home: exactly one frame is printed" $result.Output
    Assert-True ($frames[0].ok -eq $false -and $frames[0].stage -eq 'config') "file home: the frame names the failed stage" $result.Output
    # The exact exception type and failing statement are platform-dependent
    # (the first refused write differs between Windows and the POSIX lanes);
    # the contract under test is that the reason names a type, the HResult
    # and the offending path instead of a localized message alone.
    Assert-True ("$($frames[0].reason)" -match 'System\.[A-Za-z.]+Exception') "file home: the reason names the exception type" $result.Output
    Assert-True ("$($frames[0].reason)" -match 'hresult 0x[0-9A-Fa-f]{8}') "file home: the reason carries the HResult" $result.Output
    Assert-True ("$($frames[0].reason)".Contains($homeFile)) "file home: the reason names the denied path" $result.Output
} finally {
    Remove-Item -LiteralPath $homeFile -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "-- a Fail() string stays verbatim, with no diagnostics bolted on --"

# The occupied non-checkout refusal is a Fail() string: actionable already
# and asserted verbatim elsewhere, so it must not grow a diagnostic suffix.
$install = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-detail-install-" + [guid]::NewGuid().ToString('N'))
$home2 = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-detail-home2-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $install | Out-Null
Set-Content -LiteralPath (Join-Path $install 'user-file') -Value 'preserve me'
try {
    $result = Invoke-InstallerStage @('-Stage', 'repository', '-Json', '-InstallDir', $install, '-HermesHome', $home2)
    $frames = Get-Frames $result.Output
    Assert-True ($result.ExitCode -eq 1) "occupied: the stage exits 1" $result.Output
    Assert-True ("$($frames[0].reason)" -match 'exists and is not a Hermes git checkout') "occupied: the reason keeps Fail()'s message" $result.Output
    Assert-True ("$($frames[0].reason)" -notmatch 'hresult 0x') "occupied: no diagnostic suffix is bolted onto a Fail() string" $result.Output
    Assert-True (Test-Path (Join-Path $install 'user-file')) "occupied: the user's file is preserved" $result.Output
} finally {
    Remove-Item -LiteralPath $install -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
if ($failures -gt 0) {
    Write-Host "FAILED: $failures assertion(s) failed" -ForegroundColor Red
    exit 1
} else {
    Write-Host "All stage-failure detail tests passed." -ForegroundColor Green
    exit 0
}
