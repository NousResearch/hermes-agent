# Regression tests for install.ps1's PowerShell language mode preflight (#89857).
#
# Run from a PowerShell prompt:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/tests/test-install-ps1-language-mode.ps1
#
# AppLocker / WDAC enforcement puts PowerShell in ConstrainedLanguage, where
# method calls on non-core .NET types are refused. install.ps1 makes one at
# script scope long before it looks at its own parameters, so without the
# preflight even -ProtocolVersion -- a read-only query that touches nothing --
# dies with a raw, host-localized .NET error pointing into a cached copy of a
# script the operator never wrote.
#
# These tests run the installer as a real subprocess in a runspace that has
# been put into ConstrainedLanguage, which is the same restriction the policy
# applies. They do NOT need AppLocker configured on the runner.
#
# Nothing here installs anything: both invocations use -ProtocolVersion, which
# returns before Main / Invoke-AllStages.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot "scripts\install.ps1"

if (-not (Test-Path $installScript)) {
    throw "Could not locate install.ps1 at $installScript"
}

$failures = 0
function Assert-Equal {
    param([Parameter(Mandatory=$true)] $Expected,
          [Parameter(Mandatory=$true)] $Actual,
          [Parameter(Mandatory=$true)] [string]$Label)
    if ($Expected -ne $Actual) {
        Write-Host "FAIL: $Label" -ForegroundColor Red
        Write-Host "  expected: $Expected"
        Write-Host "  actual:   $Actual"
        $script:failures++
    } else {
        Write-Host "OK: $Label" -ForegroundColor Green
    }
}
function Assert-True {
    param([Parameter(Mandatory=$true)] $Condition,
          [Parameter(Mandatory=$true)] [string]$Label)
    if (-not $Condition) {
        Write-Host "FAIL: $Label" -ForegroundColor Red
        $script:failures++
    } else {
        Write-Host "OK: $Label" -ForegroundColor Green
    }
}

# The shell that delivers install.ps1 is whatever the user already has, so the
# child must be the SAME host this test is running under -- 5.1 and 7 differ in
# how they surface a terminating error from a child script, and the CI job runs
# this file under both.
$psExe = (Get-Process -Id $PID).Path

# Run a scriptblock's worth of PowerShell as a subprocess whose runspace has
# been dropped into ConstrainedLanguage first. Streams go to files rather than
# through the pipeline: Windows PowerShell 5.1 wraps a child's stderr in a
# NativeCommandError and folds it into the caller's own error stream, which
# would make an ordinary 2>&1 capture look like a failure of this test.
function Invoke-Constrained {
    param([Parameter(Mandatory=$true)] [string]$Body)

    $dir = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-clm-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    $wrapper = Join-Path $dir "wrapper.ps1"
    $outFile = Join-Path $dir "out.txt"
    $errFile = Join-Path $dir "err.txt"

    # The mode assignment is one-way within a session, so it must happen in the
    # child rather than here -- this test process stays FullLanguage.
    $script = "`$ExecutionContext.SessionState.LanguageMode = 'ConstrainedLanguage'" + [Environment]::NewLine + $Body
    Set-Content -LiteralPath $wrapper -Value $script -Encoding ascii

    $proc = Start-Process -FilePath $psExe `
        -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $wrapper) `
        -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $outFile -RedirectStandardError $errFile

    $result = [pscustomobject]@{
        ExitCode = $proc.ExitCode
        Output   = ((Get-Content -LiteralPath $outFile -Raw -ErrorAction SilentlyContinue) + [Environment]::NewLine +
                    (Get-Content -LiteralPath $errFile -Raw -ErrorAction SilentlyContinue))
    }
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    return $result
}

# -----------------------------------------------------------------------------
# Guardrail: the harness really is constrained.
#
# Without this the whole file passes vacuously on any host where the mode
# assignment silently does not take -- every assertion below would be checking
# FullLanguage behavior while claiming to check ConstrainedLanguage.
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "-- harness --"
$probe = Invoke-Constrained -Body @'
Write-Output ("mode=" + $ExecutionContext.SessionState.LanguageMode)
try { $null = [Environment]::GetEnvironmentVariable('TEMP'); Write-Output 'dotnet=ALLOWED' }
catch { Write-Output ('dotnet=' + $_.FullyQualifiedErrorId) }
'@
Assert-True ($probe.Output -match 'mode=ConstrainedLanguage') `
    -Label "harness runspace reports ConstrainedLanguage"
Assert-True ($probe.Output -match 'dotnet=MethodInvocationNotSupportedInConstrainedLanguage') `
    -Label "harness refuses [Environment]::GetEnvironmentVariable (the call install.ps1 dies on)"

# -----------------------------------------------------------------------------
# Test: the preflight fires before anything else can throw.
#
# -ProtocolVersion is the strongest form of this: it is a read-only query, so
# the ONLY reason it can fail is the language mode.
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "-- ConstrainedLanguage --"
$constrained = Invoke-Constrained -Body ("& '" + $installScript + "' -ProtocolVersion" + [Environment]::NewLine + "exit `$LASTEXITCODE")

Assert-Equal -Expected 1 -Actual $constrained.ExitCode `
    -Label "-ProtocolVersion exits 1 under ConstrainedLanguage"
Assert-True ($constrained.Output -match 'ConstrainedLanguage mode') `
    -Label "the message names the language mode"
Assert-True ($constrained.Output -match 'ExecutionPolicy Bypass does not help') `
    -Label "the message pre-empts the -ExecutionPolicy Bypass attempt (#89857's step 2)"
Assert-True ($constrained.Output -match 'allow-list') `
    -Label "the message says what an administrator has to change"

# This is the regression itself: before the preflight, the run died inside
# Set-LongProfileEnvVars with a .NET error and no mention of why.
Assert-True (-not ($constrained.Output -match 'MethodInvocationNotSupportedInConstrainedLanguage')) `
    -Label "the raw .NET error never reaches the operator"
Assert-True (-not ($constrained.Output -match 'GetEnvironmentVariable')) `
    -Label "the failure does not surface an install.ps1 line number"

# -----------------------------------------------------------------------------
# Test: FullLanguage is untouched.
#
# The guard is a refusal, so the thing most worth pinning is that it does not
# fire on the hosts every other user has.
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "-- FullLanguage --"
$version = & $psExe -NoProfile -ExecutionPolicy Bypass -File $installScript -ProtocolVersion
Assert-Equal -Expected 0 -Actual $LASTEXITCODE -Label "-ProtocolVersion still exits 0 under FullLanguage"
Assert-True ($version -match '^\d+$') -Label "-ProtocolVersion still emits an integer (got: $version)"

$manifest = & $psExe -NoProfile -ExecutionPolicy Bypass -File $installScript -Manifest
Assert-Equal -Expected 0 -Actual $LASTEXITCODE -Label "-Manifest still exits 0 under FullLanguage"
Assert-True (($manifest -join '') -match '"protocol_version"') `
    -Label "-Manifest still emits the manifest, so the guard did not pollute stdout"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
Write-Host ""
if ($failures -gt 0) {
    Write-Host "FAILED: $failures assertion(s) failed" -ForegroundColor Red
    exit 1
} else {
    Write-Host "All language mode tests passed." -ForegroundColor Green
    exit 0
}
