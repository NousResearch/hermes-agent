# Unit tests for install.ps1's Set-GitBashEnvVar helper.
#
# Run from a PowerShell prompt:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/tests/test-install-ps1-gitbash-source.ps1
#
# Background: Set-GitBashEnvVar locates bash.exe relative to the git.exe found
# on PATH via `(Get-Command git).Source`. When `git` resolves to a PowerShell
# alias or function instead of a real executable -- e.g. the `hub` wrapper,
# which makes `git --version` print "... hub version X" -- that .Source is the
# empty string. The helper then called `Split-Path (Split-Path "" -Parent)`,
# which throws "Cannot bind argument to parameter 'Path' because it is an empty
# string" and aborted the whole installer right after the "Git found" check
# (issue #55646). This verifies the helper tolerates an alias/function `git`.
#
# We extract just the function from install.ps1 via the AST so the installer's
# top-level body never runs (dot-sourcing would execute the whole script).

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot "scripts/install.ps1"

if (-not (Test-Path $installScript)) {
    throw "Could not locate install.ps1 at $installScript"
}

$failures = 0
function Assert-True {
    param([Parameter(Mandatory = $true)] $Condition,
          [Parameter(Mandatory = $true)] [string]$Label)
    if (-not $Condition) {
        Write-Host "FAIL: $Label" -ForegroundColor Red
        $script:failures++
    } else {
        Write-Host "OK: $Label" -ForegroundColor Green
    }
}

# --- Load Set-GitBashEnvVar from install.ps1 without executing the script ---
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installScript, [ref]$tokens, [ref]$errors)
$fnAst = $ast.FindAll(
    {
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Set-GitBashEnvVar'
    }, $true) | Select-Object -First 1

if (-not $fnAst) {
    throw "Set-GitBashEnvVar not found in install.ps1 -- did the helper get renamed/removed?"
}
. ([scriptblock]::Create($fnAst.Extent.Text))

# The helper depends on a few output helpers and the script-scoped $HermesHome.
# Stub the output helpers so the function can run in isolation, and point every
# candidate root at an empty temp dir so no bash.exe is found -- that keeps the
# test hermetic (it never persists HERMES_GIT_BASH_PATH to the User scope).
function Write-Info    { param($m) }
function Write-Warn    { param($m) }
function Write-Success { param($m) }

$emptyRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-gitbash-test-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $emptyRoot -Force | Out-Null

# Save and neutralize anything the candidate list keys off of.
$savedHermesHome  = $HermesHome
$savedProgramFiles = $env:ProgramFiles
$savedLocalAppData = $env:LocalAppData
$savedPf86 = [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")

try {
    $HermesHome       = $emptyRoot
    $env:ProgramFiles = $emptyRoot
    $env:LocalAppData = $emptyRoot
    [Environment]::SetEnvironmentVariable("ProgramFiles(x86)", $emptyRoot, "Process")

    Write-Host ""
    Write-Host "-- Set-GitBashEnvVar with alias/function git (empty .Source) --"

    # Shadow `git` with a function so Get-Command git returns a command whose
    # .Source is the empty string -- exactly the `hub`-wrapper situation.
    function git { "git version 2.54.0.windows.1 hub version 2.14.2" }

    $gitCmd = Get-Command git
    Assert-True ([string]::IsNullOrEmpty($gitCmd.Source)) `
        -Label "precondition: function git has empty .Source"

    $threw = $false
    try {
        Set-GitBashEnvVar
    } catch {
        $threw = $true
        Write-Host "  threw: $($_.Exception.Message)"
    }
    Assert-True (-not $threw) `
        -Label "Set-GitBashEnvVar does not throw when git resolves to an alias/function"
} finally {
    Remove-Item Function:\git -ErrorAction SilentlyContinue
    $env:ProgramFiles = $savedProgramFiles
    $env:LocalAppData = $savedLocalAppData
    [Environment]::SetEnvironmentVariable("ProgramFiles(x86)", $savedPf86, "Process")
    $HermesHome = $savedHermesHome
    Remove-Item -Recurse -Force $emptyRoot -ErrorAction SilentlyContinue
}

# --- Summary ---
Write-Host ""
if ($failures -gt 0) {
    Write-Host "FAILED: $failures assertion(s) failed" -ForegroundColor Red
    exit 1
} else {
    Write-Host "All Set-GitBashEnvVar tests passed." -ForegroundColor Green
    exit 0
}
