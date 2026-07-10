# Unit tests for install.ps1's ConvertTo-LongPath helper.
#
# Run from a PowerShell prompt:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/tests/test-install-ps1-longpath.ps1
#
# Background: on a Windows profile whose folder name contains a space (e.g.
# "First Last"), %TEMP%/%TMP% can be exposed as an 8.3 short path
# (C:\Users\FIRST~1.LAS\...). PowerShell's FileSystem provider chokes on the
# "~1.ext" component when it reaches a provider cmdlet (Tee-Object -FilePath),
# aborting the Node/Electron install+build stages. install.ps1 expands such
# paths to their long form up front; this verifies the helper's contract.
#
# We extract just the function from install.ps1 via the AST so the installer's
# top-level body never runs (dot-sourcing would execute the whole script).
# The COM-backed expansion only fires for inputs containing "~<digit>"; the
# pass-through and graceful-fallback paths are assertable on any host (incl.
# non-Windows pwsh, where the COM object is simply unavailable).

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot "scripts/install.ps1"

if (-not (Test-Path $installScript)) {
    throw "Could not locate install.ps1 at $installScript"
}

$failures = 0
function Assert-Equal {
    param($Expected,
          $Actual,
          [Parameter(Mandatory = $true)] [string]$Label)
    if ($Expected -ne $Actual) {
        Write-Host "FAIL: $Label" -ForegroundColor Red
        Write-Host "  expected: $Expected"
        Write-Host "  actual:   $Actual"
        $script:failures++
    } else {
        Write-Host "OK: $Label" -ForegroundColor Green
    }
}

# --- Load ConvertTo-LongPath from install.ps1 without executing the script ---
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installScript, [ref]$tokens, [ref]$errors)
$fnAst = $ast.FindAll(
    {
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'ConvertTo-LongPath'
    }, $true) | Select-Object -First 1

if (-not $fnAst) {
    throw "ConvertTo-LongPath not found in install.ps1 -- did the helper get renamed/removed?"
}
. ([scriptblock]::Create($fnAst.Extent.Text))

# --- Tests ---
Write-Host ""
Write-Host "-- ConvertTo-LongPath --"

Assert-Equal -Expected "" -Actual (ConvertTo-LongPath "") -Label "empty string returns empty"
Assert-Equal -Expected "" -Actual (ConvertTo-LongPath $null) -Label "null input returns empty string"

# No 8.3 component -> returned verbatim (even with spaces).
$longish = "C:\Users\First Last\AppData\Local\Temp"
Assert-Equal -Expected $longish -Actual (ConvertTo-LongPath $longish) -Label "long path with spaces is unchanged"

$noTilde = "/tmp/some/long/path"
Assert-Equal -Expected $noTilde -Actual (ConvertTo-LongPath $noTilde) -Label "tilde-free path is unchanged"

# Looks like an 8.3 name but does not exist -> graceful fallback to the input
# (FolderExists/FileExists both false, or COM unavailable on this host).
$fakeShort = "C:\Users\FIRST~1.LAS\does\not\exist"
Assert-Equal -Expected $fakeShort -Actual (ConvertTo-LongPath $fakeShort) -Label "nonexistent 8.3 path falls back to input"

# --- Load Normalize-ProfileEnvVars (the accented-username fix, GH #52842) ---
$normAst = $ast.FindAll(
    {
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Normalize-ProfileEnvVars'
    }, $true) | Select-Object -First 1

if (-not $normAst) {
    throw "Normalize-ProfileEnvVars not found in install.ps1 -- did the helper get renamed/removed?"
}
. ([scriptblock]::Create($normAst.Extent.Text))

# --- Tests for Normalize-ProfileEnvVars ---
# On a host where ConvertTo-LongPath can't resolve a short alias (COM/PInvoke
# both unavailable, e.g. non-Windows pwsh), the function must at least not
# throw and must leave an already-long path untouched.
Write-Host ""
Write-Host "-- Normalize-ProfileEnvVars --"

# A long, non-short path must survive the call unchanged (no spurious mutation).
$env:HermesTest_TMP = "C:\Users\Renombrado\AppData\Local\Temp"
Normalize-ProfileEnvVars
Assert-Equal -Expected "C:\Users\Renombrado\AppData\Local\Temp" -Actual $env:HermesTest_TMP -Label "long profile path unchanged by normalization" -ea Stop

# The function must enumerate the documented set of profile-rooted vars without
# error and without touching unrelated vars.
$env:HermesTest_MARKER = "untouched"
Normalize-ProfileEnvVars
Assert-Equal -Expected "untouched" -Actual $env:HermesTest_MARKER -Label "unrelated env var left untouched"

# --- Summary ---
Write-Host ""
if ($failures -gt 0) {
    Write-Host "FAILED: $failures assertion(s) failed" -ForegroundColor Red
    exit 1
} else {
    Write-Host "All install.ps1 path-normalization tests passed." -ForegroundColor Green
    exit 0
}
