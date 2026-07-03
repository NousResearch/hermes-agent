# Unit tests for install.ps1's ConvertTo-LongPath / Resolve-TempEnvPath helpers.
#
# Run from a PowerShell prompt:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/tests/test-install-ps1-longpath.ps1
#
# Background: on a Windows profile whose folder name contains a space (e.g.
# "First Last") or a dot (e.g. "Stone.ZEN8"), %TEMP%/%TMP% can be exposed as
# an 8.3 short path (C:\Users\FIRST~1.LAS\... or STONE~1.ZEN\...). PowerShell's
# FileSystem provider chokes on the "~1.ext" component when it reaches a provider
# cmdlet (Tee-Object -FilePath), aborting the Node/Electron install+build stages.
# install.ps1 expands such paths to their long form up front; this verifies the
# helper contracts.
#
# We extract just the functions from install.ps1 via the AST so the installer's
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
    param([Parameter(Mandatory = $true)] $Expected,
          [Parameter(Mandatory = $true)] $Actual,
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

function Load-InstallFunction {
    param([Parameter(Mandatory = $true)][string]$Name)
    $tokens = $null
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($installScript, [ref]$tokens, [ref]$errors)
    $fnAst = $ast.FindAll(
        {
            param($node)
            $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq $Name
        }, $true) | Select-Object -First 1
    if (-not $fnAst) {
        throw "$Name not found in install.ps1 -- did the helper get renamed/removed?"
    }
    . ([scriptblock]::Create($fnAst.Extent.Text))
}

Load-InstallFunction 'ConvertTo-LongPath'
Load-InstallFunction 'Resolve-TempEnvPath'

# --- Tests ---
Write-Host ""
Write-Host "-- ConvertTo-LongPath --"

Assert-Equal -Expected "" -Actual (ConvertTo-LongPath "") -Label "empty string returns empty"
Assert-Equal -Expected $null -Actual (ConvertTo-LongPath $null) -Label "null returns null"

# No 8.3 component -> returned verbatim (even with spaces).
$longish = "C:\Users\First Last\AppData\Local\Temp"
Assert-Equal -Expected $longish -Actual (ConvertTo-LongPath $longish) -Label "long path with spaces is unchanged"

$noTilde = "/tmp/some/long/path"
Assert-Equal -Expected $noTilde -Actual (ConvertTo-LongPath $noTilde) -Label "tilde-free path is unchanged"

# Looks like an 8.3 name but does not exist -> graceful fallback to the input
# (FolderExists/FileExists both false, or COM unavailable on this host).
$fakeShort = "C:\Users\FIRST~1.LAS\does\not\exist"
Assert-Equal -Expected $fakeShort -Actual (ConvertTo-LongPath $fakeShort) -Label "nonexistent 8.3 path falls back to input"

Write-Host ""
Write-Host "-- Resolve-TempEnvPath --"

Assert-Equal -Expected "" -Actual (Resolve-TempEnvPath "") -Label "empty string returns empty"
Assert-Equal -Expected $longish -Actual (Resolve-TempEnvPath $longish) -Label "long temp path is unchanged"

# Standard profile temp with an unresolvable 8.3 alias -> rebuilt from
# LocalApplicationData (long form on every host).
$shortProfileTemp = "C:\Users\STONE~1.ZEN\AppData\Local\Temp"
$localAppData = [Environment]::GetFolderPath('LocalApplicationData')
$expectedTemp = if ($localAppData -and $localAppData -notmatch '~\d') {
    Join-Path $localAppData 'Temp'
} else {
    $shortProfileTemp
}
Assert-Equal -Expected $expectedTemp -Actual (Resolve-TempEnvPath $shortProfileTemp) -Label "short profile temp rebuilds from LocalApplicationData"

# Subpaths under the default temp dir are preserved.
$shortNested = "C:\Users\STONE~1.ZEN\AppData\Local\Temp\hermes-desktop-build-123.log"
$expectedNested = if ($localAppData -and $localAppData -notmatch '~\d') {
    Join-Path (Join-Path $localAppData 'Temp') 'hermes-desktop-build-123.log'
} else {
    $shortNested
}
Assert-Equal -Expected $expectedNested -Actual (Resolve-TempEnvPath $shortNested) -Label "short temp subpath is preserved"

# Custom temp outside AppData\Local\Temp stays on the COM fallback path.
$customShort = "D:\SHORT~1\Temp"
Assert-Equal -Expected $customShort -Actual (Resolve-TempEnvPath $customShort) -Label "non-profile short temp is unchanged"

# --- Summary ---
Write-Host ""
if ($failures -gt 0) {
    Write-Host "FAILED: $failures assertion(s) failed" -ForegroundColor Red
    exit 1
} else {
    Write-Host "All long-path helper tests passed." -ForegroundColor Green
    exit 0
}
