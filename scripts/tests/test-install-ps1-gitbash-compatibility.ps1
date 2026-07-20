# Unit tests for install.ps1's Git Bash compatibility and Mandatory-ASLR
# guidance helpers. The installer itself is never executed: functions are
# extracted through the PowerShell AST to avoid downloads, PATH changes, or
# user-environment writes.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot "scripts\install.ps1"

$failures = 0
function Assert-Equal {
    param($Expected, $Actual, [string]$Label)
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
    param($Condition, [string]$Label)
    if (-not $Condition) {
        Write-Host "FAIL: $Label" -ForegroundColor Red
        $script:failures++
    } else {
        Write-Host "OK: $Label" -ForegroundColor Green
    }
}

$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $installScript, [ref]$tokens, [ref]$parseErrors
)
if ($parseErrors.Count -gt 0) {
    throw "install.ps1 has parse errors: $($parseErrors -join '; ')"
}

foreach ($name in @(
    "Test-GitBashCompatibility",
    "Test-MandatoryAslrEnabled",
    "Get-GitRootFromBashPath",
    "New-GitBashAslrFailureReason",
    "Stage-Git"
)) {
    $fnAst = $ast.FindAll(
        {
            param($node)
            $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq $name
        }, $true
    ) | Select-Object -First 1
    if (-not $fnAst) { throw "$name not found in install.ps1" }
    . ([scriptblock]::Create($fnAst.Extent.Text))
}

Write-Host ""
Write-Host "-- Git root resolution --"
Assert-Equal "C:\Program Files\Git" `
    (Get-GitRootFromBashPath "C:\Program Files\Git\bin\bash.exe") `
    "PortableGit/full Git bin layout"
Assert-Equal "C:\Program Files\Git" `
    (Get-GitRootFromBashPath "C:\Program Files\Git\usr\bin\bash.exe") `
    "usr/bin layout"

Write-Host ""
Write-Host "-- Mandatory ASLR detection --"
function Get-ProcessMitigation {
    param([switch]$System)
    [pscustomobject]@{ Aslr = [pscustomobject]@{ ForceRelocateImages = "ON" } }
}
Assert-Equal $true (Test-MandatoryAslrEnabled) "ForceRelocateImages ON is detected"
function Get-ProcessMitigation {
    param([switch]$System)
    [pscustomobject]@{ Aslr = [pscustomobject]@{ ForceRelocateImages = "NOTSET" } }
}
Assert-Equal $false (Test-MandatoryAslrEnabled) "ForceRelocateImages NOTSET is not diagnosed"

Write-Host ""
Write-Host "-- Actionable remediation --"
$reason = New-GitBashAslrFailureReason "C:\Program Files\Git\bin\bash.exe"
Assert-True ($reason -match "Mandatory ASLR") "reason identifies Mandatory ASLR"
Assert-True ($reason -match "Reinstalling Git will not change") "reason rejects ineffective reinstall"
Assert-True ($reason -match [regex]::Escape("C:\Program Files\Git")) "reason uses selected Git root"
Assert-True ($reason -match "Set-ProcessMitigation") "reason includes targeted mitigation command"

Write-Host ""
Write-Host "-- External-program probe --"
$gitCommand = Get-Command git -ErrorAction SilentlyContinue
$gitBash = $null
if ($gitCommand -and $gitCommand.Source) {
    $gitRoot = Split-Path (Split-Path $gitCommand.Source -Parent) -Parent
    foreach ($candidate in @("$gitRoot\bin\bash.exe", "$gitRoot\usr\bin\bash.exe")) {
        if (Test-Path -LiteralPath $candidate) { $gitBash = $candidate; break }
    }
}
if ($gitBash) {
    Assert-Equal $true (Test-GitBashCompatibility $gitBash) `
        "installed Git Bash launches external MSYS programs"
} else {
    Write-Host "SKIP: no Git Bash found next to git.exe" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "-- Stage failure propagation --"
function Install-Git { return $false }
$script:GitInstallFailureReason = "specific Git Bash failure"
$stageError = $null
try { Stage-Git } catch { $stageError = $_.Exception.Message }
Assert-Equal "specific Git Bash failure" $stageError "Git stage preserves actionable reason"

Write-Host ""
if ($failures -gt 0) {
    Write-Host "FAILED: $failures assertion(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All Git Bash compatibility tests passed." -ForegroundColor Green
exit 0
