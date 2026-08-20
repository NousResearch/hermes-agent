# Behavioral test for install.ps1 Python runtime validation.
#
# Run on Windows PowerShell 5.1:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/ci/test_install_ps1_python_runtime_validation.ps1
#
# This parses install.ps1 and executes the shipped runtime-resolution helpers.
# The fake managed interpreter passes --version but fails native stdlib imports
# with the Windows Application Control error. The fake system interpreter works.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$installPs1 = Join-Path (Join-Path $PSScriptRoot "..") "install.ps1" | Resolve-Path
$parseErrors = $null
$tokens = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $installPs1, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -gt 0) {
    throw "install.ps1 has PowerShell parse errors: $($parseErrors -join '; ')"
}

foreach ($functionName in @(
    "Test-HermesPythonRuntime",
    "Find-UsablePythonForVersion",
    "Resolve-AvailablePythonVersion"
)) {
    $fn = $ast.Find({
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq $functionName
    }, $true)
    if (-not $fn) {
        throw "$functionName not found in $installPs1"
    }
    Invoke-Expression $fn.Extent.Text
}

$script:Failures = 0
$script:Warnings = @()

function Write-Warn {
    param([string]$Message)
    $script:Warnings += $Message
}

function Assert-Equal {
    param($Expected, $Actual, [string]$Name)
    if ($Expected -ceq $Actual) {
        Write-Host "  PASS  $Name"
    } else {
        Write-Host "  FAIL  $Name"
        Write-Host "        expected: [$Expected]"
        Write-Host "        actual:   [$Actual]"
        $script:Failures++
    }
}

function Assert-True {
    param($Condition, [string]$Name)
    Assert-Equal $true ([bool]$Condition) $Name
}

function Assert-Match {
    param([string]$Pattern, [string]$Actual, [string]$Name)
    Assert-True ($Actual -match $Pattern) $Name
}

$fixtureRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    "Hermes Runtime Validation " + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $fixtureRoot | Out-Null

try {
    $script:BadPython = Join-Path $fixtureRoot "blocked python.cmd"
    $script:GoodPython = Join-Path $fixtureRoot "trusted python.cmd"

    @'
@echo off
if "%~1"=="--version" (
  echo Python 3.11.15
  exit /b 0
)
if "%~1"=="-c" (
  1>&2 echo ImportError: DLL load failed while importing select: An Application Control policy has blocked this file.
  exit /b 1
)
exit /b 2
'@ | Set-Content -LiteralPath $script:BadPython -Encoding ASCII

    @'
@echo off
if "%~1"=="--version" (
  echo Python 3.11.9
  exit /b 0
)
if "%~1"=="-c" exit /b 0
exit /b 2
'@ | Set-Content -LiteralPath $script:GoodPython -Encoding ASCII

    $script:UvCalls = @()
    function Invoke-FakeUv {
        param(
            [Parameter(ValueFromRemainingArguments = $true)]
            [string[]]$CommandArgs
        )
        $script:UvCalls += ($CommandArgs -join " ")
        $global:LASTEXITCODE = 0
        if ($CommandArgs -contains "--system" -and
            $CommandArgs -contains "--no-managed-python") {
            return $script:GoodPython
        }
        return $script:BadPython
    }

    $script:UvCmd = "Invoke-FakeUv"
    $script:PythonVersion = "3.11"
    $script:PythonFallbackVersions = @("3.12", "3.13", "3.10")

    $version = & $script:BadPython --version
    Assert-Equal 0 $LASTEXITCODE "blocked runtime still passes --version"
    Assert-Match "Python 3\.11" ([string]$version) "blocked runtime reports its version"

    $runtimeWorks = Test-HermesPythonRuntime -PythonPath $script:BadPython
    Assert-Equal $false $runtimeWorks "native-module probe rejects blocked runtime"
    Assert-Match "Application Control policy has blocked this file" `
        $script:PythonRuntimeProbeOutput `
        "underlying Application Control error is preserved"

    $selected = Find-UsablePythonForVersion -Version "3.11"
    Assert-Equal $script:GoodPython $selected "trusted system runtime is selected"
    Assert-True ($selected -match " ") "selected exact interpreter path includes spaces"
    Assert-Equal "python find 3.11" $script:UvCalls[0] "normal uv candidate is tried first"
    Assert-Equal "python find --system --no-managed-python 3.11" $script:UvCalls[1] `
        "system-only uv fallback excludes managed Python"
    Assert-True (($script:Warnings -join "`n") -match [regex]::Escape($script:BadPython)) `
        "rejected interpreter path is reported"

    $script:UvCalls = @()
    $resolvedVersion = Resolve-AvailablePythonVersion
    Assert-Equal "3.11" $resolvedVersion "requested Python version remains preferred"
    Assert-Equal $script:GoodPython $script:ResolvedPythonPath `
        "resolver retains the exact validated interpreter path"
} finally {
    Remove-Item -LiteralPath $fixtureRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($script:Failures -gt 0) {
    Write-Host ""
    Write-Host "$($script:Failures) assertion(s) failed"
    exit 1
}

Write-Host ""
Write-Host "all assertions passed"
