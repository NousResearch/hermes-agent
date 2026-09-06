# Behavioral tests for install.ps1's Computer Use (cua-driver) provisioning
# gate (#104413).
#
# The installer is dot-sourced without running its entry point, then the
# upstream-installer job, command lookup and the runtime-contract probe are
# replaced with deterministic in-process stubs. This exercises the real
# Install-CuaDriver / Find-CuaDriver logic without network access, without
# installing anything, and without touching the user's profile or Hermes home.
#
# Contract under test: the cua-driver pre-install is a convenience for FRESH
# installs. Re-running install.ps1 over a completed install (bootstrap marker
# present) repairs a driver that is already there but never introduces a new
# one; -WithComputerUse forces it; the upstream install-event telemetry is off.

$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot 'scripts\install.ps1'
$testRoot = Join-Path $env:TEMP ("hermes-computer-use-test-" + [Guid]::NewGuid().ToString('N'))
$HermesHome = Join-Path $testRoot 'home'
$InstallDir = Join-Path $testRoot 'hermes-agent'
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
. $installScript -HermesHome $HermesHome -InstallDir $InstallDir

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

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
function Assert-Contains {
    param([string]$Needle, [string[]]$Haystack, [string]$Label)
    $hit = @($Haystack | Where-Object { $_ -like "*$Needle*" }).Count -gt 0
    if ($hit) {
        Write-Host "PASS: $Label"
    } else {
        Write-Host "FAIL: $Label"
        Write-Host "  expected a line containing: [$Needle]"
        Write-Host ("  got: " + ($Haystack -join ' | '))
        $script:Failures++
    }
}

# Isolate the profile so Find-CuaDriver can only see drivers this test plants.
$savedUserProfile = $env:USERPROFILE
$savedLocalAppData = $env:LOCALAPPDATA
$env:USERPROFILE = Join-Path $testRoot 'profile'
$env:LOCALAPPDATA = Join-Path $env:USERPROFILE 'AppData\Local'
New-Item -ItemType Directory -Force -Path $env:LOCALAPPDATA | Out-Null
$fakeDriver = Join-Path $env:LOCALAPPDATA 'Programs\Cua\cua-driver\bin\cua-driver.exe'
$bootstrapMarker = Join-Path $InstallDir '.hermes-bootstrap-complete'

# Controlled command surface used by the real Install-CuaDriver function.
$script:JobStarts = 0
$script:ContractReady = $false
$script:Messages = New-Object System.Collections.Generic.List[string]
$global:CuaTestJobTelemetry = $null

function Write-Info { param([string]$Message) $script:Messages.Add("INFO: $Message") }
function Write-Warn { param([string]$Message) $script:Messages.Add("WARN: $Message") }
function Write-Success { param([string]$Message) $script:Messages.Add("OK: $Message") }
function Test-CuaDriverRuntimeContract { param([string]$DriverPath) return $script:ContractReady }
function Get-Command {
    [CmdletBinding()]
    param([Parameter(Position = 0)][string]$Name)
    # Nothing on PATH; a planted driver resolves only by its full path.
    if ($Name -eq 'cua-driver') { return $null }
    if (Test-Path -LiteralPath $Name) { return [pscustomobject]@{ Source = $Name } }
    return $null
}
function Invoke-RestMethod {
    [CmdletBinding()]
    param([switch]$UseBasicParsing, [Parameter(Position = 0)][string]$Uri)
    # Stand-in for the upstream installer body: record the env the job gave it.
    return '$global:CuaTestJobTelemetry = $env:CUA_DRIVER_RS_TELEMETRY_ENABLED'
}
function Start-Job {
    [CmdletBinding()]
    param([scriptblock]$ScriptBlock)
    $script:JobStarts++
    & $ScriptBlock | Out-Null
    return [pscustomobject]@{ Id = 1 }
}
function Wait-Job { [CmdletBinding()] param([Parameter(Position = 0)]$Job, [int]$Timeout) return $Job }
function Receive-Job { [CmdletBinding()] param([Parameter(Position = 0)]$Job) }
function Remove-Job { [CmdletBinding()] param([Parameter(Position = 0)]$Job, [switch]$Force) }
function Stop-Job { [CmdletBinding()] param([Parameter(Position = 0)]$Job) }

function Invoke-InstallProbe {
    param(
        [bool]$ExistingInstall = $false,
        [bool]$DriverPlanted = $false,
        [bool]$DriverCompatible = $false
    )
    $script:JobStarts = 0
    $script:Messages.Clear()
    $script:ContractReady = $DriverCompatible
    $global:CuaTestJobTelemetry = $null
    Remove-Item -LiteralPath $bootstrapMarker -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $fakeDriver -Force -ErrorAction SilentlyContinue
    if ($ExistingInstall) { Set-Content -LiteralPath $bootstrapMarker -Value '{}' -Encoding Ascii }
    if ($DriverPlanted) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $fakeDriver) | Out-Null
        Set-Content -LiteralPath $fakeDriver -Value 'fake' -Encoding Ascii
    }
    Install-CuaDriver
    return [pscustomobject]@{
        JobStarts = $script:JobStarts
        Messages = @($script:Messages)
        Telemetry = $global:CuaTestJobTelemetry
    }
}

try {
    Write-Host '-- fresh install --'
    $r = Invoke-InstallProbe
    Assert-Equal 1 $r.JobStarts 'fresh install provisions the driver'
    Assert-Contains '.cua-driver' $r.Messages 'fresh install announces what it writes outside HERMES_HOME'
    Assert-Equal '0' $r.Telemetry 'upstream install-event telemetry is off inside the job'

    Write-Host ''
    Write-Host '-- update run over an existing install --'
    $r = Invoke-InstallProbe -ExistingInstall $true
    Assert-Equal 0 $r.JobStarts 'update run never adds a missing driver'
    Assert-Contains 'hermes computer-use install' $r.Messages 'update run says how to opt in'

    $WithComputerUse = $true
    $r = Invoke-InstallProbe -ExistingInstall $true
    Assert-Equal 1 $r.JobStarts '-WithComputerUse forces the pre-install on an update run'
    $WithComputerUse = $false

    $r = Invoke-InstallProbe -ExistingInstall $true -DriverPlanted $true -DriverCompatible $false
    Assert-Equal 1 $r.JobStarts 'update run still repairs a present-but-incompatible driver'
    Assert-Contains 'repairing' $r.Messages 'repair is announced'

    $r = Invoke-InstallProbe -ExistingInstall $true -DriverPlanted $true -DriverCompatible $true
    Assert-Equal 0 $r.JobStarts 'compatible driver off PATH is found and left alone'
    Assert-Contains 'already installed and compatible' $r.Messages 'off-PATH driver is reported as installed'

    Write-Host ''
    Write-Host '-- opt-out --'
    $SkipComputerUse = $true
    $r = Invoke-InstallProbe
    Assert-Equal 0 $r.JobStarts '-SkipComputerUse wins on a fresh install'
    $SkipComputerUse = $false
} finally {
    $env:USERPROFILE = $savedUserProfile
    $env:LOCALAPPDATA = $savedLocalAppData
    Remove-Item Env:\CUA_DRIVER_RS_TELEMETRY_ENABLED -ErrorAction SilentlyContinue
    Remove-Variable -Name CuaTestJobTelemetry -Scope Global -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $testRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($script:Failures -gt 0) {
    Write-Host ''
    Write-Host "$script:Failures assertion(s) failed"
    exit 1
}

Write-Host ''
Write-Host 'all assertions passed'
