# Behavioral fault-injection tests for interrupted venv transactions.
#
# The test lifts the shipped functions from install.ps1's PowerShell AST and
# executes their real marker, directory-rename, stale-sweep, and rollback logic
# against isolated temporary trees. Only process discovery, task control, uv,
# and sleeps are synthetic.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot 'scripts\install.ps1'
$tokens = $null
$parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    $installScript, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -gt 0) { throw 'Installer parse failed' }

$functionNames = @(
    'Install-Venv',
    'Get-PendingVenvBackup',
    'Restore-VenvBackup',
    'Complete-VenvTransaction'
)
foreach ($name in $functionNames) {
    $matches = @($ast.FindAll({
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq $name
    }, $true))
    if ($matches.Count -ne 1) { throw "Expected one complete function: $name" }
    Invoke-Expression $matches[0].Extent.Text
}

$testRoot = Join-Path $env:TEMP ("hermes-venv-retry-test-" + [Guid]::NewGuid().ToString('N'))
$NoVenv = $false
$script:UvCmd = 'synthetic-uv-never-executed.exe'
$script:Failures = 0
$script:FailNextVenvRename = $false
$script:FakeVenvExitCode = 0

function Write-Info { param([string]$Message) }
function Write-Warn { param([string]$Message) }
function Write-Success { param([string]$Message) }
function Resolve-AvailablePythonVersion {
    [pscustomobject]@{ Path = 'synthetic-python'; Version = '3.11' }
}
function schtasks { $global:LASTEXITCODE = 0 }
function taskkill { $global:LASTEXITCODE = 0 }
function Get-CimInstance { @() }
function Start-Sleep { }
function Rename-Item {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$LiteralPath,
        [Parameter(Mandatory = $true)][string]$NewName
    )
    if ($script:FailNextVenvRename -and (Split-Path -Leaf $LiteralPath) -eq 'venv') {
        $script:FailNextVenvRename = $false
        throw 'synthetic rename interruption'
    }
    Microsoft.PowerShell.Management\Rename-Item -LiteralPath $LiteralPath -NewName $NewName -ErrorAction Stop
}
function New-Object {
    param([string]$TypeName)
    if ($TypeName -eq 'System.Diagnostics.ProcessStartInfo') {
        return [pscustomobject]@{
            FileName = ''
            Arguments = ''
            WorkingDirectory = ''
            UseShellExecute = $false
            CreateNoWindow = $false
            RedirectStandardOutput = $false
            RedirectStandardError = $false
        }
    }
    if ($TypeName -ne 'System.Diagnostics.Process') {
        throw "Unexpected object type $TypeName"
    }
    $reader = [pscustomobject]@{}
    $reader | Add-Member ScriptMethod ReadToEndAsync {
        [pscustomobject]@{ Result = '' }
    }
    $process = [pscustomobject]@{
        StartInfo = $null
        StandardOutput = $reader
        StandardError = $reader
        ExitCode = $script:FakeVenvExitCode
    }
    $process | Add-Member ScriptMethod Start {
        $scriptsDir = Join-Path $this.StartInfo.WorkingDirectory 'venv\Scripts'
        [IO.Directory]::CreateDirectory($scriptsDir) | Out-Null
        [IO.File]::WriteAllText(
            (Join-Path $scriptsDir 'python.exe'),
            'synthetic interpreter placeholder')
        [IO.File]::WriteAllText(
            (Join-Path $this.StartInfo.WorkingDirectory 'venv\generation.txt'),
            'PARTIAL_NEW_ENV_WITHOUT_DEPENDENCIES')
        return $true
    }
    $process | Add-Member ScriptMethod WaitForExit { }
    $process | Add-Member ScriptMethod Dispose { }
    return $process
}

function Assert-Equal {
    param($Expected, $Actual, [Parameter(Mandatory = $true)][string]$Label)
    if ($Expected -ceq $Actual) {
        Write-Host "PASS: $Label"
    } else {
        Write-Host "FAIL: $Label"
        Write-Host "  expected: [$Expected]"
        Write-Host "  actual:   [$Actual]"
        $script:Failures++
    }
}

function New-TestCase {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [switch]$SeedLiveVenv
    )
    $caseDir = Join-Path $testRoot $Name
    [IO.Directory]::CreateDirectory($caseDir) | Out-Null
    if ($SeedLiveVenv) {
        $venv = Join-Path $caseDir 'venv'
        [IO.Directory]::CreateDirectory($venv) | Out-Null
        [IO.File]::WriteAllText(
            (Join-Path $venv 'generation.txt'), 'ORIGINAL_WORKING_ENV')
    }
    return $caseDir
}

function Read-LiveGeneration {
    return [IO.File]::ReadAllText((Join-Path $script:InstallDir 'venv\generation.txt'))
}

try {
    Write-Host '-- first-attempt venv failure --'
    $script:InstallDir = New-TestCase -Name 'first-failure' -SeedLiveVenv
    $script:FakeVenvExitCode = 1
    try {
        Install-Venv
        $script:Failures++
        Write-Host 'FAIL: uv failure aborts the first venv stage'
    } catch {
        Assert-Equal $true ($_.Exception.Message -like '*uv venv exited with 1*') 'uv failure aborts the first venv stage'
    } finally {
        $script:FakeVenvExitCode = 0
    }
    Assert-Equal 'ORIGINAL_WORKING_ENV' (Read-LiveGeneration) 'first-attempt failure restores the original generation'
    Assert-Equal $false (Test-Path -LiteralPath (Join-Path $script:InstallDir 'venv.pending-backup')) 'first-attempt rollback clears the pending marker'

    Write-Host ''
    Write-Host '-- retry before dependency validation --'
    $script:InstallDir = New-TestCase -Name 'retry' -SeedLiveVenv
    Install-Venv
    $firstBackup = Get-PendingVenvBackup
    Assert-Equal $true ([bool]$firstBackup) 'the first venv stage publishes a rollback marker'
    Assert-Equal $true (Test-Path -LiteralPath (Join-Path $script:InstallDir "$firstBackup\generation.txt")) 'the first stage preserves the original generation'

    Install-Venv
    $retryBackup = Get-PendingVenvBackup
    Assert-Equal $true ([bool]$retryBackup) 'the retry keeps a rollback marker'
    Assert-Equal $true (Test-Path -LiteralPath (Join-Path $script:InstallDir "$retryBackup\generation.txt")) 'the retry keeps the original rollback generation'
    Assert-Equal 'ORIGINAL_WORKING_ENV' ([IO.File]::ReadAllText((Join-Path $script:InstallDir "$retryBackup\generation.txt"))) 'the retry marker still names the original generation'
    Restore-VenvBackup
    Assert-Equal 'ORIGINAL_WORKING_ENV' (Read-LiveGeneration) 'dependency failure after retry restores the original generation'

    Write-Host ''
    Write-Host '-- interruption before the original rename --'
    $script:InstallDir = New-TestCase -Name 'before-rename' -SeedLiveVenv
    $orphanMarkerName = 'venv.stale.pre-rename'
    Set-Content -LiteralPath (Join-Path $script:InstallDir 'venv.pending-backup') -Value $orphanMarkerName -Encoding ascii
    Install-Venv
    Restore-VenvBackup
    Assert-Equal 'ORIGINAL_WORKING_ENV' (Read-LiveGeneration) 'a marker published before a missing rename does not replace the live original'

    Write-Host ''
    Write-Host '-- rename failure after marker publication --'
    $script:InstallDir = New-TestCase -Name 'rename-failure' -SeedLiveVenv
    $script:FailNextVenvRename = $true
    try {
        Install-Venv
        $script:Failures++
        Write-Host 'FAIL: the rename failure aborts the venv stage'
    } catch {
        Assert-Equal $true ($_.Exception.Message -like '*previous install was left intact*') 'the rename failure aborts the venv stage'
    }
    Assert-Equal 'ORIGINAL_WORKING_ENV' (Read-LiveGeneration) 'rename failure leaves the live original intact'
    Assert-Equal $false (Test-Path -LiteralPath (Join-Path $script:InstallDir 'venv.pending-backup')) 'rename failure clears its unpublished rollback marker'

    Write-Host ''
    Write-Host '-- interruption after the original rename --'
    $script:InstallDir = New-TestCase -Name 'after-rename'
    $parkedOriginal = 'venv.stale.after-rename'
    $parkedPath = Join-Path $script:InstallDir $parkedOriginal
    [IO.Directory]::CreateDirectory($parkedPath) | Out-Null
    [IO.File]::WriteAllText((Join-Path $parkedPath 'generation.txt'), 'ORIGINAL_WORKING_ENV')
    Set-Content -LiteralPath (Join-Path $script:InstallDir 'venv.pending-backup') -Value $parkedOriginal -Encoding ascii
    Install-Venv
    Restore-VenvBackup
    Assert-Equal 'ORIGINAL_WORKING_ENV' (Read-LiveGeneration) 'a retry after rename-before-create restores the original generation'
} finally {
    if (Test-Path -LiteralPath $testRoot) {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $testRoot -Recurse -Force
    }
}

if ($script:Failures -gt 0) {
    Write-Host ''
    Write-Host "$script:Failures assertion(s) failed"
    exit 1
}

Write-Host ''
Write-Host 'all assertions passed'
