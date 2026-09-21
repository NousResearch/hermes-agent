# Behavioral tests for install.ps1's persisted User PATH writes.
#
# The installer is dot-sourced without running its entry point, then the
# script-scope registry location is pointed at a scratch key under HKCU.  The
# real call sites (Set-ManagedNodeFirstOnUserPath, Add-UserPathEntries,
# Set-HermesBinOnUserPath) run end to end against that key, so
# HKCU\Environment\Path is never read or written, no WM_SETTINGCHANGE is
# broadcast, and the scratch key is deleted in `finally`.

$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot 'scripts\install.ps1'
$testRoot = Join-Path $env:TEMP ("hermes-user-path-test-" + [Guid]::NewGuid().ToString('N'))
$HermesHome = Join-Path $testRoot 'home'
$InstallDir = Join-Path $HermesHome 'hermes-agent'
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

# Scratch registry location.  Everything below goes through the installer's
# own helpers, which read these two script-scope names.
$scratchSubKey = "Software\HermesInstallTest\" + [Guid]::NewGuid().ToString('N')
$script:UserPathRegistrySubKey = $scratchSubKey
$script:UserPathRegistryName = 'Path'

$script:Broadcasts = 0
function Send-EnvironmentChanged { $script:Broadcasts++ }
function Write-Info { param([string]$Message) }
function Write-Warn { param([string]$Message) }
function Write-Success { param([string]$Message) }

$ExpandString = [Microsoft.Win32.RegistryValueKind]::ExpandString
$String = [Microsoft.Win32.RegistryValueKind]::String

function Set-ScratchPath {
    param($Value, $Kind)
    $key = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey($scratchSubKey)
    try {
        if ($null -eq $Value) {
            $key.DeleteValue('Path', $false)
        } else {
            $key.SetValue('Path', $Value, $Kind)
        }
    } finally {
        $key.Close()
    }
    $script:Broadcasts = 0
}

function Get-ScratchPath {
    $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($scratchSubKey)
    try {
        $value = $key.GetValue('Path', $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        $kind = if ($null -eq $value) { $null } else { $key.GetValueKind('Path') }
        return [pscustomobject]@{ Value = $value; Kind = $kind }
    } finally {
        $key.Close()
    }
}

# An entry only the raw read can preserve.  TEMP always exists, and its
# expansion differs from the literal text.
$varEntry = '%TEMP%\hermes-user-path-probe'
$varEntryExpanded = [Environment]::ExpandEnvironmentVariables($varEntry)

try {
    Write-Host '-- helpers --'
    Set-ScratchPath $null $null
    $raw = Get-UserPathRaw
    Assert-Equal '' $raw.Value 'missing value reads as empty'
    Assert-Equal $ExpandString $raw.Kind 'missing value defaults to REG_EXPAND_SZ'

    Set-ScratchPath "$varEntry;C:\literal" $ExpandString
    $raw = Get-UserPathRaw
    Assert-Equal "$varEntry;C:\literal" $raw.Value 'raw read keeps %VARS% unexpanded'
    Assert-Equal $ExpandString $raw.Kind 'raw read reports REG_EXPAND_SZ'
    Assert-Equal $varEntryExpanded (Expand-UserPathEntry $varEntry) 'comparison form is the expanded entry'
    Assert-Equal '' (Expand-UserPathEntry '') 'empty segment expands to itself'

    Write-Host ''
    Write-Host '-- Set-ManagedNodeFirstOnUserPath --'
    $nodeDir = Join-Path $HermesHome 'node'
    Set-ScratchPath "$varEntry;C:\literal;$nodeDir" $ExpandString
    Set-ManagedNodeFirstOnUserPath $nodeDir
    $after = Get-ScratchPath
    Assert-Equal "$nodeDir;$varEntry;C:\literal" $after.Value 'node dir moves to the front, %VARS% entry survives raw'
    Assert-Equal $ExpandString $after.Kind 'REG_EXPAND_SZ is preserved'
    Assert-Equal 1 $script:Broadcasts 'a write broadcasts once'

    $script:Broadcasts = 0
    Set-ManagedNodeFirstOnUserPath $nodeDir
    Assert-Equal "$nodeDir;$varEntry;C:\literal" (Get-ScratchPath).Value 'second run is a no-op'
    Assert-Equal 0 $script:Broadcasts 'a no-op neither writes nor broadcasts'

    Set-ScratchPath "C:\literal;%HERMES_USER_PATH_TEST_NODE%;C:\tail" $ExpandString
    $env:HERMES_USER_PATH_TEST_NODE = $nodeDir
    Set-ManagedNodeFirstOnUserPath $nodeDir
    Remove-Item Env:\HERMES_USER_PATH_TEST_NODE
    Assert-Equal "$nodeDir;C:\literal;C:\tail" (Get-ScratchPath).Value 'a %VAR% spelling of the node dir is matched, not duplicated'

    Set-ScratchPath "C:\literal;C:\tail" $String
    Set-ManagedNodeFirstOnUserPath $nodeDir
    Assert-Equal $String (Get-ScratchPath).Kind 'an existing REG_SZ value stays REG_SZ'

    Write-Host ''
    Write-Host '-- Add-UserPathEntries (Install-Git) --'
    $gitEntries = @('C:\g\cmd', 'C:\g\bin', 'C:\g\usr\bin')

    Set-ScratchPath $null $null
    Add-UserPathEntries -Entries $gitEntries
    $after = Get-ScratchPath
    Assert-Equal 'C:\g\cmd;C:\g\bin;C:\g\usr\bin' $after.Value 'empty User PATH: entries stay separate'
    Assert-Equal $ExpandString $after.Kind 'a created value is REG_EXPAND_SZ'

    Set-ScratchPath 'C:\only-entry' $ExpandString
    Add-UserPathEntries -Entries $gitEntries
    Assert-Equal 'C:\only-entry;C:\g\cmd;C:\g\bin;C:\g\usr\bin' (Get-ScratchPath).Value 'single-entry User PATH: nothing is glued together'

    Set-ScratchPath "$varEntry;C:\g\bin;" $ExpandString
    Add-UserPathEntries -Entries $gitEntries
    $after = Get-ScratchPath
    Assert-Equal "$varEntry;C:\g\bin;;C:\g\cmd;C:\g\usr\bin" $after.Value 'present entry skipped, %VARS% and the empty segment kept'
    Assert-Equal $ExpandString $after.Kind 'REG_EXPAND_SZ is preserved'

    $script:Broadcasts = 0
    Add-UserPathEntries -Entries $gitEntries
    Assert-Equal 0 $script:Broadcasts 'all entries present: no write, no broadcast'

    Write-Host ''
    Write-Host '-- Set-HermesBinOnUserPath (Set-PathVariable) --'
    $hermesBinDir = Join-Path $HermesHome 'bin'
    $legacyScripts = "$InstallDir\venv\Scripts"
    $legacyBin = "$InstallDir\bin"

    Set-ScratchPath "$varEntry;$legacyScripts;C:\literal;$legacyBin" $ExpandString
    Set-HermesBinOnUserPath -HermesBin $hermesBinDir
    $after = Get-ScratchPath
    Assert-Equal "$hermesBinDir;$varEntry;C:\literal" $after.Value 'legacy entries stripped, bin prepended, %VARS% entry survives raw'
    Assert-Equal $ExpandString $after.Kind 'REG_EXPAND_SZ is preserved'

    $script:Broadcasts = 0
    Set-HermesBinOnUserPath -HermesBin $hermesBinDir
    Assert-Equal "$hermesBinDir;$varEntry;C:\literal" (Get-ScratchPath).Value 'second run is a no-op'
    Assert-Equal 0 $script:Broadcasts 'a no-op neither writes nor broadcasts'

    $env:HERMES_USER_PATH_TEST_HOME = $HermesHome
    Set-ScratchPath "C:\literal;%HERMES_USER_PATH_TEST_HOME%\bin;%HERMES_USER_PATH_TEST_HOME%\hermes-agent\venv\Scripts" $ExpandString
    Set-HermesBinOnUserPath -HermesBin $hermesBinDir
    Remove-Item Env:\HERMES_USER_PATH_TEST_HOME
    Assert-Equal "C:\literal;%HERMES_USER_PATH_TEST_HOME%\bin" (Get-ScratchPath).Value 'a %VAR% spelling is recognised: legacy stripped, bin not duplicated'
} finally {
    [Microsoft.Win32.Registry]::CurrentUser.DeleteSubKeyTree($scratchSubKey, $false)
    $parent = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Software\HermesInstallTest', $true)
    if ($parent) {
        $empty = ($parent.SubKeyCount -eq 0 -and $parent.ValueCount -eq 0)
        $parent.Close()
        if ($empty) {
            [Microsoft.Win32.Registry]::CurrentUser.DeleteSubKey('Software\HermesInstallTest', $false)
        }
    }
}

Write-Host ''
if ($script:Failures -gt 0) {
    Write-Host "$($script:Failures) assertion(s) failed"
    exit 1
}
Write-Host 'All User PATH assertions passed'
exit 0
