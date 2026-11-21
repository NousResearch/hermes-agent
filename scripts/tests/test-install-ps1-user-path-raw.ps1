# install.ps1's persisted User PATH write keeps %VARS%, the value kind and
# whole entries. The installer is dot-sourced (definitions only) and its
# registry location is pointed at a scratch key under HKCU, so the real
# HKCU\Environment\Path is never read or written and nothing is broadcast.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot 'scripts\install.ps1'
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ("hermes-user-path-" + [Guid]::NewGuid().ToString('N'))
. $installScript -HermesHome (Join-Path $testRoot 'home') -InstallDir (Join-Path $testRoot 'checkout')

$script:Failures = 0
function Assert-Equal($Expected, $Actual, [string]$Label) {
    if ($Expected -ceq $Actual) { Write-Host "PASS: $Label" }
    else { Write-Host "FAIL: $Label`n  expected: [$Expected]`n  actual:   [$Actual]"; $script:Failures++ }
}

$scratch = "Software\HermesInstallTest\" + [Guid]::NewGuid().ToString('N')
$script:UserPathRegistrySubKey = $scratch
$script:Broadcasts = 0
function Send-EnvironmentChanged { $script:Broadcasts++ }
function Write-Ok([string]$msg) { }
function Write-Warn([string]$msg) { }
$Expand = [Microsoft.Win32.RegistryValueKind]::ExpandString
$Plain = [Microsoft.Win32.RegistryValueKind]::String

function Set-Scratch($Value, $Kind) {
    $key = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey($scratch)
    try { if ($null -eq $Value) { $key.DeleteValue('Path', $false) } else { $key.SetValue('Path', $Value, $Kind) } }
    finally { $key.Close() }
    $script:Broadcasts = 0
}
function Get-Scratch {
    $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($scratch)
    try {
        $value = $key.GetValue('Path', $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        "$value|$(if ($null -ne $value) { $key.GetValueKind('Path') })"
    } finally { $key.Close() }
}

$bin = Join-Path $env:TEMP ("hermes-bin-" + [Guid]::NewGuid().ToString('N'))
$leaf = Split-Path $bin -Leaf
try {
    Set-Scratch '%USERPROFILE%\probe-bin;C:\literal' $Expand
    Set-LauncherUserPath $bin
    Assert-Equal "$bin;%USERPROFILE%\probe-bin;C:\literal|ExpandString" (Get-Scratch) '%VAR% entries survive and the value stays REG_EXPAND_SZ'
    Assert-Equal 1 $script:Broadcasts 'the raw write is broadcast'

    Set-Scratch "%TEMP%\$leaf;C:\other" $Expand
    Set-LauncherUserPath $bin
    Assert-Equal "%TEMP%\$leaf;C:\other|ExpandString" (Get-Scratch) 'an entry that expands to the bin dir counts as present'
    Assert-Equal 0 $script:Broadcasts 'and nothing is written'

    Set-Scratch "${bin}2;C:\other" $Plain
    Set-LauncherUserPath $bin
    Assert-Equal "$bin;${bin}2;C:\other|String" (Get-Scratch) 'a longer entry sharing the prefix is not a match; REG_SZ stays REG_SZ'

    Set-Scratch $null $null
    Set-LauncherUserPath $bin
    Assert-Equal "$bin|ExpandString" (Get-Scratch) 'a missing value is created as exactly the bin dir'
} finally {
    [Microsoft.Win32.Registry]::CurrentUser.DeleteSubKeyTree($scratch, $false)
}

if ($script:Failures) { Write-Host "$script:Failures failure(s)"; exit 1 }
Write-Host 'all passed'
