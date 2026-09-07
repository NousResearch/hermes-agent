param([string]$WorkRoot = $env:TEMP)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$caseRoot = Join-Path $WorkRoot ('package-deadlines-' + [guid]::NewGuid().ToString('N'))
$bin = Join-Path $caseRoot 'bin'
[IO.Directory]::CreateDirectory($bin) | Out-Null
$env:DEADLINE_PACKAGE_ROOT = $caseRoot
$env:PATH = $bin + ';' + $PSHOME
$env:TEMP = $caseRoot
$env:TMP = $caseRoot
$fixture = @'
$ErrorActionPreference = 'Stop'
$name = [IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Path)
$root = $env:DEADLINE_PACKAGE_ROOT
$pids = Join-Path $root 'owned.pids'
$identities = Join-Path $root 'owned.identities'
if (Test-Path -LiteralPath $pids) {
    foreach ($identity in [IO.File]::ReadAllLines($identities)) {
        $parts = $identity.Split('|')
        $previous = Get-Process -Id ([int]$parts[0]) -ErrorAction SilentlyContinue
        if ($previous -and -not $previous.HasExited -and $previous.StartTime.ToUniversalTime().Ticks -eq [long]$parts[1]) { throw 'previous package writer survived into next attempt' }
    }
}
[IO.File]::AppendAllText((Join-Path $root 'calls.log'), $name + ' ' + ($args -join ' ') + "`n")
[Console]::Out.WriteLine('package progress: ' + $name)
if ($name -eq 'winget' -and $args -notcontains '--force') { exit -1978335189 }
if ($name -in @('winget', 'choco')) {
    [IO.File]::AppendAllText((Join-Path $root 'owned.identities'), "$PID|$((Get-Process -Id $PID).StartTime.ToUniversalTime().Ticks)`n")
    [IO.File]::AppendAllText($pids, [string]$PID + "`n")
    Start-Sleep -Seconds 60
    exit 99
}
$tool = if ($args[1] -eq 'ripgrep') { 'rg' } else { 'ffmpeg' }
[IO.File]::WriteAllText((Join-Path $root "bin/$tool.ps1"), "'fixture package'")
exit 0
'@
foreach ($name in @('winget', 'choco', 'scoop')) {
    [IO.File]::WriteAllText((Join-Path $bin "$name.ps1"), $fixture, [Text.UTF8Encoding]::new($true))
}
try {
    . (Join-Path $repoRoot 'scripts/install.ps1') -HermesHome (Join-Path $caseRoot 'home') -InstallDir (Join-Path $caseRoot 'repo')
    function Sync-EnvPath { }
    function Update-ProcessPathForPackages { }
    $script:InstallerCommandTimeouts.Packages = 8
    $Json = $true
    $frames = @(Invoke-Stage -StageDef @{ Name = 'system-packages'; Worker = 'Stage-SystemPackages' })
    if ($frames.Count -ne 1 -or -not ($frames[0] | ConvertFrom-Json).ok) { throw 'Package stage did not emit one successful final frame' }
    if (-not $script:HasRipgrep -or -not $script:HasFfmpeg) { throw 'Optional package fallback did not install both capabilities' }
    $calls = [IO.File]::ReadAllLines((Join-Path $caseRoot 'calls.log'))
    $managers = @($calls | ForEach-Object { ($_ -split ' ')[0] }) -join ','
    if ($managers -ne 'winget,winget,winget,winget,choco,choco,scoop,scoop') { throw "Unexpected package-manager sequence: $managers" }
    if ($calls[1] -notmatch '--force' -or $calls[3] -notmatch '--force') { throw 'Stale winget registration did not take the guarded force retry' }
    Write-Host 'PASS: real package stage bounds initial/force attempts and reaches Chocolatey/Scoop only after old writers exit'
} finally {
    $pids = Join-Path $caseRoot 'owned.identities'
    if (Test-Path -LiteralPath $pids) {
        foreach ($identity in [IO.File]::ReadAllLines($pids)) {
            $parts = $identity.Split('|')
            $owned = Get-Process -Id ([int]$parts[0]) -ErrorAction SilentlyContinue
            if ($owned -and $owned.StartTime.ToUniversalTime().Ticks -eq [long]$parts[1]) {
                Stop-Process -InputObject $owned -Force -ErrorAction SilentlyContinue
            }
        }
    }
}
