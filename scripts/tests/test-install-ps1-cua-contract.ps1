param([string]$WorkRoot, [ValidateSet('failure', 'timeout', 'missing')][string]$Mode = 'failure')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
[IO.Directory]::CreateDirectory($WorkRoot) | Out-Null
$env:PATH = $PSHOME
$env:TEMP = $WorkRoot; $env:TMP = $WorkRoot
. (Join-Path $repoRoot 'scripts/install.ps1') -InstallDir (Join-Path $WorkRoot 'repo') -HermesHome (Join-Path $WorkRoot 'home')
$script:RealNativeCommand = ${function:Invoke-ProcessWithWallClockTimeout}
$script:ObservedTimeout = 0
function Invoke-ProcessWithWallClockTimeout {
    param($FilePath, $ArgumentList, [int]$TimeoutSec, $Label)
    $script:ObservedTimeout = $TimeoutSec
    # Substitute only the remote downloader payload at the native process
    # boundary. Run the actual owned process/deadline helper and CUA worker.
    $payload = switch ($Mode) {
        'failure' { '[Console]::Out.WriteLine("fixture-download-failed"); exit 7' }
        'timeout' { '[Console]::Out.WriteLine("fixture-download-started"); Start-Sleep -Seconds 60; exit 0' }
        'missing' { 'exit 0' }
    }
    $arguments = @($ArgumentList)
    $arguments[-1] = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($payload))
    & $script:RealNativeCommand -FilePath $FilePath -ArgumentList $arguments -TimeoutSec 8 -Label $Label
}
Install-CuaDriver
if ($script:ObservedTimeout -lt 660) { throw 'Computer Use deadline is below the upstream lock recovery window' }
[Console]::Out.WriteLine('CUA_FIXTURE_RESULT=' + (@{ timeout = $script:ObservedTimeout; completed = $true } | ConvertTo-Json -Compress))
