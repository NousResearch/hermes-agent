#Requires -Version 5.1
<#
.SYNOPSIS
  既存の World Monitor 展開から local sidecar のみ起動する（UAC 不要）。
#>
[CmdletBinding()]
param(
    [int] $Port = 46123,
    [string] $WmRoot = (Join-Path $env:LOCALAPPDATA 'Programs\World Monitor')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath (Join-Path $WmRoot 'sidecar\local-api-server.mjs'))) {
    throw "World Monitor not found at: $WmRoot"
}

$node = Join-Path $WmRoot 'sidecar\node\node.exe'
if (-not (Test-Path -LiteralPath $node)) {
    $cmd = Get-Command node -ErrorAction SilentlyContinue
    if (-not $cmd) { throw 'node.exe not found' }
    $node = $cmd.Source
}

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "Sidecar already on port $Port"
    exit 0
}

$mjs = Join-Path $WmRoot 'sidecar\local-api-server.mjs'
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $node
$psi.Arguments = "`"$mjs`""
$psi.WorkingDirectory = $WmRoot
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.Environment['LOCAL_API_RESOURCE_DIR'] = $WmRoot
$psi.Environment['LOCAL_API_PORT'] = [string] $Port
$p = [System.Diagnostics.Process]::Start($psi)
Write-Host "Sidecar started pid=$($p.Id) port=$Port root=$WmRoot"
