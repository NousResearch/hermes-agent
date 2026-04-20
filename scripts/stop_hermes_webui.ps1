#Requires -Version 5.1
<#
.SYNOPSIS
  Stop any running Hermes WebUI process (port 8643 by default).
#>
[CmdletBinding()]
param([int]$Port = 8643)

$ErrorActionPreference = "Continue"

$conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $conns) {
    Write-Host "[hermes-webui] nothing listening on port $Port"
    return
}

$pids = $conns.OwningProcess | Sort-Object -Unique
foreach ($processId in $pids) {
    $p = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($p) {
        Write-Host "[hermes-webui] stopping pid=$($p.Id) ($($p.ProcessName))"
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
}
Write-Host "[hermes-webui] stopped."
