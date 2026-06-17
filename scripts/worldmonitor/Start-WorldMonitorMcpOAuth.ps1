#Requires -Version 5.1
<#
.SYNOPSIS
  World Monitor MCP OAuth をエージェント環境から実行（ブラウザ自動起動・10分待機）。
#>
param(
    [int] $TimeoutSeconds = 600
)

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$log = Join-Path $env:LOCALAPPDATA 'WorldMonitorInstall\mcp-oauth-login.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

Write-Host "World Monitor MCP OAuth — browser will open. Sign in with Pro account."
Write-Host "Log: $log"

Push-Location $repo
try {
    py -3 (Join-Path $PSScriptRoot 'run_mcp_oauth_login.py') --timeout $TimeoutSeconds 2>&1 |
        Tee-Object -FilePath $log
    $exit = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($exit -eq 0) {
    py -3 -m hermes_cli.main mcp test worldmonitor
}
exit $exit
