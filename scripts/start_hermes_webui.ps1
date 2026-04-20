#Requires -Version 5.1
<#
.SYNOPSIS
  Start the Hermes WebUI monitoring dashboard.

.DESCRIPTION
  Activates the bundled venv at $env:USERPROFILE\.hermes\hermes-webui\venv,
  exports HERMES_HOME and NO_PROXY for loopback hosts, then launches
  `python -m webui --localhost`.

  Auth is DISABLED by default (HERMES_WEBUI_NO_AUTH=1) so the dashboard is
  immediately usable on http://127.0.0.1:8643 with no token prompt. Pass
  `-RequireAuth` to re-enable token authentication. When -Lan is used, auth
  is forced ON unless `-RequireAuth:$false` is given explicitly, since binding
  to 0.0.0.0 without auth would expose the dashboard on the local network.

.PARAMETER Port
  Override the default port (8643).

.PARAMETER Lan
  Bind to 0.0.0.0 instead of 127.0.0.1 (LAN-accessible). Off by default.

.PARAMETER RequireAuth
  Re-enable the X-Hermes-Token gate. Auto-enabled when -Lan is set.

.EXAMPLE
  .\start_hermes_webui.ps1
  .\start_hermes_webui.ps1 -Port 9000
  .\start_hermes_webui.ps1 -Lan -RequireAuth
#>
[CmdletBinding()]
param(
    [int]$Port = 8643,
    [switch]$Lan,
    [switch]$RequireAuth
)

$ErrorActionPreference = "Stop"

$webuiDir = Join-Path $env:USERPROFILE ".hermes\hermes-webui"
$venvPython = Join-Path $webuiDir "venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Error "hermes-webui venv not found at $venvPython. Did you finish the install steps?"
    exit 1
}

$env:HERMES_HOME = Join-Path $env:USERPROFILE ".hermes"
$env:NO_PROXY    = "127.0.0.1,localhost"
$env:HERMES_WEBUI_PORT = "$Port"

$authOn = $RequireAuth.IsPresent -or $Lan.IsPresent
if ($authOn) {
    Remove-Item Env:HERMES_WEBUI_NO_AUTH -ErrorAction SilentlyContinue
} else {
    $env:HERMES_WEBUI_NO_AUTH = "1"
}

$bindArgs = @()
if (-not $Lan) { $bindArgs += "--localhost" }

Write-Host "[hermes-webui] HERMES_HOME = $env:HERMES_HOME"
Write-Host "[hermes-webui] starting on port $Port  (LAN=$Lan, auth=$authOn)"
if ($authOn) {
    Write-Host "[hermes-webui] auth token (paste this on the login screen):"
    $authJson = Join-Path $env:HERMES_HOME "auth.json"
    if (Test-Path $authJson) {
        $tok = (Get-Content $authJson -Raw | ConvertFrom-Json).webui_token
        Write-Host "    $tok" -ForegroundColor Cyan
    } else {
        Write-Host "    (will be generated on first run; printed by the server below)"
    }
} else {
    Write-Host "[hermes-webui] auth disabled  ->  open http://127.0.0.1:$Port and start using it" -ForegroundColor Green
}

Push-Location $webuiDir
try {
    & $venvPython -m webui @bindArgs
} finally {
    Pop-Location
}
