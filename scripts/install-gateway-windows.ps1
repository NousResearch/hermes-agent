# Install Hermes Gateway as a Windows Scheduled Task (login auto-start).
# Launches ONE UAC prompt — approve it to complete install.
#
# Usage:
#   .\scripts\install-gateway-windows.ps1
#   .\scripts\install-gateway-windows.ps1 -Profile secretary
#   .\scripts\install-gateway-windows.ps1 -Force

param(
    [string]$Profile = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$RepoRoot = if ($PSScriptRoot -match "scripts$") {
    (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
    (Get-Location).Path
}

Set-Location $RepoRoot

if ($Profile) {
    $env:HERMES_PROFILE = $Profile
}

$env:HERMES_GATEWAY_INSTALL_START_NOW = "yes"
$env:HERMES_GATEWAY_INSTALL_START_ON_LOGIN = "yes"

$PythonExe = if (Test-Path ".venv\Scripts\python.exe") {
    (Resolve-Path ".venv\Scripts\python.exe").Path
} else {
    "py"
}
$PythonArgs = if ($PythonExe -eq "py") { @("-3") } else { @() }

$launcher = Join-Path $RepoRoot "scripts\install_gateway_windows_launcher.py"
$launchArgs = @($launcher, "--repo", $RepoRoot)
if ($Force) { $launchArgs += "--force" }

Write-Host "== Hermes Gateway install (Windows Scheduled Task) ==" -ForegroundColor Cyan
Write-Host "Repo: $RepoRoot"
if ($Profile) { Write-Host "Profile: $Profile" }
Write-Host ""
Write-Host 'UAC: approve the administrator prompt when it appears (click Yes).' -ForegroundColor Yellow
Write-Host ""

& $PythonExe @PythonArgs @launchArgs
if ($LASTEXITCODE -ne 0) {
    throw "Gateway install launcher failed (exit $LASTEXITCODE)"
}

$deadline = (Get-Date).AddSeconds(90)
$registered = $false

Write-Host ""
Write-Host "Waiting for Scheduled Task registration (up to 90s)..." -ForegroundColor DarkGray

while ((Get-Date) -lt $deadline) {
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    schtasks /Query /TN "Hermes_Gateway" /FO LIST | Out-Null
    $queryOk = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prevEap
    if ($queryOk) {
        $registered = $true
        break
    }
    Start-Sleep -Seconds 2
}

Write-Host ""
if ($registered) {
    Write-Host "Scheduled Task registered: Hermes_Gateway" -ForegroundColor Green
    schtasks /Query /TN "Hermes_Gateway" /FO LIST | Select-String "TaskName|Status|Next Run"
} else {
    Write-Host "Task not visible yet (UAC pending or Startup-folder fallback)." -ForegroundColor Yellow
    Write-Host "Check manually: schtasks /Query /TN Hermes_Gateway"
}

Write-Host ""
Write-Host "Gateway status:" -ForegroundColor Cyan
$statusArgs = @("-m", "hermes_cli.main", "gateway", "status")
if ($Profile) { $statusArgs = @("-m", "hermes_cli.main", "-p", $Profile, "gateway", "status") }
& $PythonExe @PythonArgs @statusArgs 2>&1

Write-Host ""
Write-Host 'Done. Login auto-start enabled; manage with: hermes gateway status / restart / stop' -ForegroundColor Green
