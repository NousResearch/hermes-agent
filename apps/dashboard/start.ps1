# HERMES//HUB — one-click launcher (Windows).
# Right-click → "Run with PowerShell", or run:  .\start.ps1
# Pulls the latest build, opens the dashboard, and starts the local server.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Updating to the latest build..." -ForegroundColor Cyan
try { git pull origin main } catch { Write-Host "(skipped git pull: $_)" -ForegroundColor Yellow }

# Open the browser a moment after the server comes up.
Start-Job { Start-Sleep 2; Start-Process "http://127.0.0.1:8787" } | Out-Null

Write-Host "Starting HERMES//HUB on http://127.0.0.1:8787  (Ctrl+C to stop)" -ForegroundColor Green
python server.py
