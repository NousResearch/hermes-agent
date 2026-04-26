# hermes-startup.ps1 - Windows Startup Script for Hermes Agent
# Based on the Coda Engine V7.1 Constitution

$HERMES_ROOT = Get-Location
Write-Host "🚀 Starting Hermes Agent (Windows Edition)..." -ForegroundColor Cyan
Write-Host "📍 Root: $HERMES_ROOT"

# Check if .venv exists
if (Test-Path "$HERMES_ROOT\.venv") {
    Write-Host "🐍 Activating virtual environment..."
    & "$HERMES_ROOT\.venv\Scripts\Activate.ps1"
} elseif (Test-Path "$HERMES_ROOT\venv") {
    Write-Host "🐍 Activating virtual environment (venv)..."
    & "$HERMES_ROOT\venv\Scripts\Activate.ps1"
} else {
    Write-Host "⚠ No virtual environment found. Running with system python." -ForegroundColor Yellow
}

# Run doctor if requested
if ($args -contains "doctor") {
    python -m hermes_cli.main doctor
} else {
    python -m hermes_cli.main chat
}
