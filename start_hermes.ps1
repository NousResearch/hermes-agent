<# Hermes Agent Startup Script #>
param(
    [string]$Prompt = ""
)

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "          Hermes Agent Launcher" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

$env:HERMES_HOME = 'C:\Users\smith\AppData\Local\hermes'
$env:OBSIDIAN_VAULT_PATH = 'D:\sks-local\九天'

Write-Host "Reading configuration from config.yaml..." -ForegroundColor Yellow
Write-Host "HERMES_HOME: $($env:HERMES_HOME)"
Write-Host "OBSIDIAN_VAULT_PATH: $($env:OBSIDIAN_VAULT_PATH)"
Write-Host ""

Write-Host "Starting Hermes Agent..." -ForegroundColor Yellow
Write-Host ""

if ($Prompt) {
    python hermes -z $Prompt
} else {
    python hermes chat
}

Write-Host ""
Write-Host "Hermes Agent exited" -ForegroundColor Cyan