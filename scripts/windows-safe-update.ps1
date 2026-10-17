# Hermes Safe Update Script for Windows
# Preserves .env configuration across updates
# Usage: .\scripts\windows-safe-update.ps1 [-Force]

param([switch]$Force)

Write-Host "=== HERMES SAFE UPDATE ===" -ForegroundColor Cyan
Write-Host ""

$EnvFile = "$env:LOCALAPPDATA\hermes\hermes-agent\.env"
$PreUpdateBackup = "$EnvFile.pre-update"

# 1. Save .env
if (Test-Path $EnvFile) {
    Copy-Item -Path $EnvFile -Destination $PreUpdateBackup -Force
    Write-Host "📦 Backup .env saved" -ForegroundColor Green
}

# 2. Run update
Write-Host "🔄 Running update..." -ForegroundColor Yellow
Write-Host ""
if ($Force) {
    hermes update --force
} else {
    hermes update
}
$exitCode = $LASTEXITCODE
Write-Host ""

# 3. Restore .env
if (Test-Path $PreUpdateBackup) {
    Copy-Item -Path $PreUpdateBackup -Destination $EnvFile -Force
    Remove-Item -Path $PreUpdateBackup -Force -ErrorAction SilentlyContinue
    Write-Host "✅ Configuration restored" -ForegroundColor Green
}

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "=== UPDATE COMPLETE ===" -ForegroundColor Green
} else {
    Write-Host "=== UPDATE FAILED (exit code: $exitCode) ===" -ForegroundColor Red
}
