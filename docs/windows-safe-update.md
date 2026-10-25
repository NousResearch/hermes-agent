# Windows Safe Update Script

This script preserves your `.env` configuration when running `hermes update`.

## Why?

By default, `hermes update --force` overwrites the `.env` file with the default configuration, losing all your custom settings (headless mode, CDP ports, memory services, etc.).

This script:
1. Saves your `.env` before the update
2. Runs `hermes update`
3. Restores your `.env` after the update

## Usage

### Basic update
```powershell
.\scripts\windows-safe-update.ps1
```

### Force update
```powershell
.\scripts\windows-safe-update.ps1 -Force
```

## What it preserves

- `BROWSER_USE_HEADLESS`
- `BROWSER_USE_CDP_PORT`
- `BROWSER_USE_BROWSER`
- `BROWSER_USE_ARGS`
- `MEMORY_CORE_URL`
- `MEMORY_HUB_URL`
- `KNOWLEDGE_API_URL`
- `SEARXNG_URL`
- All other custom `.env` settings

## Alternative: PowerShell Alias

For even easier usage, add this to your PowerShell profile:

```powershell
function Update-Hermes {
    param([switch]$Force)
    $EnvFile = "$env:LOCALAPPDATA\hermes\hermes-agent\.env"
    $PreUpdateBackup = "$EnvFile.pre-update"
    if (Test-Path $EnvFile) {
        Copy-Item -Path $EnvFile -Destination $PreUpdateBackup -Force
    }
    if ($Force) { hermes update --force } else { hermes update }
    if (Test-Path $PreUpdateBackup) {
        Copy-Item -Path $PreUpdateBackup -Destination $EnvFile -Force
        Remove-Item -Path $PreUpdateBackup -Force -ErrorAction SilentlyContinue
    }
}
Set-Alias hermes-update Update-Hermes
```

Then just run:
```powershell
hermes-update -Force
```

## Author

Contributed by: Fabrizio (AI Systems Architect)
Date: August 23, 2026
