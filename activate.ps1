# ============================================================================
# venv-style activation for the Hermes dev environment (pm-managed tools).
#
#   .\activate.ps1          (repo root; if script execution is disabled:
#   powershell -ExecutionPolicy Bypass -File .\activate.ps1, or set
#   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser once)
#
# Emits the composed pm env (PATH + tool vars) into the CURRENT session, with
# save/restore: `deactivate` undoes exactly what activation changed.
#
# Requires a completed .\setup-hermes.ps1 — it never invokes uv. It runs the
# pm store's pinned python (fallback: the repo venv) to emit the env JSON.
# ============================================================================
$ErrorActionPreference = 'Stop'

# Guard against double-sourcing: re-activating deactivates first.
if (Test-Path function:deactivate) { deactivate }

$repo = $PSScriptRoot
$machineArch = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment').PROCESSOR_ARCHITECTURE
$arch = if ($machineArch -eq 'ARM64') { 'arm64' } else { 'x64' }
$target = "win32-$arch"

$store = if ($env:HERMES_RUNTIME_DIR) { $env:HERMES_RUNTIME_DIR } else { Join-Path $HOME '.hermes/tools' }

$py = $null
$entry = Get-ChildItem -Path $store -Directory -Filter "python-*-$target" -ErrorAction SilentlyContinue |
    Sort-Object Name | Select-Object -Last 1
if ($entry -and (Test-Path (Join-Path $entry.FullName 'bin/python.exe'))) {
    $py = Join-Path $entry.FullName 'bin/python.exe'
}
if (-not $py) {
    foreach ($candidate in @(
        (Join-Path $repo 'venv/Scripts/python.exe'),
        (Join-Path $repo 'venv/bin/python.exe'))) {
        if (Test-Path $candidate) { $py = $candidate; break }
    }
}
if (-not $py) {
    Write-Error 'activate: no pm python found - run .\setup-hermes.ps1 first'
}

$envJSON = (& $py -m pm.cli env 2>$null) -join "`n"
if (-not $envJSON) {
    Write-Error 'activate: could not read pm env - run .\setup-hermes.ps1 first'
}
$composed = $envJSON | ConvertFrom-Json

# --- snapshot what we are about to change (deactivate restores this) ---
$global:_hermesKeys = @($composed.PSObject.Properties.Name)
$global:_hermesSaved = @{}
foreach ($k in $global:_hermesKeys) {
    $global:_hermesSaved[$k] = [pscustomobject]@{
        WasSet = (Test-Path "env:$k")
        Value  = [Environment]::GetEnvironmentVariable($k)
    }
}

foreach ($prop in $composed.PSObject.Properties) {
    Set-Item -Path "env:$($prop.Name)" -Value ([string]$prop.Value)
}

function global:deactivate {
    foreach ($k in $global:_hermesKeys) {
        $saved = $global:_hermesSaved[$k]
        if ($saved.WasSet) {
            Set-Item -Path "env:$k" -Value $saved.Value
        } else {
            Remove-Item -Path "env:$k" -ErrorAction SilentlyContinue
        }
    }
    $global:_hermesKeys = $null
    $global:_hermesSaved = $null
    Remove-Item function:deactivate
}
