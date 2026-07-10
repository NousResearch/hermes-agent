<#
.SYNOPSIS
  Reverse a Hermes for Excel install -- stop and unregister everything.

.DESCRIPTION
  requires elevation? NO. Removes only per-user artifacts (HKCU + %LOCALAPPDATA%).

  Reverses, in order:
    1. Stop + unregister the "Hermes_Excel_Bridge" scheduled task.
    2. Stop any running bridge launched from the install dir.
    3. Remove the Startup-folder shortcut.
    4. Remove the WEF Developer registry value + catalog manifest (sideload).
    5. Delete %LOCALAPPDATA%\hermes\excel-addin (prompts unless -Force).

  Idempotent: every step tolerates already-absent artifacts.

.PARAMETER OfficeVersion
  Office registry version segment for the sideload cleanup. Default 16.0.

.PARAMETER Force
  Delete the install dir without prompting.
#>
[CmdletBinding()]
param(
  [string] $OfficeVersion = '16.0',
  [switch] $Force
)

$ErrorActionPreference = 'Stop'

# ---- Config block ----------------------------------------------------------
$InstallDir = Join-Path $env:LOCALAPPDATA 'hermes\excel-addin'
$TaskName   = 'Hermes_Excel_Bridge'
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
# ----------------------------------------------------------------------------

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

try {
  Write-Host ''
  Write-Host "Hermes for Excel rollback" -ForegroundColor White
  Write-Host "Install: $InstallDir"
  Write-Host ''

  # 1. Scheduled task (prefer the installed register-task.ps1; fall back inline).
  Step "1. Unregister scheduled task '$TaskName'"
  $regTask = Join-Path $InstallDir 'install\register-task.ps1'
  if (-not (Test-Path -LiteralPath $regTask)) { $regTask = Join-Path $ScriptDir 'register-task.ps1' }
  if (Test-Path -LiteralPath $regTask) {
    try { & powershell -NoProfile -ExecutionPolicy Bypass -File $regTask -InstallDir $InstallDir -Unregister } catch { Write-Warning $_.Exception.Message }
  } else {
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($t) {
      try { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch {}
      Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
      Write-Host "    Removed task '$TaskName'."
    } else { Write-Host "    Task '$TaskName' not present." }
  }

  # 2. Stop any running bridge from the install dir.
  Step '2. Stop running bridge'
  $idir = $InstallDir.TrimEnd('\')
  try {
    Get-CimInstance Win32_Process -Filter "Name='node.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -and ($_.CommandLine -like "*$idir*") } |
      ForEach-Object {
        Write-Host "    killing bridge PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
      }
  } catch { Write-Warning $_.Exception.Message }

  # 3. Startup shortcut.
  Step '3. Remove Startup shortcut'
  $lnk = Join-Path ([Environment]::GetFolderPath('Startup')) 'Hermes Excel Bridge.lnk'
  if (Test-Path -LiteralPath $lnk) {
    Remove-Item -LiteralPath $lnk -Force -ErrorAction SilentlyContinue
    Write-Host "    Removed '$lnk'."
  } else { Write-Host '    No Startup shortcut present.' }

  # 4. Sideload (WEF Developer value + catalog manifest).
  Step '4. Remove developer sideload'
  $regSide = Join-Path $InstallDir 'install\register-sideload.ps1'
  if (-not (Test-Path -LiteralPath $regSide)) { $regSide = Join-Path $ScriptDir 'register-sideload.ps1' }
  if (Test-Path -LiteralPath $regSide) {
    try { & powershell -NoProfile -ExecutionPolicy Bypass -File $regSide -InstallDir $InstallDir -OfficeVersion $OfficeVersion -Unregister } catch { Write-Warning $_.Exception.Message }
  } else {
    # Inline fallback if the installed script is already gone.
    $key = "HKCU:\Software\Microsoft\Office\$OfficeVersion\WEF\Developer"
    if (Test-Path -LiteralPath $key) {
      Remove-ItemProperty -LiteralPath $key -Name 'HermesExcelAddinCatalog' -Force -ErrorAction SilentlyContinue
      Write-Host '    Removed WEF Developer value (inline).'
    }
  }

  # 5. Delete install dir.
  Step "5. Delete install dir"
  if (Test-Path -LiteralPath $InstallDir) {
    $proceed = $Force
    if (-not $proceed) {
      $answer = Read-Host "    Delete '$InstallDir' and ALL its data (uploads/exports/logs/token)? [y/N]"
      $proceed = ($answer -match '^(y|yes)$')
    }
    if ($proceed) {
      Remove-Item -LiteralPath $InstallDir -Recurse -Force -ErrorAction Stop
      Write-Host "    Deleted '$InstallDir'."
    } else {
      Write-Host '    Left install dir in place (not confirmed).'
    }
  } else {
    Write-Host '    Install dir already absent.'
  }

  # Clean the User env var regardless.
  [Environment]::SetEnvironmentVariable('HERMES_EXCEL_BRIDGE_TOKEN', $null, 'User')

  Write-Host ''
  Write-Host 'Rollback complete.' -ForegroundColor Green
  exit 0
}
catch {
  Write-Error "ROLLBACK FAILED: $($_.Exception.Message)"
  exit 1
}
