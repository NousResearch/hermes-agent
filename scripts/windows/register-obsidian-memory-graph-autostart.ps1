# Register logon autostart for Obsidian memory-graph HTTP server (Quest VR LAN).
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/register-obsidian-memory-graph-autostart.ps1
#   ... -Unregister

[CmdletBinding()]
param(
    [switch]$Unregister,
    [string]$TaskName = "HermesObsidianMemoryGraphServer",
    [int]$Port = 8765,
    [int]$DelaySeconds = 45
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$ServerScript = Resolve-Path (Join-Path $ScriptDir "start-obsidian-memory-graph-server.ps1")
$LogonAccount = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

function Unregister-Task {
    param([string]$Name)
    $existing = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if ($null -eq $existing) { return $false }
    Unregister-ScheduledTask -TaskName $Name -Confirm:$false
    return $true
}

if ($Unregister) {
    if (Unregister-Task -Name $TaskName) {
        Write-Host "Unregistered scheduled task: $TaskName"
    } else {
        Write-Host "Task not found: $TaskName"
    }
    exit 0
}

$envPrefix = "`$env:HERMES_MEMORY_GRAPH_PORT='$Port'; "
$psCommand = "$envPrefix& '$ServerScript' -Port $Port"
$argumentList = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command $psCommand"

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argumentList -WorkingDirectory $RepoRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $LogonAccount
if ($DelaySeconds -gt 0) {
    $trigger.Delay = "PT${DelaySeconds}S"
}

$principal = New-ScheduledTaskPrincipal -UserId $LogonAccount -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Hermes Obsidian memory graph Go HTTP server for Quest/VIVE VR (port $Port)" `
    -Force | Out-Null

Write-Host "Registered logon task: $TaskName (delay ${DelaySeconds}s, port $Port)"
Write-Host "Manual start: powershell -File scripts/windows/start-obsidian-memory-graph-server.ps1"
