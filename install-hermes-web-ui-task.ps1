$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$daemonScript = Join-Path $repoRoot "run-hermes-web-ui-daemon.ps1"
$taskName = "Hermes Web UI"
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

$action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$daemonScript`""

$triggers = @(
  (New-ScheduledTaskTrigger -AtLogOn -User $currentUser),
  (New-ScheduledTaskTrigger -AtStartup)
)

$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -MultipleInstances IgnoreNew `
  -RestartCount 999 `
  -RestartInterval (New-TimeSpan -Minutes 1) `
  -StartWhenAvailable

$principal = New-ScheduledTaskPrincipal `
  -UserId $currentUser `
  -LogonType Interactive `
  -RunLevel Highest

$task = New-ScheduledTask -Action $action -Trigger $triggers -Settings $settings -Principal $principal
Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null

$webUiListening = Get-NetTCPConnection -LocalPort 8648 -State Listen -ErrorAction SilentlyContinue
if (-not $webUiListening) {
  Start-ScheduledTask -TaskName $taskName
  Write-Host "Installed and started scheduled task: $taskName"
} else {
  Write-Host "Installed scheduled task: $taskName (web UI already running, skipped immediate start)"
}
