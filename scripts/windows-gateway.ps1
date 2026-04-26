[CmdletBinding()]
param(
  [ValidateSet("help", "install", "uninstall", "start", "stop", "restart", "status", "logs", "errors", "follow", "tail")]
  [string]$Action = "status",

  [string]$TaskName = "Hermes Gateway",

  [int]$Limit = 200,

  [string]$WorkingDirectory = ""
)

$ErrorActionPreference = "Stop"

function Write-Info {
  param([string]$Message)
  Write-Host "[hermes-gateway] $Message"
}

function Get-HermesHome {
  if ($env:HERMES_HOME -and $env:HERMES_HOME.Trim().Length -gt 0) {
    return $env:HERMES_HOME
  }
  return Join-Path $env:USERPROFILE ".hermes"
}

function Get-HermesLogsDir {
  return Join-Path (Get-HermesHome) "logs"
}

function Get-AgentLogPath {
  return Join-Path (Get-HermesLogsDir) "agent.log"
}

function Get-ErrorLogPath {
  return Join-Path (Get-HermesLogsDir) "errors.log"
}

function Get-WrapperLogPath {
  return Join-Path (Get-HermesLogsDir) ("gateway-hidden-{0}.log" -f (Get-Date -Format "yyyy-MM-dd"))
}

function Get-PidPath {
  return Join-Path (Get-HermesHome) "gateway.pid"
}

function Get-StatePath {
  return Join-Path (Get-HermesHome) "gateway_state.json"
}

function Get-RunnerPath {
  return Join-Path (Get-HermesHome) "start-gateway-hidden.ps1"
}

function Get-VbsPath {
  return Join-Path (Get-HermesHome) "start-gateway-hidden.vbs"
}

function Get-PowerShellExe {
  return Join-Path $env:WINDIR "System32\WindowsPowerShell\v1.0\powershell.exe"
}

function Get-WscriptExe {
  return Join-Path $env:WINDIR "System32\wscript.exe"
}

function Assert-HermesCommand {
  $command = Get-Command hermes -ErrorAction SilentlyContinue
  if (-not $command) {
    throw "Cannot find 'hermes' in PATH. Open a new PowerShell or check the Hermes installation."
  }
  return $command.Source
}

function Resolve-WorkingDirectory {
  param([string]$Requested)

  if ($Requested -and (Test-Path -LiteralPath $Requested)) {
    return (Resolve-Path -LiteralPath $Requested).Path
  }

  $homeInstallDir = Join-Path (Get-HermesHome) "hermes-agent"
  if (Test-Path -LiteralPath $homeInstallDir) {
    return (Resolve-Path -LiteralPath $homeInstallDir).Path
  }

  if ($PSScriptRoot) {
    $repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..") -ErrorAction SilentlyContinue
    if ($repoRoot) {
      return $repoRoot.Path
    }
  }

  return $env:USERPROFILE
}

function Read-JsonFile {
  param([string]$Path)

  if (-not (Test-Path -LiteralPath $Path)) {
    return $null
  }

  $raw = Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue
  if (-not $raw -or $raw.Trim().Length -eq 0) {
    return $null
  }

  try {
    return ($raw | ConvertFrom-Json)
  } catch {
    return $null
  }
}

function Get-GatewayPidRecord {
  $path = Get-PidPath
  if (-not (Test-Path -LiteralPath $path)) {
    return $null
  }

  $raw = Get-Content -LiteralPath $path -Raw -ErrorAction SilentlyContinue
  if (-not $raw -or $raw.Trim().Length -eq 0) {
    return $null
  }

  try {
    $json = $raw | ConvertFrom-Json
    if ($json -is [int]) {
      return [pscustomobject]@{ pid = $json; kind = $null; argv = @() }
    }
    return $json
  } catch {
    $trimmed = $raw.Trim()
    $parsedPid = 0
    if ([int]::TryParse($trimmed, [ref]$parsedPid)) {
      return [pscustomobject]@{ pid = $parsedPid; kind = $null; argv = @() }
    }
  }

  return $null
}

function Test-RecordLooksLikeGateway {
  param($Record)

  if (-not $Record) {
    return $false
  }

  if ($Record.kind -and $Record.kind -ne "hermes-gateway") {
    return $false
  }

  $argvText = ""
  if ($Record.argv) {
    $argvText = ($Record.argv | ForEach-Object { $_.ToString() }) -join " "
  }

  if ($argvText.Trim().Length -eq 0) {
    return $true
  }

  return ($argvText -match "hermes" -and $argvText -match "gateway" -and $argvText -match "run")
}

function Get-GatewayProcessInfo {
  $record = Get-GatewayPidRecord
  if (-not (Test-RecordLooksLikeGateway -Record $record)) {
    return $null
  }

  $gatewayPid = [int]$record.pid
  if ($gatewayPid -le 0) {
    return $null
  }

  $process = Get-Process -Id $gatewayPid -ErrorAction SilentlyContinue
  if (-not $process) {
    return $null
  }

  $cim = Get-CimInstance Win32_Process -Filter "ProcessId=$gatewayPid" -ErrorAction SilentlyContinue
  return [pscustomobject]@{
    Pid = $gatewayPid
    ProcessName = $process.ProcessName
    StartTime = $process.StartTime
    Path = $process.Path
    ParentPid = if ($cim) { $cim.ParentProcessId } else { $null }
    CommandLine = if ($cim) { $cim.CommandLine } else { $null }
  }
}

function Test-GatewayRunning {
  return [bool](Get-GatewayProcessInfo)
}

function Remove-StalePidFile {
  if (-not (Test-GatewayRunning)) {
    $pidPath = Get-PidPath
    if (Test-Path -LiteralPath $pidPath) {
      Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    }
  }
}

function Write-GeneratedRunner {
  param(
    [string]$HermesExe,
    [string]$WorkDir
  )

  $hermesHome = Get-HermesHome
  New-Item -ItemType Directory -Force -Path $hermesHome | Out-Null
  New-Item -ItemType Directory -Force -Path (Get-HermesLogsDir) | Out-Null

  $runnerPath = Get-RunnerPath
  $escapedHermes = $HermesExe.Replace("'", "''")
  $escapedWorkDir = $WorkDir.Replace("'", "''")
  $escapedHermesHome = $hermesHome.Replace("'", "''")
  $content = @"
`$ErrorActionPreference = "Stop"
`$hermesHome = '$escapedHermesHome'
if (-not `$env:HERMES_HOME) {
  [Environment]::SetEnvironmentVariable("HERMES_HOME", `$hermesHome, "Process")
}

function Import-HermesRuntimeEnv {
  `$envFile = Join-Path `$hermesHome ".env"
  if (-not (Test-Path -LiteralPath `$envFile)) { return }

  `$allowed = @(
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "PYTHONUTF8", "PYTHONIOENCODING",
    "HERMES_API_TIMEOUT", "HERMES_STREAM_READ_TIMEOUT"
  )

  Get-Content -LiteralPath `$envFile | ForEach-Object {
    if (`$_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
      `$key = `$Matches[1]
      if (`$allowed -contains `$key) {
        [Environment]::SetEnvironmentVariable(`$key, `$Matches[2], "Process")
      }
    }
  }
}

`$env:PYTHONUTF8 = if (`$env:PYTHONUTF8) { `$env:PYTHONUTF8 } else { "1" }
`$env:PYTHONIOENCODING = if (`$env:PYTHONIOENCODING) { `$env:PYTHONIOENCODING } else { "utf-8" }
Import-HermesRuntimeEnv

`$logDir = Join-Path `$hermesHome "logs"
New-Item -ItemType Directory -Force -Path `$logDir | Out-Null
`$wrapperLog = Join-Path `$logDir ("gateway-hidden-{0}.log" -f (Get-Date -Format "yyyy-MM-dd"))
Set-Location -LiteralPath '$escapedWorkDir'

"[{0}] starting hermes gateway hidden" -f (Get-Date -Format o) | Add-Content -LiteralPath `$wrapperLog -Encoding UTF8
& '$escapedHermes' gateway run --replace --accept-hooks *>> `$wrapperLog
`$exitCode = `$LASTEXITCODE
"[{0}] hermes gateway exited code {1}" -f (Get-Date -Format o), `$exitCode | Add-Content -LiteralPath `$wrapperLog -Encoding UTF8
exit `$exitCode
"@

  Set-Content -LiteralPath $runnerPath -Value $content -Encoding UTF8
  return $runnerPath
}

function Write-GeneratedVbs {
  $vbsPath = Get-VbsPath
  $psExe = Get-PowerShellExe
  $runnerPath = Get-RunnerPath

  $content = @"
Set shell = CreateObject("WScript.Shell")
quote = Chr(34)
cmd = quote & "$psExe" & quote & " -NoProfile -ExecutionPolicy Bypass -File " & quote & "$runnerPath" & quote
shell.Run cmd, 0, True
"@
  Set-Content -LiteralPath $vbsPath -Value $content -Encoding ASCII
  return $vbsPath
}

function Install-HermesGatewayTask {
  $hermesExe = Assert-HermesCommand
  $workDir = Resolve-WorkingDirectory -Requested $WorkingDirectory
  $runnerPath = Write-GeneratedRunner -HermesExe $hermesExe -WorkDir $workDir
  $vbsPath = Write-GeneratedVbs

  $taskAction = New-ScheduledTaskAction -Execute (Get-WscriptExe) -Argument "`"$vbsPath`""
  $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
  $trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
  $principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
  $settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Days 30) `
    -MultipleInstances IgnoreNew `
    -RestartCount 10 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable

  try {
    Register-ScheduledTask `
      -TaskName $TaskName `
      -Action $taskAction `
      -Trigger $trigger `
      -Principal $principal `
      -Settings $settings `
      -Description "Run Hermes gateway hidden in the background for the current Windows user." `
      -Force | Out-Null
  } catch {
    throw "Could not register scheduled task '$TaskName'. Try an Administrator terminal if Windows returns Access denied. Original error: $($_.Exception.Message)"
  }

  Write-Info "Installed scheduled task '$TaskName'."
  Write-Info "Hidden launcher: $vbsPath"
  Write-Info "Runner script   : $runnerPath"
  Write-Info "Hermes command  : $hermesExe"
  Write-Info "Working dir     : $workDir"
}

function Uninstall-HermesGatewayTask {
  $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if ($task) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Info "Uninstalled scheduled task '$TaskName'."
  } else {
    Write-Info "Scheduled task '$TaskName' is not installed."
  }
}

function Start-HermesGateway {
  if (Test-GatewayRunning) {
    $info = Get-GatewayProcessInfo
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task -and $task.State -eq "Running") {
      Write-Info "Gateway is already running through scheduled task '$TaskName'. PID: $($info.Pid)"
    } else {
      Write-Info "Gateway is already running outside scheduled task '$TaskName'. PID: $($info.Pid)"
      Write-Info "Run restart, or close the foreground Hermes terminal once and then run start."
    }
    return
  }

  Remove-StalePidFile

  $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $task) {
    Install-HermesGatewayTask
  }

  Start-ScheduledTask -TaskName $TaskName
  Write-Info "Start requested through scheduled task '$TaskName'."
}

function Stop-HermesGateway {
  $info = Get-GatewayProcessInfo

  $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if ($task -and $task.State -eq "Running") {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  }

  if ($info) {
    Write-Info "Stopping gateway PID $($info.Pid)."
    Stop-Process -Id $info.Pid -Force -ErrorAction SilentlyContinue

    for ($i = 0; $i -lt 20; $i++) {
      if (-not (Get-Process -Id $info.Pid -ErrorAction SilentlyContinue)) {
        break
      }
      Start-Sleep -Milliseconds 300
    }

    if (Get-Process -Id $info.Pid -ErrorAction SilentlyContinue) {
      $taskkill = Join-Path $env:WINDIR "System32\taskkill.exe"
      & $taskkill /PID $info.Pid /T /F | Out-Null
      for ($i = 0; $i -lt 20; $i++) {
        if (-not (Get-Process -Id $info.Pid -ErrorAction SilentlyContinue)) {
          break
        }
        Start-Sleep -Milliseconds 300
      }
    }

    if ((Get-Process -Id $info.Pid -ErrorAction SilentlyContinue) -and $info.ParentPid) {
      $parent = Get-Process -Id $info.ParentPid -ErrorAction SilentlyContinue
      if ($parent -and $parent.ProcessName -like "hermes*") {
        Write-Info "Stopping parent Hermes launcher PID $($parent.Id)."
        Stop-Process -Id $parent.Id -Force -ErrorAction SilentlyContinue
        for ($i = 0; $i -lt 20; $i++) {
          if (-not (Get-Process -Id $info.Pid -ErrorAction SilentlyContinue)) {
            break
          }
          Start-Sleep -Milliseconds 300
        }

        if (Get-Process -Id $info.Pid -ErrorAction SilentlyContinue) {
          & $taskkill /PID $parent.Id /T /F | Out-Null
          for ($i = 0; $i -lt 20; $i++) {
            if (-not (Get-Process -Id $info.Pid -ErrorAction SilentlyContinue)) {
              break
            }
            Start-Sleep -Milliseconds 300
          }
        }
      }
    }

    if (Get-Process -Id $info.Pid -ErrorAction SilentlyContinue) {
      throw "Could not stop gateway PID $($info.Pid). Close the foreground Hermes terminal once, or run this script from the same/elevated permission level, then run restart again."
    }
  } else {
    Write-Info "No running gateway PID found for this Hermes profile."
  }

  Remove-StalePidFile
}

function Restart-HermesGateway {
  Stop-HermesGateway
  Start-Sleep -Seconds 2
  Start-HermesGateway
}

function Show-HermesGatewayStatus {
  $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  $info = Get-GatewayProcessInfo
  $state = Read-JsonFile -Path (Get-StatePath)
  $runtimeSource = "not running"
  if ($info -and $task -and $task.State -eq "Running") {
    $runtimeSource = "scheduled task"
  } elseif ($info) {
    $runtimeSource = "external or foreground process"
  }

  Write-Host ""
  Write-Host "Task name       : $TaskName"
  Write-Host "Task state      : $(if ($task) { $task.State } else { 'not installed' })"
  Write-Host "Runtime source  : $runtimeSource"
  Write-Host "Hermes home     : $(Get-HermesHome)"
  Write-Host "Gateway PID     : $(if ($info) { $info.Pid } else { 'not running' })"
  Write-Host "Process         : $(if ($info) { $info.ProcessName } else { '-' })"
  Write-Host "Started         : $(if ($info) { $info.StartTime } else { '-' })"
  Write-Host "Gateway state   : $(if ($state -and $state.gateway_state) { $state.gateway_state } else { 'unknown' })"
  Write-Host "Agent log       : $(Get-AgentLogPath)"
  Write-Host "Error log       : $(Get-ErrorLogPath)"
  Write-Host "Wrapper log     : $(Get-WrapperLogPath)"

  if ($state -and $state.platforms) {
    Write-Host ""
    Write-Host "Platforms:"
    $state.platforms.PSObject.Properties | ForEach-Object {
      $platform = $_.Name
      $payload = $_.Value
      Write-Host ("  {0}: {1} {2}" -f $platform, $payload.state, $(if ($payload.error_message) { "- $($payload.error_message)" } else { "" }))
    }
  }

  Write-Host ""
}

function Show-Log {
  param(
    [string]$Path,
    [switch]$Wait
  )

  if (-not (Test-Path -LiteralPath $Path)) {
    Write-Info "Log file does not exist yet: $Path"
    return
  }

  if ($Wait) {
    Get-Content -LiteralPath $Path -Tail $Limit -Wait
  } else {
    Get-Content -LiteralPath $Path -Tail $Limit
  }
}

function Show-Help {
  Write-Host @"
Hermes Gateway hidden launcher for native Windows.

Usage:
  powershell -ExecutionPolicy Bypass -File .\scripts\windows-gateway.ps1 -Action install
  powershell -ExecutionPolicy Bypass -File .\scripts\windows-gateway.ps1 -Action restart
  powershell -ExecutionPolicy Bypass -File .\scripts\windows-gateway.ps1 -Action status
  powershell -ExecutionPolicy Bypass -File .\scripts\windows-gateway.ps1 -Action logs
  powershell -ExecutionPolicy Bypass -File .\scripts\windows-gateway.ps1 -Action errors
  powershell -ExecutionPolicy Bypass -File .\scripts\windows-gateway.ps1 -Action follow
  powershell -ExecutionPolicy Bypass -File .\scripts\windows-gateway.ps1 -Action stop
  powershell -ExecutionPolicy Bypass -File .\scripts\windows-gateway.ps1 -Action uninstall

Notes:
  - Hermes has no fixed local gateway port to probe.
  - Status uses HERMES_HOME/gateway.pid plus the real Windows process.
  - Hidden startup runs: hermes gateway run --replace --accept-hooks.
  - Hermes internal logs are under HERMES_HOME/logs/.
"@
}

switch ($Action) {
  "help" { Show-Help }
  "install" { Install-HermesGatewayTask }
  "uninstall" { Uninstall-HermesGatewayTask }
  "start" { Start-HermesGateway }
  "stop" { Stop-HermesGateway }
  "restart" { Restart-HermesGateway }
  "status" { Show-HermesGatewayStatus }
  "logs" { Show-Log -Path (Get-AgentLogPath) }
  "errors" { Show-Log -Path (Get-ErrorLogPath) }
  "follow" { Show-Log -Path (Get-AgentLogPath) -Wait }
  "tail" { Show-Log -Path (Get-WrapperLogPath) }
}
