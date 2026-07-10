# Restart Hermes services and install boot autostart tasks.
#
# This script is intentionally local-machine oriented. It preserves the
# existing logon autostart tasks and adds separate boot-triggered tasks so a
# power cycle can bring Hermes back without relying only on the Startup folder.

[CmdletBinding()]
param(
    [switch]$NoElevate,
    [string]$LogPath = "",
    [string]$HermesHome = ""
)

$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$stamp] $Message"
}

if (-not $LogPath) {
    $LogPath = Join-Path $env:TEMP ("hermes-autostart-admin-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
}

if (-not (Test-IsAdmin) -and -not $NoElevate) {
    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`"",
        "-NoElevate",
        "-LogPath", "`"$LogPath`""
    )
    if ($HermesHome) {
        $args += @("-HermesHome", "`"$HermesHome`"")
    }

    Write-Step "Requesting administrator elevation via UAC..."
    $proc = Start-Process -FilePath "powershell.exe" -ArgumentList $args -Verb RunAs -Wait -PassThru
    Write-Step "Elevated run exited with code $($proc.ExitCode). Log: $LogPath"
    if (Test-Path -LiteralPath $LogPath) {
        Get-Content -LiteralPath $LogPath -Tail 240
    }
    exit $proc.ExitCode
}

$transcriptStarted = $false
try {
    $logDir = Split-Path -Parent $LogPath
    if ($logDir) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }
    Start-Transcript -Path $LogPath -Force | Out-Null
    $transcriptStarted = $true
} catch {
    Write-Warning "Transcript could not be started: $($_.Exception.Message)"
}

try {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
    $PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        $PythonExe = Join-Path $RepoRoot "venv\Scripts\python.exe"
    }
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        $PythonExe = (Get-Command python.exe -ErrorAction Stop | Select-Object -First 1 -ExpandProperty Source)
    }

    if (-not $HermesHome) {
        $HermesHome = Join-Path $env:USERPROFILE ".hermes"
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $HermesHome "logs") | Out-Null

    $CurrentUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $GatewayScript = Join-Path $ScriptDir "start-hermes-gateway.ps1"
    $DesktopScript = Join-Path $ScriptDir "start-hermes-desktop.ps1"
    $DashboardScript = Join-Path $ScriptDir "start-hermes-dashboard.ps1"
    $MemoryGraphScript = Join-Path $ScriptDir "start-obsidian-memory-graph-server.ps1"
    $RepoTailscaleScript = Join-Path $ScriptDir "Update-HermesTailscaleServe.ps1"
    $LineNgrokScript = "C:\Users\downl\AppData\Local\HermesWebUI\Start-HermesLineNgrok.ps1"
    $WebUiScript = "C:\Users\downl\AppData\Local\HermesWebUI\Start-HermesWebUI.ps1"
    $TailscaleScript = "C:\Users\downl\AppData\Local\HermesWebUI\Update-HermesTailscaleServe.ps1"

    foreach ($path in @($GatewayScript, $DesktopScript, $DashboardScript, $MemoryGraphScript, $LineNgrokScript, $WebUiScript)) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Required script not found: $path"
        }
    }
    if (-not (Test-Path -LiteralPath $RepoTailscaleScript)) {
        throw "Required script not found: $RepoTailscaleScript"
    }
    $hermesWebUiDir = Split-Path -Parent $TailscaleScript
    New-Item -ItemType Directory -Force -Path $hermesWebUiDir | Out-Null
    Copy-Item -LiteralPath $RepoTailscaleScript -Destination $TailscaleScript -Force

    function New-HermesTaskSettings {
        New-ScheduledTaskSettingsSet `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries `
            -StartWhenAvailable `
            -MultipleInstances IgnoreNew `
            -ExecutionTimeLimit ([TimeSpan]::Zero)
    }

    function Join-EnvPrefix {
        param([hashtable]$Env)
        $parts = @()
        foreach ($key in ($Env.Keys | Sort-Object)) {
            $value = [string]$Env[$key]
            $escapedValue = $value -replace "'", "''"
            $parts += "`$env:$key='$escapedValue'"
        }
        if ($parts.Count -eq 0) { return "" }
        return (($parts -join "; ") + "; ")
    }

    function Register-HermesBootTask {
        param(
            [Parameter(Mandatory = $true)][string]$TaskName,
            [Parameter(Mandatory = $true)][string]$Description,
            [Parameter(Mandatory = $true)][string]$PowerShellCommand,
            [Parameter(Mandatory = $true)][string]$WorkingDirectory,
            [int]$DelaySeconds = 30
        )

        $actionArgs = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command $PowerShellCommand"
        $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $actionArgs -WorkingDirectory $WorkingDirectory
        $trigger = New-ScheduledTaskTrigger -AtStartup
        if ($DelaySeconds -gt 0) {
            $trigger.Delay = "PT${DelaySeconds}S"
        }
        $principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType S4U -RunLevel Highest
        $settings = New-HermesTaskSettings

        Register-ScheduledTask `
            -TaskName $TaskName `
            -Action $action `
            -Trigger $trigger `
            -Principal $principal `
            -Settings $settings `
            -Description $Description `
            -Force | Out-Null

        Write-Step "Registered boot task: $TaskName"
    }

    function Register-HermesLogonTask {
        param(
            [Parameter(Mandatory = $true)][string]$TaskName,
            [Parameter(Mandatory = $true)][string]$Description,
            [Parameter(Mandatory = $true)][string]$PowerShellCommand,
            [Parameter(Mandatory = $true)][string]$WorkingDirectory,
            [int]$DelaySeconds = 30
        )

        $actionArgs = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command $PowerShellCommand"
        $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $actionArgs -WorkingDirectory $WorkingDirectory
        $trigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser
        if ($DelaySeconds -gt 0) {
            $trigger.Delay = "PT${DelaySeconds}S"
        }
        $principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel Limited
        $settings = New-HermesTaskSettings

        Register-ScheduledTask `
            -TaskName $TaskName `
            -Action $action `
            -Trigger $trigger `
            -Principal $principal `
            -Settings $settings `
            -Description $Description `
            -Force | Out-Null

        Write-Step "Registered logon task: $TaskName"
    }

    $envPrefix = Join-EnvPrefix @{
        HERMES_HOME = $HermesHome
    }
    $gatewayEnvPrefix = Join-EnvPrefix @{
        HERMES_HOME = $HermesHome
        HERMES_STARTUP_DELAY_SECONDS = "20"
        HERMES_GATEWAY_WINDOW_STYLE = "Hidden"
    }
    $desktopEnvPrefix = Join-EnvPrefix @{
        HERMES_HOME = $HermesHome
        HERMES_DESKTOP_HERMES_ROOT = $RepoRoot
        HERMES_DESKTOP_CWD = $RepoRoot
    }

    Register-HermesBootTask `
        -TaskName "HermesGatewayBootAutoStart" `
        -Description "Boot auto-start Hermes Gateway from restored checkout" `
        -PowerShellCommand "$gatewayEnvPrefix& '$GatewayScript'" `
        -WorkingDirectory $RepoRoot `
        -DelaySeconds 20

    Register-HermesBootTask `
        -TaskName "HermesHypuraHarnessBootAutoStart" `
        -Description "Boot auto-start Hypura Harness for Hermes" `
        -PowerShellCommand "$envPrefix& '$PythonExe' -m hermes_cli.main harness start" `
        -WorkingDirectory $RepoRoot `
        -DelaySeconds 40

    Register-HermesBootTask `
        -TaskName "HermesLineNgrokBootAutoStart" `
        -Description "Boot auto-start ngrok tunnel for Hermes LINE webhook" `
        -PowerShellCommand "$envPrefix& '$LineNgrokScript'" `
        -WorkingDirectory (Split-Path -Parent $LineNgrokScript) `
        -DelaySeconds 50

    Register-HermesBootTask `
        -TaskName "HermesMemoryGraphBootAutoStart" `
        -Description "Boot auto-start Obsidian memory-graph Go HTTP server (:8765)" `
        -PowerShellCommand "$envPrefix& '$MemoryGraphScript' -NoWatchdog" `
        -WorkingDirectory $RepoRoot `
        -DelaySeconds 55

    Register-HermesBootTask `
        -TaskName "HermesWebUIBootAutoStart" `
        -Description "Boot auto-start Hermes WebUI from restored checkout" `
        -PowerShellCommand "$envPrefix& '$WebUiScript'" `
        -WorkingDirectory "C:\Users\downl\Documents\New project\hermes-WebUI" `
        -DelaySeconds 60

    Register-HermesBootTask `
        -TaskName "HermesDashboardBootAutoStart" `
        -Description "Boot auto-start Hermes Dashboard from the canonical checkout" `
        -PowerShellCommand "$envPrefix& '$DashboardScript' -HermesRoot '$RepoRoot' -HermesHome '$HermesHome' -HostName '127.0.0.1' -Port 9120" `
        -WorkingDirectory $RepoRoot `
        -DelaySeconds 70

    Register-HermesBootTask `
        -TaskName "HermesTailscaleServeBootUpdate" `
        -Description "Boot update Tailscale Serve routes for Hermes WebUI and LINE webhook" `
        -PowerShellCommand "$envPrefix& '$TailscaleScript'" `
        -WorkingDirectory (Split-Path -Parent $TailscaleScript) `
        -DelaySeconds 80

    Register-HermesLogonTask `
        -TaskName "HermesDesktopAutoStart" `
        -Description "Logon auto-start Hermes Desktop from the canonical checkout" `
        -PowerShellCommand "$desktopEnvPrefix& '$DesktopScript' -HermesRoot '$RepoRoot' -Cwd '$RepoRoot' -HermesHome '$HermesHome'" `
        -WorkingDirectory $RepoRoot `
        -DelaySeconds 90

    Register-HermesLogonTask `
        -TaskName "HermesDashboardAutoStart" `
        -Description "Logon auto-start Hermes Dashboard from the canonical checkout" `
        -PowerShellCommand "$envPrefix& '$DashboardScript' -HermesRoot '$RepoRoot' -HermesHome '$HermesHome' -HostName '127.0.0.1' -Port 9120" `
        -WorkingDirectory $RepoRoot `
        -DelaySeconds 75

    Register-HermesLogonTask `
        -TaskName "HermesMemoryGraphAutoStart" `
        -Description "Logon auto-start Obsidian memory-graph Go HTTP server (:8765)" `
        -PowerShellCommand "$envPrefix& '$MemoryGraphScript' -NoWatchdog" `
        -WorkingDirectory $RepoRoot `
        -DelaySeconds 78

    Write-Step "Stopping current Hermes gateway..."
    try {
        & $PythonExe -m hermes_cli.main gateway stop --all
    } catch {
        Write-Warning "gateway stop failed or found no process: $($_.Exception.Message)"
    }

    Write-Step "Stopping current Hypura Harness..."
    try {
        & $PythonExe -m hermes_cli.main harness stop
    } catch {
        Write-Warning "harness stop failed or found no process: $($_.Exception.Message)"
    }

    Write-Step "Stopping current Hermes WebUI processes..."
    $webUiProcesses = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -and $_.CommandLine -match [regex]::Escape("C:\Users\downl\Documents\New project\hermes-WebUI\server.py")
    }
    foreach ($proc in $webUiProcesses) {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Step "Stopped WebUI process PID $($proc.ProcessId)"
    }

    Write-Step "Stopping current LINE ngrok tunnel processes..."
    $ngrokProcesses = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -and $_.CommandLine -match "ngrok" -and $_.CommandLine -match [regex]::Escape("127.0.0.1:8646")
    }
    foreach ($proc in $ngrokProcesses) {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Step "Stopped ngrok process PID $($proc.ProcessId)"
    }

    Start-Sleep -Seconds 4

    function Start-HermesTask {
        param(
            [Parameter(Mandatory = $true)][string]$TaskName,
            [int]$WaitSeconds = 8
        )
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
        if ($task.State -eq "Disabled") {
            Enable-ScheduledTask -TaskName $TaskName | Out-Null
        }
        Start-ScheduledTask -TaskName $TaskName
        Start-Sleep -Seconds $WaitSeconds
        $info = Get-ScheduledTaskInfo -TaskName $TaskName
        Write-Step ("Started task {0}; last result={1}; last run={2}" -f $TaskName, $info.LastTaskResult, $info.LastRunTime)
    }

    Write-Step "Starting Hermes tasks in order..."
    Start-HermesTask -TaskName "HermesGatewayAutoStart" -WaitSeconds 12
    Start-HermesTask -TaskName "HermesHypuraHarnessAutoStart" -WaitSeconds 6
    Start-HermesTask -TaskName "HermesLineNgrokAutoStart" -WaitSeconds 4
    Start-HermesTask -TaskName "HermesMemoryGraphAutoStart" -WaitSeconds 3
    Start-HermesTask -TaskName "HermesWebUINativeAutoStart" -WaitSeconds 12
    Start-HermesTask -TaskName "HermesDashboardAutoStart" -WaitSeconds 8
    Start-HermesTask -TaskName "HermesDesktopAutoStart" -WaitSeconds 8
    Start-HermesTask -TaskName "HermesTailscaleServeUpdate" -WaitSeconds 4

    Write-Step "Verification: gateway status"
    & $PythonExe -m hermes_cli.main gateway status

    Write-Step "Verification: harness status"
    & $PythonExe -m hermes_cli.main harness status

    Write-Step "Verification: WebUI health"
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8787/health" -TimeoutSec 8
        $health | ConvertTo-Json -Depth 5
    } catch {
        Write-Warning "WebUI health check failed: $($_.Exception.Message)"
    }

    Write-Step "Verification: Dashboard health"
    try {
        $dashboard = Invoke-WebRequest -Uri "http://127.0.0.1:9120/" -UseBasicParsing -TimeoutSec 8
        "Dashboard HTTP status: $($dashboard.StatusCode)"
    } catch {
        Write-Warning "Dashboard health check failed: $($_.Exception.Message)"
    }

    Write-Step "Verification: memory-graph health"
    try {
        $mg = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -TimeoutSec 8
        "Memory graph: build=$($mg.build) ok=$($mg.ok)"
    } catch {
        Write-Warning "Memory graph health check failed: $($_.Exception.Message)"
    }

    Write-Step "Verification: gateway runtime state"
    $statePath = Join-Path $HermesHome "gateway_state.json"
    if (Test-Path -LiteralPath $statePath) {
        Get-Content -LiteralPath $statePath -Raw
    }

    Write-Step "Verification: boot task triggers"
    foreach ($name in @(
        "HermesGatewayBootAutoStart",
        "HermesHypuraHarnessBootAutoStart",
        "HermesLineNgrokBootAutoStart",
        "HermesMemoryGraphBootAutoStart",
        "HermesWebUIBootAutoStart",
        "HermesDashboardBootAutoStart",
        "HermesTailscaleServeBootUpdate",
        "HermesDashboardAutoStart",
        "HermesMemoryGraphAutoStart",
        "HermesDesktopAutoStart"
    )) {
        $task = Get-ScheduledTask -TaskName $name -ErrorAction Stop
        $triggers = ($task.Triggers | ForEach-Object { $_.CimClass.CimClassName }) -join ","
        Write-Step "$name state=$($task.State) triggers=$triggers"
    }

    Write-Step "Hermes restart and boot autostart setup completed."
    exit 0
} catch {
    Write-Error $_
    exit 1
} finally {
    if ($transcriptStarted) {
        try {
            Stop-Transcript | Out-Null
        } catch {
            # ignore transcript shutdown failures
        }
    }
}
