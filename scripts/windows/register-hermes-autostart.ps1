# Register / unregister Hermes logon autostart via Task Scheduler.
#
# Default: llama.cpp RTX3060 fallback + Hermes Gateway (gateway script also
# ensures llama if the dedicated task has not finished yet; both exit early
# when port 8080 / gateway is already up).
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/register-hermes-autostart.ps1
#   powershell ... -File scripts/windows/register-hermes-autostart.ps1 -Unregister
#   powershell ... -File scripts/windows/register-hermes-autostart.ps1 -GatewayOnly
#   powershell ... -File scripts/windows/register-hermes-autostart.ps1 -IncludeLegacyStack

[CmdletBinding()]
param(
    [switch]$Unregister,
    [switch]$GatewayOnly,
    [switch]$IncludeLegacyStack,
    [string]$LlamaTaskName = "HermesLlamaFallbackRTX3060",
    [string]$GatewayTaskName = "HermesGatewayAutoStart",
    [string]$LegacyStackTaskName = "HermesAgentStackAutoStart"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$LlamaScript = Resolve-Path (Join-Path $ScriptDir "start-llama-secretary.ps1")
$GatewayScript = Resolve-Path (Join-Path $ScriptDir "start-hermes-gateway.ps1")
$StackScript = Resolve-Path (Join-Path $ScriptDir "start-hermes-stack.ps1")

$LogonAccount = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

$StaleRunValueNames = @(
    "HermesLlamaFallbackRTX3060",
    "HermesLlamaFallbackRTX3080",
    "HermesGatewayAutoStart",
    "HermesAgentStackAutoStart"
)

$StaleScheduledTaskNames = @(
    "HermesLlamaFallbackRTX3060Watchdog",
    "HermesLlamaFallbackRTX3060",
    "HermesLlamaFallbackRTX3080"
)

$StaleStartupFiles = @(
    "HermesAgentStackAutoStart.cmd",
    "HermesGatewayAutoStart.cmd",
    "HermesLlamaFallbackRTX3080.cmd",
    "HermesLlamaFallbackRTX3060.cmd"
)

function Remove-HkcuRunEntries {
    param([string[]]$Names)

    $runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
    if (-not (Test-Path -LiteralPath $runKey)) { return @() }

    $removed = @()
    foreach ($name in $Names) {
        $existing = Get-ItemProperty -Path $runKey -Name $name -ErrorAction SilentlyContinue
        if ($null -ne $existing) {
            Remove-ItemProperty -Path $runKey -Name $name -Force
            $removed += $name
        }
    }
    return $removed
}

function Remove-StartupFolderLaunchers {
    param([string[]]$FileNames)

    $startupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
    $removed = @()
    foreach ($fileName in $FileNames) {
        $path = Join-Path $startupDir $fileName
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force
            $removed += $path
        }
    }
    return $removed
}

function Unregister-HermesScheduledTask {
    param([string]$Name)

    $existing = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if ($null -eq $existing) {
        return $false
    }
    Unregister-ScheduledTask -TaskName $Name -Confirm:$false
    return $true
}

function New-HermesTaskSettings {
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit ([TimeSpan]::Zero)
    return $settings
}

function Register-HermesScheduledTask {
    param(
        [string]$TaskName,
        [string]$Description,
        [string]$ScriptPath,
        [hashtable]$Env = @{},
        [int]$DelaySeconds = 0
    )

    $envPrefix = ""
    foreach ($key in ($Env.Keys | Sort-Object)) {
        $value = $Env[$key]
        if ($null -eq $value) { continue }
        $envPrefix += "`$env:$key='$($value -replace "'", "''")'; "
    }

    $psCommand = "$envPrefix& '$ScriptPath'"
    $argumentList = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command $psCommand"

    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argumentList -WorkingDirectory $RepoRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $LogonAccount
    if ($DelaySeconds -gt 0) {
        $trigger.Delay = "PT${DelaySeconds}S"
    }

    $principal = New-ScheduledTaskPrincipal -UserId $LogonAccount -LogonType Interactive -RunLevel Limited
    $settings = New-HermesTaskSettings

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Principal $principal `
        -Settings $settings `
        -Description $Description `
        -Force | Out-Null

    return [PSCustomObject]@{
        TaskName    = $TaskName
        ScriptPath  = $ScriptPath
        DelaySeconds = $DelaySeconds
        Env         = $Env
    }
}

if ($Unregister) {
    $removedTasks = @()
    foreach ($name in @($LlamaTaskName, $GatewayTaskName, $LegacyStackTaskName) + $StaleScheduledTaskNames) {
        if (Unregister-HermesScheduledTask -Name $name) {
            $removedTasks += $name
        }
    }

    $removedRun = Remove-HkcuRunEntries -Names $StaleRunValueNames
    $removedStartup = Remove-StartupFolderLaunchers -FileNames $StaleStartupFiles

    Write-Host "Unregistered tasks: $(if ($removedTasks) { $removedTasks -join ', ' } else { '(none)' })"
    Write-Host "Removed HKCU Run: $(if ($removedRun) { $removedRun -join ', ' } else { '(none)' })"
    foreach ($path in $removedStartup) {
        Write-Host "Removed startup launcher: $path"
    }
    exit 0
}

# Prefer Task Scheduler; remove fragile HKCU Run / Startup-folder duplicates.
$removedRun = Remove-HkcuRunEntries -Names $StaleRunValueNames
$removedStartup = Remove-StartupFolderLaunchers -FileNames $StaleStartupFiles
if ($removedRun) {
    Write-Host "Cleaned HKCU Run entries: $($removedRun -join ', ')"
}
foreach ($path in $removedStartup) {
    Write-Host "Removed legacy startup launcher: $path"
}


foreach ($staleTask in $StaleScheduledTaskNames) {
    if (Unregister-HermesScheduledTask -Name $staleTask) {
        Write-Host "Removed stale scheduled task: $staleTask"
    }
}

$registered = @()

if (-not $GatewayOnly) {
    $llamaEnv = @{}
    $dotEnv = Join-Path $env:USERPROFILE ".hermes\.env"
    if (Test-Path -LiteralPath $dotEnv) {
        Get-Content -LiteralPath $dotEnv | ForEach-Object {
            $line = $_.Trim()
            if (-not $line -or $line.StartsWith('#')) { return }
            $eq = $line.IndexOf('=')
            if ($eq -lt 1) { return }
            $key = $line.Substring(0, $eq).Trim()
            if ($key -notin @('HERMES_LLAMA_MODEL', 'HERMES_LLAMA_ALIAS', 'HERMES_LLAMA_MODEL_PATH', 'HERMES_LLAMA_SERVER_EXE', 'HERMES_LLAMA_CTX', 'HERMES_LLAMA_GPU_LAYERS')) { return }
            $value = $line.Substring($eq + 1).Trim().Trim('"').Trim("'")
            if ($value) { $llamaEnv[$key] = $value }
        }
    }

    $registered += Register-HermesScheduledTask `
        -TaskName $LlamaTaskName `
        -Description "Auto-start llama.cpp secretary (HF -hf, port 8080, 64K context) at logon" `
        -ScriptPath $LlamaScript `
        -Env $llamaEnv `
        -DelaySeconds 10
}

$registered += Register-HermesScheduledTask `
    -TaskName $GatewayTaskName `
    -Description "Auto-start Hermes Gateway at logon (llama fallback first if needed)" `
    -ScriptPath $GatewayScript `
    -Env @{
        HERMES_STARTUP_DELAY_SECONDS = "30"
        HERMES_GATEWAY_WINDOW_STYLE    = "Minimized"
    } `
    -DelaySeconds 20

if ($IncludeLegacyStack) {
    $registered += Register-HermesScheduledTask `
        -TaskName $LegacyStackTaskName `
        -Description "Legacy full Hermes stack autostart (Hypura, TUI, ngrok, ...)" `
        -ScriptPath $StackScript `
        -DelaySeconds 30
}

Write-Host ""
Write-Host "Registered Hermes autostart tasks:" -ForegroundColor Green
$registered | Format-Table -AutoSize TaskName, ScriptPath, DelaySeconds

Write-Host "Disable autostart:" -ForegroundColor Cyan
Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`" -Unregister"
Write-Host ""
Write-Host "Manual task control:" -ForegroundColor Cyan
Write-Host "  Get-ScheduledTask -TaskName '$LlamaTaskName','$GatewayTaskName' | Format-Table TaskName,State"
Write-Host "  Disable-ScheduledTask -TaskName '$GatewayTaskName'"
Write-Host "  Enable-ScheduledTask -TaskName '$GatewayTaskName'"
