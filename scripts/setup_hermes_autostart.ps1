param(
    [string]$Distro = "",
    [string]$RepoPath = "",
    [ValidateSet("gateway", "headless", "dashboard", "webui", "both")]
    [string]$Mode = "gateway",
    [string]$TaskName = "HermesAgentAutoStart"
)

$ErrorActionPreference = "Stop"

function Get-WslDistros {
    $raw = & wsl.exe -l -q 2>$null
    if (-not $raw) {
        throw "No WSL distros were found. Install WSL2 and a Linux distribution first."
    }
    $distros = @()
    foreach ($line in $raw) {
        $name = ($line | ForEach-Object { $_.Trim() })
        if ($name) { $distros += $name }
    }
    if (-not $distros) {
        throw "Could not enumerate WSL distros."
    }
    return $distros
}

function Get-DefaultDistro {
    param([string[]]$Distros)
    foreach ($preferred in @("Ubuntu", "Ubuntu-24.04", "Ubuntu-22.04", "Debian", "openSUSE-Leap", "Arch")) {
        if ($Distros -contains $preferred) {
            return $preferred
        }
    }
    return $Distros[0]
}

function Get-WslHome {
    param([string]$Distribution)
    $home = & wsl.exe -d $Distribution -- bash -lc 'printf %s "$HOME"' 2>$null
    if (-not $home) {
        throw "Unable to determine the WSL home directory for distribution '$Distribution'."
    }
    return ($home | Out-String).Trim()
}

function New-HermesTaskAction {
    param(
        [string]$Distribution,
        [string]$ScriptPath,
        [string]$AutostartMode
    )

    $bashCommand = "HERMES_AUTOSTART_MODE='$AutostartMode' '$ScriptPath'"
    $argument = "-d $Distribution -- bash -lc `"$bashCommand`""
    return New-ScheduledTaskAction -Execute "wsl.exe" -Argument $argument
}

$distros = Get-WslDistros
if (-not $Distro) {
    $Distro = Get-DefaultDistro -Distros $distros
}

if ($distros -notcontains $Distro) {
    throw "WSL distro '$Distro' was not found. Available distros: $($distros -join ', ')"
}

$wslHome = Get-WslHome -Distribution $Distro
if (-not $RepoPath) {
    $RepoPath = "$wslHome/hermes-agent"
}

$wslScriptPath = "$RepoPath/scripts/hermes_autostart.sh"
$scriptCheck = & wsl.exe -d $Distro -- bash -lc "test -x '$wslScriptPath'" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw @"
Could not find an executable Hermes autostart script at:
  $wslScriptPath

If Hermes lives somewhere other than $wslHome/hermes-agent, re-run this script with:
  .\setup_hermes_autostart.ps1 -RepoPath /path/to/hermes-agent

The path you pass must be a WSL path, not a Windows path.
"@
}

$action = New-HermesTaskAction -Distribution $Distro -ScriptPath $wslScriptPath -AutostartMode $Mode
$trigger = New-ScheduledTaskTrigger -AtLogOn
$userId = if ($env:USERDOMAIN) { "$env:USERDOMAIN\$env:USERNAME" } else { $env:USERNAME }
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel LeastPrivilege
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Auto-start Hermes Agent in WSL at logon" `
    -Force | Out-Null

Write-Host "✓ Scheduled task '$TaskName' created." -ForegroundColor Green
Write-Host "  Distro : $Distro" -ForegroundColor Cyan
Write-Host "  Repo   : $RepoPath" -ForegroundColor Cyan
Write-Host "  Mode   : $Mode" -ForegroundColor Cyan
Write-Host ""
Write-Host "To test it now, run:" -ForegroundColor Yellow
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Yellow
Write-Host ""
Write-Host "To remove it later, run:" -ForegroundColor Yellow
Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:$false" -ForegroundColor Yellow
