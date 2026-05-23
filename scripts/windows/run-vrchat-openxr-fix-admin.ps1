#Requires -Version 5.1
<#
.SYNOPSIS
  Elevated OpenXR ActiveRuntime sync (HKLM + HKCU) for Virtual Desktop / VRChat.

.DESCRIPTION
  Dot-sources vrchat_quest2_openxr_fix.ps1 and runs Invoke-OpenXrFix.
  Intended for Start-Process -Verb RunAs or double-click (self-elevates).

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\run-vrchat-openxr-fix-admin.ps1 -Preference VirtualDesktop
#>
[CmdletBinding()]
param(
    [ValidateSet('Auto', 'VirtualDesktop', 'SteamVR')]
    [string]$Preference = 'VirtualDesktop',
    [switch]$ResetBindings,
    [string]$LogPath = ''
)

$ErrorActionPreference = 'Stop'
if (-not $LogPath) {
    $LogPath = Join-Path $env:TEMP 'vrchat_openxr_admin_fix.log'
}

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format 'o'), $Message
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
    Write-Host $line
}

try {
    Write-Log "OpenXR admin fix starting (Preference=$Preference, ResetBindings=$($ResetBindings.IsPresent))"
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
    if (-not $isAdmin) {
        Write-Log 'Not elevated; re-launching with RunAs (approve UAC)...'
        $self = $MyInvocation.MyCommand.Path
        $argList = @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$self`"",
            '-Preference', $Preference,
            '-LogPath', "`"$LogPath`""
        )
        if ($ResetBindings) { $argList += '-ResetBindings' }
        Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList ($argList -join ' ') -Wait
        Write-Log 'Elevated child process finished.'
        exit $LASTEXITCODE
    }

    $fixScript = Join-Path $PSScriptRoot 'vrchat_quest2_openxr_fix.ps1'
    if (-not (Test-Path $fixScript)) { throw "Missing fix module: $fixScript" }
    . $fixScript

    $result = Invoke-OpenXrFix -Preference $Preference -ResetBindings:$ResetBindings
    Write-Log ("chosen_manifest=" + $result.chosen_manifest)
    foreach ($w in $result.registry_writes) { Write-Log ("wrote: " + $w) }
    Write-Log 'SUCCESS'
    exit 0
}
catch {
    Write-Log ("FAILED: " + $_.Exception.Message)
    exit 1
}
