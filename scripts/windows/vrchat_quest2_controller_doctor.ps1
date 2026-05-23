#Requires -Version 5.1
<#
.SYNOPSIS
  Read-only Quest 2 + Virtual Desktop + VRChat controller / VR stack doctor.

.DESCRIPTION
  Collects process, registry, service, port, and VRChat config evidence to diagnose
  "controllers not working" on Windows 11 with Quest 2 via Virtual Desktop.
  Does NOT modify system state unless -ApplyFixHints is passed (prints hints only).

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\vrchat_quest2_controller_doctor.ps1 -Json
#>
[CmdletBinding()]
param(
    [switch]$Json,
    [switch]$ApplyFixHints,
    [string]$OutputPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'SilentlyContinue'

function Get-ProcessMatches {
    param([string[]]$Terms)
    Get-Process -ErrorAction SilentlyContinue | ForEach-Object {
        $proc = $_
        $path = $null
        try { $path = $proc.Path } catch {}
        foreach ($term in $Terms) {
            if ($proc.ProcessName -like "*$term*") {
                return [PSCustomObject]@{
                    pid         = $proc.Id
                    name        = $proc.ProcessName
                    path        = $path
                }
            }
            if ($path -and ($path -like "*$term*")) {
                return [PSCustomObject]@{
                    pid         = $proc.Id
                    name        = $proc.ProcessName
                    path        = $path
                }
            }
        }
    } | Sort-Object pid -Unique
}

function Get-RegistryOpenXr {
    $results = @()
    foreach ($root in @('HKLM:\SOFTWARE\Khronos\OpenXR\1', 'HKCU:\SOFTWARE\Khronos\OpenXR\1')) {
        $active = Join-Path $root 'ActiveRuntime'
        $item = [PSCustomObject]@{
            root          = $root
            active_exists = (Test-Path $active)
            active_json   = $null
            active_dll    = $null
        }
        if (Test-Path $active) {
            $props = Get-ItemProperty $active
            $item.active_json = ($props | ConvertTo-Json -Compress)
            if ($props.PSObject.Properties.Name -contains '(default)') {
                $item.active_dll = $props.'(default)'
            }
        }
        $results += $item
    }
    return $results
}

function Get-SteamVrInstall {
    $dirs = @(
        'C:\Program Files (x86)\Steam\steamapps\common\SteamVR',
        'C:\Program Files\Steam\steamapps\common\SteamVR'
    ) | Where-Object { Test-Path $_ }
    $steamPath = $null
    try {
        $steamPath = (Get-ItemProperty 'HKCU:\Software\Valve\Steam').SteamPath
    } catch {}
    $first = $dirs | Select-Object -First 1
    $vrstartup = if ($first) { Join-Path $first 'bin\win64\vrstartup.exe' } else { $null }
    return [PSCustomObject]@{
        steam_path     = $steamPath
        steamvr_dirs   = @($dirs)
        vrstartup_exe  = $vrstartup
        vrstartup_exists = [bool]($vrstartup -and (Test-Path $vrstartup))
    }
}

function Get-OculusInstall {
    $roots = @(
        'C:\Program Files\Oculus',
        'C:\Program Files\Meta Horizon',
        "${env:ProgramFiles(x86)}\Oculus"
    ) | Where-Object { Test-Path $_ }
    $client = @(
        'C:\Program Files\Oculus\Support\oculus-client\OculusClient.exe',
        'C:\Program Files\Meta Horizon\Client\OculusClient.exe'
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    return [PSCustomObject]@{
        install_roots = @($roots)
        client_exe    = $client
    }
}

function Get-VrChatWindows {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections.Generic;
public static class VrWinEnum {
  public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
  public static List<string> Collect(string filter) {
    var list = new List<string>();
    EnumWindows((hWnd, lParam) => {
      if (!IsWindowVisible(hWnd)) return true;
      var sb = new StringBuilder(512);
      GetWindowText(hWnd, sb, 512);
      var title = sb.ToString();
      if (title.IndexOf(filter, StringComparison.OrdinalIgnoreCase) >= 0) {
        uint pid; GetWindowThreadProcessId(hWnd, out pid);
        list.Add(title + "|pid=" + pid);
      }
      return true;
    }, IntPtr.Zero);
    return list;
  }
}
"@
    return [VrWinEnum]::Collect('VRChat')
}

function Get-VrChatConfigHints {
    $localLow = Join-Path $env:USERPROFILE 'AppData\LocalLow\VRChat\VRChat'
    $hints = [PSCustomObject]@{
        config_dir_exists = (Test-Path $localLow)
        config_dir        = $localLow
        osc_settings      = $null
        binding_files       = @()
        recent_logs         = @()
    }
    if (-not (Test-Path $localLow)) { return $hints }

    $osc = Join-Path $localLow 'OSC\config.json'
    if (Test-Path $osc) {
        try { $hints.osc_settings = Get-Content $osc -Raw } catch {}
    }

    $bindingsRoot = Join-Path $localLow 'Bindings'
    if (Test-Path $bindingsRoot) {
        $hints.binding_files = @(Get-ChildItem $bindingsRoot -Recurse -File -ErrorAction SilentlyContinue |
            Select-Object -First 20 | ForEach-Object { $_.FullName })
    }

    $logs = Join-Path $localLow 'Logs'
    if (Test-Path $logs) {
        $hints.recent_logs = @(Get-ChildItem $logs -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 3 |
            ForEach-Object { $_.FullName })
    }
    return $hints
}

function Get-PortListeners {
    param([int[]]$Ports)
    Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -in $Ports } |
        ForEach-Object {
            $procName = $null
            try { $procName = (Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName } catch {}
            [PSCustomObject]@{
                port = $_.LocalPort
                address = $_.LocalAddress
                pid = $_.OwningProcess
                process = $procName
            }
        }
}

function Get-VdServices {
    Get-Service -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like '*VirtualDesktop*' -or $_.DisplayName -like '*Virtual Desktop*' } |
        Select-Object Name, Status, DisplayName
}

function Test-VrModeLikely {
    param(
        $VrProcesses,
        $VrChatWindows
    )
    $steamvrRunning = @($VrProcesses | Where-Object { $_.name -match 'vr(server|compositor|monitor|startup|webhelper)' }).Count -gt 0
    $oculusRunning = @($VrProcesses | Where-Object { $_.name -match 'OVR|Oculus|MetaQuest|QuestLink' }).Count -gt 0
    $vdRunning = @($VrProcesses | Where-Object { $_.name -match 'VirtualDesktop' }).Count -gt 0
    $desktopTitleOnly = $true
    foreach ($w in $VrChatWindows) {
        if ($w -match '(?i)VRChat.*\(VR\)|VR Mode|SteamVR|OpenXR') { $desktopTitleOnly = $false }
    }
    return [PSCustomObject]@{
        steamvr_running = $steamvrRunning
        oculus_runtime_running = $oculusRunning
        virtual_desktop_running = $vdRunning
        vrchat_title_suggests_desktop = $desktopTitleOnly
        likely_in_vr = ($steamvrRunning -or $oculusRunning -or $vdRunning) -and -not $desktopTitleOnly
        likely_desktop_or_headless = (-not $steamvrRunning -and -not $oculusRunning) -or $desktopTitleOnly
    }
}

function Get-VrChatLogSignals {
    $localLow = Join-Path $env:USERPROFILE 'AppData\LocalLow\VRChat\VRChat'
    if (-not (Test-Path $localLow)) {
        return [PSCustomObject]@{ log_found = $false }
    }
    $latest = Get-ChildItem $localLow -Filter 'output_log_*.txt' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) {
        return [PSCustomObject]@{ log_found = $false; log_dir = $localLow }
    }
    $text = Get-Content $latest.FullName -Raw -ErrorAction SilentlyContinue
    $signals = [ordered]@{
        log_found                         = $true
        log_path                          = $latest.FullName
        log_mtime                         = $latest.LastWriteTime.ToString('o')
        steamvr_initialized               = [bool]($text -match '\[SteamVR\].*Initialized')
        vd_oculus_driver                  = [bool]($text -match 'oculus_virtualdesktop')
        quest2_hmd                        = [bool]($text -match 'Oculus Quest2|Quest2')
        touch_controller_usable           = if ($text -match 'Oculus Touch controller = (True|False)') { $Matches[1] } else { $null }
        openxr_controller_usable          = [bool]($text -match 'VRCInputProcessorOpenXR: can use OpenXR controller')
        openxr_binding                    = if ($text -match 'Loaded Input Binding\[_OPENXR_GENERIC\]: (\w+)') { $Matches[1] } else { $null }
        osc_enabled_setting               = if ($text -match 'OSC enabled: (True|False)') { $Matches[1] } else { $null }
        xr_device_none_at_boot            = [bool]($text -match 'XR Device: None')
    }
    return [PSCustomObject]$signals
}

function Get-FixRecommendations {
    param($Report)
    $recs = New-Object System.Collections.Generic.List[string]

    $log = $Report.vrchat_log_signals
    if ($log.log_found -and $log.touch_controller_usable -eq 'False' -and $log.vd_oculus_driver) {
        $recs.Add('1-HIGH: Latest VRChat log shows VD+Quest2 HMD OK but "Oculus Touch controller = False". VD is not passing controller tracking to SteamVR/VRChat — fix VD Streamer controller/SteamVR passthrough BEFORE blaming Hermes.')
    }
    if ($log.log_found -and $log.openxr_binding -eq 'Custom') {
        $recs.Add('1-HIGH: VRChat loaded Custom OpenXR binding. Reset: Quick Menu > Options > Controls > Reset VR Controls; or delete %LOCALAPPDATA%Low\\VRChat\\VRChat\\Bindings then restart.')
    }
    if ($Report.openxr_registry | Where-Object { -not $_.active_exists }) {
        $recs.Add('1-HIGH: No OpenXR ActiveRuntime in registry. Install Meta Quest Link (PC) OR set SteamVR as OpenXR runtime via SteamVR Settings > Developer > Set SteamVR as OpenXR Runtime.')
    }
    if (-not $Report.oculus_install.client_exe) {
        $recs.Add('2-MED: Meta Quest Link / Oculus PC app not installed on this machine. VD-only setups often still need SteamVR controller passthrough configured; Link app helps register OpenXR runtime.')
    }

    if ($Report.vr_mode.likely_desktop_or_headless -and -not ($log.log_found -and $log.vd_oculus_driver)) {
        $recs.Add('1-HIGH: VRChat appears to be desktop/flat mode OR no VR runtime is active. Launch from Virtual Desktop Games tab (not Desktop mirror), or start SteamVR/Oculus first.')
    }
    if (-not $Report.vr_mode.steamvr_running -and -not $Report.vr_mode.oculus_runtime_running) {
        $recs.Add('2-MED: No SteamVR (vrserver/vrcompositor) and no Oculus runtime (OVRServer) detected right now. During play, SteamVR may only run while a VR title is active — launch VRChat in HMD and re-run this doctor.')
    }
    if ($Report.vr_mode.virtual_desktop_running -and -not $Report.vr_mode.steamvr_running) {
        $recs.Add('2-MED: VD streamer is running but SteamVR is not. In VD Streamer: enable SteamVR integration + controller tracking; launch VRChat from Games tab with controllers awake in VD.')
    }
    foreach ($reg in $Report.openxr_registry) {
        if ($reg.active_dll -and ($reg.active_dll -notmatch 'oculus|meta|steam')) {
            $recs.Add('2-MED: OpenXR active runtime is neither Oculus nor SteamVR: ' + $reg.active_dll)
        }
    }
    if (-not $Report.steamvr_install.vrstartup_exists) {
        $recs.Add('2-MED: SteamVR install not found or incomplete. Repair via Steam > SteamVR > Properties > Verify.')
    }
    if ($Report.port_listeners | Where-Object { $_.port -eq 9001 }) {
        $recs.Add('3-LOW: Port 9001 in use — may conflict with VRChat OSC input if misconfigured (Hermes uses outbound OSC; usually not controller-related).')
    }
    $recs.Add('2-MED: In VRChat Quick Menu > Options > Controls > Reset VR Controls / Calibrate FB Tracker.')
    $recs.Add('2-MED: VRChat Settings > OSC > disable "OSC as Input Controller" unless you intentionally drive input via OSC.')
    $recs.Add('3-MED: Clear bindings cache: backup then delete %LOCALAPPDATA%Low\VRChat\VRChat\Bindings and restart VRChat.')
    $recs.Add('3-MED: Oculus PC app > Settings > General > Set Meta Quest Link as active OpenXR runtime (or use SteamVR OpenXR overlay).')
    $recs.Add('4-ALT: Bypass VD — USB Link/Air Link + launch VRChat from Oculus PC library to isolate VD passthrough issues.')

    return @($recs)
}

$vrTerms = @(
    'VRChat', 'steam', 'steamvr', 'vrserver', 'vrcompositor', 'vrmonitor', 'vrstartup', 'vrwebhelper',
    'OVRServer', 'OVRService', 'OVRRedir', 'OculusClient', 'OculusApp', 'VirtualDesktop',
    'VirtualDesktopStream', 'MetaQuest', 'QuestLink', 'RemoteDesktop', 'MixedReality'
)

$report = [ordered]@{
    timestamp_utc     = (Get-Date).ToUniversalTime().ToString('o')
    hostname          = $env:COMPUTERNAME
    username          = $env:USERNAME
    vr_processes      = @(Get-ProcessMatches -Terms $vrTerms)
    vrchat_windows    = @(Get-VrChatWindows)
    openxr_registry   = @(Get-RegistryOpenXr)
    steamvr_install   = Get-SteamVrInstall
    oculus_install    = Get-OculusInstall
    vd_services       = @(Get-VdServices)
    port_listeners    = @(Get-PortListeners -Ports @(9000, 9001, 27000, 27001, 27002, 24500, 24501))
    vrchat_config     = Get-VrChatConfigHints
    vrchat_log_signals = Get-VrChatLogSignals
}

$report['vr_mode'] = Test-VrModeLikely -VrProcesses $report.vr_processes -VrChatWindows $report.vrchat_windows
$report['recommendations'] = Get-FixRecommendations -Report $report

if ($OutputPath) {
    $report | ConvertTo-Json -Depth 8 | Set-Content -Path $OutputPath -Encoding UTF8
}

if ($Json) {
    $report | ConvertTo-Json -Depth 8
} else {
    Write-Host '=== VRChat Quest2 Controller Doctor ===' -ForegroundColor Cyan
    Write-Host ('UTC: ' + $report.timestamp_utc)
    Write-Host ''
    Write-Host 'VR mode assessment:' -ForegroundColor Yellow
    $report.vr_mode | Format-List
    Write-Host 'VR-related processes:' -ForegroundColor Yellow
    $report.vr_processes | Format-Table -AutoSize
    Write-Host 'VRChat window titles:' -ForegroundColor Yellow
    $report.vrchat_windows
    Write-Host 'OpenXR registry:' -ForegroundColor Yellow
    $report.openxr_registry | Format-List
    Write-Host 'Recommendations (try IN ORDER):' -ForegroundColor Green
    $report.recommendations | ForEach-Object { Write-Host $_ }
}

if ($ApplyFixHints) {
    Write-Host ''
    Write-Host 'ApplyFixHints is informational only — no automatic destructive fixes.' -ForegroundColor DarkYellow
}
