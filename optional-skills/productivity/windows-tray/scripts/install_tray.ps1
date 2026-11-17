# install_tray.ps1 - set up the Hermes Windows tray from this skill's scripts/.
# Creates a dedicated venv (never Hermes' own, which its dep-sync strips),
# copies the scripts into %LOCALAPPDATA%\hermes\tray, installs pystray+pillow,
# writes a shell:startup shortcut, and starts the single resident tray process.
# ASCII only. -Uninstall removes the shortcut and stops the tray.
param([switch]$Uninstall)
$ErrorActionPreference = "Stop"
$src    = Split-Path -Parent $MyInvocation.MyCommand.Path
$dst    = Join-Path $env:LOCALAPPDATA "hermes\tray"
$plugin = Join-Path $env:LOCALAPPDATA "hermes\plugins\tray-needs-input"
$lnk    = Join-Path ([Environment]::GetFolderPath("Startup")) "HermesTray.lnk"

if ($Uninstall) {
    # remove the autostart entry FIRST: a scan failure below must not strand it
    if (Test-Path $lnk) { Remove-Item $lnk -Force }
    # scope the kill: match only our tray process, never every pythonw on the box
    try {
        Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" | ForEach-Object {
            if ($_.CommandLine -like "*hermes_tray.py*") {
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {
        Write-Host "process scan failed, tray may still be running: $_"
    }
    Write-Host "uninstalled: startup shortcut removed, tray stopped."
    Write-Host "leftovers you can delete manually: $dst  and  $plugin"
    exit 0
}

New-Item -ItemType Directory -Force -Path $dst | Out-Null
foreach ($f in "hermes_tray.py","windows_tray_state.py") {
    Copy-Item (Join-Path $src $f) (Join-Path $dst $f) -Force
}
New-Item -ItemType Directory -Force -Path $plugin | Out-Null
Copy-Item (Join-Path $src "tray-needs-input\*") $plugin -Recurse -Force

$venv = Join-Path $dst "venv"
if (-not (Test-Path (Join-Path $venv "Scripts\pythonw.exe"))) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv $venv --python 3.11 | Out-Null
        uv pip install --python (Join-Path $venv "Scripts\python.exe") pystray pillow | Out-Null
    } else {
        python -m venv $venv
        & (Join-Path $venv "Scripts\python.exe") -m pip install --quiet pystray pillow
    }
}

$pythonw = Join-Path $venv "Scripts\pythonw.exe"
$tray = Join-Path $dst "hermes_tray.py"
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnk)
$sc.TargetPath = $pythonw
$sc.Arguments  = """$tray"""
$sc.WorkingDirectory = $dst
$sc.Description = "Hermes tray (single resident process)"
$sc.Save()

Start-Process -WindowStyle Hidden -FilePath $pythonw -ArgumentList """$tray""" -WorkingDirectory $dst
Write-Host "tray installed + started (icon appears once the desktop app runs)."
# Enable the observer plugin by READ-MODIFY-WRITE: a bare `set` with a
# one-item list REPLACES plugins.enabled and silently drops other plugins.
$items = @()
if (Get-Command hermes -ErrorAction SilentlyContinue) {
    try {
        foreach ($line in (((& hermes config get plugins.enabled 2>$null) -join "`n") -split "`n")) {
            if ($line -match '^\s*-\s*(\S+)\s*$') { $items += $Matches[1] }
        }
    } catch { $items = @() }
}
if ($items -contains "tray-needs-input") {
    Write-Host "needs-input plugin: already enabled - nothing to do."
} elseif ($items.Count -gt 0) {
    $list = ($items + "tray-needs-input") -join ", "
    Write-Host "needs-input plugin: enable with (keeps your existing entries):"
    Write-Host "  hermes config set plugins.enabled ""[$list]"""
} else {
    Write-Host "needs-input plugin: read your current list first, then append - do NOT replace:"
    Write-Host "  1) hermes config get plugins.enabled"
    Write-Host "  2) hermes config set plugins.enabled ""[<every existing entry>, tray-needs-input]"""
}
Write-Host "then restart the Hermes desktop app once so its backend loads the plugin."
