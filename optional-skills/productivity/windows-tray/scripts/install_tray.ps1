# install_tray.ps1 - set up the Hermes Windows tray from this skill's scripts/.
# Creates a dedicated venv (never Hermes' own, which its dep-sync strips),
# copies the scripts into %LOCALAPPDATA%\hermes\tray, installs pystray+pillow,
# writes a shell:startup shortcut for the watchdog, and starts it.
# ASCII only. -Uninstall removes the shortcut and stops the helpers.
param([switch]$Uninstall)
$ErrorActionPreference = "Stop"
$src    = Split-Path -Parent $MyInvocation.MyCommand.Path
$dst    = Join-Path $env:LOCALAPPDATA "hermes\tray"
$plugin = Join-Path $env:LOCALAPPDATA "hermes\plugins\tray-needs-input"
$lnk    = Join-Path ([Environment]::GetFolderPath("Startup")) "HermesTray.lnk"

if ($Uninstall) {
    Stop-Process -Name pythonw -Force -ErrorAction SilentlyContinue
    if (Test-Path $lnk) { Remove-Item $lnk -Force }
    Write-Host "uninstalled: startup shortcut removed, helpers stopped."
    Write-Host "leftovers you can delete manually: $dst  and  $plugin"
    exit 0
}

New-Item -ItemType Directory -Force -Path $dst | Out-Null
foreach ($f in "hermes_tray.py","tray_watchdog.py","windows_tray_state.py","start_watchdog.js") {
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
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnk)
$sc.TargetPath = $pythonw
$sc.Arguments  = """$(Join-Path $dst "tray_watchdog.py")"""
$sc.WorkingDirectory = $dst
$sc.Description = "Hermes tray watchdog"
$sc.Save()

Start-Process -WindowStyle Hidden -FilePath $pythonw -ArgumentList """$(Join-Path $dst "tray_watchdog.py")""" -WorkingDirectory $dst
Write-Host "tray installed + watchdog started. Enable the amber 'needs input' dot with:"
Write-Host "  hermes config set plugins.enabled ""[tray-needs-input]""   (append to any existing entries)"
Write-Host "then restart the Hermes desktop app once so its backend loads the plugin."
