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
    Stop-Process -Name pythonw -Force -ErrorAction SilentlyContinue
    if (Test-Path $lnk) { Remove-Item $lnk -Force }
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
Write-Host "Enable the amber 'needs input' dot with:"
Write-Host "  hermes config set plugins.enabled ""[tray-needs-input]""   (append to any existing entries)"
Write-Host "then restart the Hermes desktop app once so its backend loads the plugin."
