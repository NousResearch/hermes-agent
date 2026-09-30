# SamAgent Local Platform & VS Code Bridge Installer (Windows PowerShell)
$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $PSScriptRoot
$Port = if ($env:SAMAGENT_PORT) { $env:SAMAGENT_PORT } else { "8080" }

$env:PYTHONPATH = $RepoDir
python -c "from samagent.platform_installer import install_os_desktop_platform; import json; print(json.dumps(install_os_desktop_platform(port=$Port), indent=2))"
Write-Host "SamAgent Local Platform & VS Code Bridge installed at http://127.0.0.1:$Port"
