$ErrorActionPreference = 'Stop'
$Port = 18790
$Distro = 'Ubuntu-24.04'
$Command = "export HERMES_HOME=/home/goran/.hermes-doni-clean; cd /mnt/d/HermesAgent/app; ./venv/bin/python -m hermes_cli.agents_os web --host 127.0.0.1 --port $Port"
Start-Process -FilePath 'cmd.exe' -ArgumentList @('/K', "wsl.exe -d $Distro -- bash -lc `"$Command`"") -WindowStyle Normal
Start-Sleep -Seconds 3
Start-Process "http://127.0.0.1:$Port"
Write-Host "Agents OS Mission Control: http://127.0.0.1:$Port"
