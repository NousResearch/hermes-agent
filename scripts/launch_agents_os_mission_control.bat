@echo off
setlocal
set PORT=18790
set DISTRO=Ubuntu-24.04
start "Agents OS Mission Control Server" cmd.exe /K "wsl.exe -d %DISTRO% -- bash -lc ""export HERMES_HOME=/home/goran/.hermes-doni-clean; cd /mnt/d/HermesAgent/app; ./venv/bin/python -m hermes_cli.agents_os web --host 127.0.0.1 --port %PORT%"""
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:%PORT%"
echo Agents OS Mission Control: http://127.0.0.1:%PORT%
endlocal
