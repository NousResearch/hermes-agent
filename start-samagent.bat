@echo off
setlocal
cd /d "%~dp0"

set VENV_DIR=.samagent-venv
if "%SAMAGENT_PORT%"=="" set SAMAGENT_PORT=8080

echo ============================================================
echo  Starting SamAgent Local Platform (Windows)
echo ============================================================

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/3] Creating local Python environment in %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    "%VENV_DIR%\Scripts\pip.exe" install -q pyyaml fastapi uvicorn httpx pytest snowballstemmer pydantic "ruamel.yaml"
)

echo [2/3] Installing VS Code extension and initializing ~\SamAgentProjects ...
set PYTHONPATH=%CD%
"%VENV_DIR%\Scripts\python.exe" -c "from samagent.platform_installer import install_os_desktop_platform; install_os_desktop_platform(port=%SAMAGENT_PORT%)"

echo [3/3] Opening http://127.0.0.1:%SAMAGENT_PORT% (Hot-Reload ON) ...
start "" "http://127.0.0.1:%SAMAGENT_PORT%"
"%VENV_DIR%\Scripts\python.exe" -m samagent.ui_server --host 127.0.0.1 --port %SAMAGENT_PORT% --reload
