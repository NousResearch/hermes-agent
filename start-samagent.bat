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
    set "BOOT_PY="
    for /d %%D in ("%LOCALAPPDATA%\hermes\tools\python-*") do if exist "%%D\python.exe" set "BOOT_PY=%%D\python.exe"
    if not defined BOOT_PY (
        for /d %%D in ("%APPDATA%\uv\python\cpython-*") do if exist "%%D\python.exe" set "BOOT_PY=%%D\python.exe"
    )
    if not defined BOOT_PY (
        python -c "import sys" >nul 2>&1
        if not errorlevel 1 set "BOOT_PY=python"
    )
    set "BOOT_UV="
    for /d %%D in ("%USERPROFILE%\.hermes\tools\uv-*") do if exist "%%D\uv.exe" set "BOOT_UV=%%D\uv.exe"

    if defined BOOT_UV (
        "%BOOT_UV%" venv "%VENV_DIR%"
        "%BOOT_UV%" pip install --python "%VENV_DIR%\Scripts\python.exe" pyyaml fastapi uvicorn httpx pytest snowballstemmer pydantic "ruamel.yaml"
    ) else if defined BOOT_PY (
        "%BOOT_PY%" -m venv "%VENV_DIR%"
        "%VENV_DIR%\Scripts\pip.exe" install -q pyyaml fastapi uvicorn httpx pytest snowballstemmer pydantic "ruamel.yaml"
    ) else (
        echo Error: Python was not found. Please run .\setup-hermes.ps1 first.
        exit /b 1
    )
)

echo [2/3] Installing VS Code extension and initializing ~\SamAgentProjects ...
set PYTHONPATH=%CD%
"%VENV_DIR%\Scripts\python.exe" -c "from samagent.platform_installer import install_os_desktop_platform; install_os_desktop_platform(port=%SAMAGENT_PORT%)"

echo [3/3] Opening http://127.0.0.1:%SAMAGENT_PORT% (Hot-Reload ON) ...
start "" "http://127.0.0.1:%SAMAGENT_PORT%"
"%VENV_DIR%\Scripts\python.exe" -m samagent.ui_server --host 127.0.0.1 --port %SAMAGENT_PORT% --reload
