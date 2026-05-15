@echo off
REM Start Hermes Gateway with all required env vars
set HERMES_DIR=%~dp0..
cd /d %HERMES_DIR%

REM Load .env vars explicitly
for /f "tokens=* delims=" %%i in ('type %USERPROFILE%\.hermes\.env ^| findstr /v "^#" ^| findstr /v "^$"') do set %%i

REM Ensure PATH includes venv
set PATH=%HERMES_DIR%\.venv\Scripts;%PATH%

echo Starting Hermes Gateway...
hermes gateway run
