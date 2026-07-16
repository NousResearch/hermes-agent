@echo off
REM bin/hermes.cmd — Windows in-repo launcher stub for ejected mode.
REM
REM See docs/updater-world.md §2.5.1. Same logic as bin/hermes (POSIX).

setlocal

REM --- env hygiene ---
set PYTHONPATH=
set PYTHONHOME=
set UV_NO_CONFIG=1

set DIR=%~dp0
set REPO_ROOT=%DIR%..

REM --- try native launcher ---
if exist "%REPO_ROOT%\.hermes-launcher\hermes.exe" (
    "%REPO_ROOT%\.hermes-launcher\hermes.exe" %*
    exit /b %ERRORLEVEL%
)

REM --- fallback: exec venv python ---
if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    set VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe
) else if exist "%REPO_ROOT%\venv\Scripts\python.exe" (
    set VENV_PYTHON=%REPO_ROOT%\venv\Scripts\python.exe
) else (
    echo hermes: this tree's virtualenv is missing or broken. 1>&2
    echo   tree: %REPO_ROOT% 1>&2
    echo   fix:  hermes dev sync        (source checkout) 1>&2
    exit /b 3
)

"%VENV_PYTHON%" -m hermes_cli.main %*
exit /b %ERRORLEVEL%
