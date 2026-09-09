@echo off
REM collect-logs.cmd — double-clickable wrapper. Rebuilds D:\logs\ + D:\logs.zip and runs the completeness self-check.
REM Run any time by hand; also run at the end of every agent task. Real logic is in collect-logs.ps1 (same folder).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0collect-logs.ps1" %*
set _rc=%ERRORLEVEL%
echo.
if %_rc%==0 (echo Result: COMPLETE) else (echo Result: INCOMPLETE - see FAIL lines above)
echo.
pause
exit /b %_rc%
