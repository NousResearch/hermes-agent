@echo off
rem Start IDLE using the appropriate Python interpreter
set CURRDIR=%~dp0
start "IDLE" "%CURRDIR%..\..\pythonw.exe" "%CURRDIR%idle.pyw" %1 %2 %3 %4 %5 %6 %7 %8 %9
