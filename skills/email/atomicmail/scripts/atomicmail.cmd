@echo off
setlocal
if "%ATOMIC_MAIL_CREDENTIALS_DIR%"=="" set "ATOMIC_MAIL_CREDENTIALS_DIR=%USERPROFILE%\.hermes\atomicmail"
node "%~dp0..\lib\esm\skill\cli.js" %*
