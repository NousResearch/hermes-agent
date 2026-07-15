@echo off
setlocal
if "%ATOMIC_MAIL_CREDENTIALS_DIR%"=="" (
  if defined HERMES_HOME (
    set "ATOMIC_MAIL_CREDENTIALS_DIR=%HERMES_HOME%\atomicmail"
  ) else (
    set "ATOMIC_MAIL_CREDENTIALS_DIR=%USERPROFILE%\.hermes\atomicmail"
  )
)
node "%~dp0..\lib\esm\skill\cli.js" %*
