@echo off
chcp 65001 >nul
title Hermes Agent

setlocal enabledelayedexpansion

echo ==============================================
echo          Hermes Agent 启动脚本
echo ==============================================
echo.

set "HERMES_HOME=C:\Users\smith\AppData\Local\hermes"
set "OBSIDIAN_VAULT_PATH=D:\sks-local\九天"

echo 正在读取配置...
echo HERMES_HOME: %HERMES_HOME%
echo OBSIDIAN_VAULT_PATH: %OBSIDIAN_VAULT_PATH%
echo.

echo 正在启动 Hermes Agent...
echo.

python hermes chat

echo.
echo Hermes Agent 已退出
endlocal
pause