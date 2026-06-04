@echo off
cd /d "C:\Users\Administrator\AppData\Local\hermes\hermes-agent\apps\desktop"

echo 正在替换桌面版中文界面...
echo.

REM 删除旧的 JS 文件
del /q /f "C:\Users\Administrator\AppData\Local\hermes\hermes-agent\apps\desktop\release\win-unpacked\resources\app.asar" 2>nul
echo 已删除旧 app.asar

REM 重建 ASAR
call npx -y @electron/asar pack /tmp/asar-extract "C:\Users\Administrator\AppData\Local\hermes\hermes-agent\apps\desktop\release\win-unpacked\resources\app.asar"

echo.
echo ✅ 完成！请重新运行桌面版 Hermes
pause
