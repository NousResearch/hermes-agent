# start-desktop.ps1 - Windows 端 hermes desktop 启动器
# 用法: 双击 Hermes Desktop.lnk（指向本脚本）
# 作用: 设 3 env vars + 探 WSL 9119 + 启 electron

$ErrorActionPreference = 'Continue'

# ============ 配置 ============
$WSL_HERMES_HOME = '\\wsl$\Ubuntu\home\lujun\.hermes'
$WSL_URL = 'http://127.0.0.1:9119'
$HERMES_AGENT_WIN = 'C:\Users\lujun\AppData\Local\hermes\hermes-agent'
$ELECTRON_CLI = "$HERMES_AGENT_WIN\node_modules\electron\cli.js"
$DESKTOP_DIR = "$HERMES_AGENT_WIN\apps\desktop"
$NODE_EXE = 'C:\Program Files\nodejs\node.exe'
$ICON = "$HERMES_AGENT_WIN\apps\desktop\dist\hermes.png"

# ============ Step 1: 读 WSL token ============
Write-Host "=== Hermes Desktop 启动器 ===" -ForegroundColor Cyan
Write-Host ""

$tokenPath = "$WSL_HERMES_HOME\dashboard.token"
if (-not (Test-Path $tokenPath)) {
    Write-Host "[ERROR] Token 文件不存在: $tokenPath" -ForegroundColor Red
    Write-Host "请先在 WSL 跑: bash ~/.hermes/skills/hermes-windows-wsl-install/scripts/setup-shared-memory.sh" -ForegroundColor Yellow
    Read-Host "按 Enter 退出"
    exit 1
}

$TOKEN = (Get-Content $tokenPath -Raw).Trim()
if ($TOKEN.Length -lt 24) {
    Write-Host "[ERROR] Token 太短 ($($TOKEN.Length)B)，应是 24+" -ForegroundColor Red
    Read-Host "按 Enter 退出"
    exit 1
}
Write-Host "[1/4] Token: $($TOKEN.Substring(0,8))...$($TOKEN.Substring($TOKEN.Length-4)) ($($TOKEN.Length)B)"

# ============ Step 2: 设 3 env vars ============
$env:HERMES_HOME = $WSL_HERMES_HOME
$env:HERMES_DESKTOP_REMOTE_URL = $WSL_URL
$env:HERMES_DESKTOP_REMOTE_TOKEN = $TOKEN
Write-Host "[2/4] HERMES_HOME = $env:HERMES_HOME"
Write-Host "      REMOTE_URL  = $env:HERMES_DESKTOP_REMOTE_URL"

# ============ Step 3: 探 WSL 9119 ============
Write-Host ""
Write-Host -NoNewline "[3/4] 探 WSL 9119 ... "
try {
    $resp = Invoke-WebRequest -Uri "$WSL_URL/api/status" -Headers @{"X-Hermes-Session-Token" = $TOKEN} -UseBasicParsing -TimeoutSec 3
    Write-Host "HTTP $($resp.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "FAILED" -ForegroundColor Red
    Write-Host ""
    Write-Host "[WARN] WSL 9119 dashboard 不可达！" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "请先在 WSL 启 dashboard："
    Write-Host '  wsl.exe -- bash -c "bash ~/.hermes/skills/hermes-windows-wsl-install/scripts/restart-9119.sh"' -ForegroundColor Cyan
    Write-Host ""
    $choice = Read-Host "按 Enter 继续启动（部分功能不可用），或按 Ctrl+C 取消"
}

# ============ Step 4: 启 electron ============
Write-Host ""
Write-Host "[4/4] 启 Electron ..." -ForegroundColor Cyan
Write-Host ""

# 杀残留（5s 等待）
Get-Process -Name electron, node -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*hermes-agent*" } | ForEach-Object {
    Write-Host "  杀残留 PID $($_.Id) ($($_.ProcessName))"
    Stop-Process -Id $_.Id -Force
}
Start-Sleep -Seconds 2

# 切到 desktop 目录
Set-Location $DESKTOP_DIR
if (-not (Test-Path $DESKTOP_DIR)) {
    Write-Host "[ERROR] Desktop 目录不存在: $DESKTOP_DIR" -ForegroundColor Red
    Read-Host "按 Enter 退出"
    exit 1
}
if (-not (Test-Path $ELECTRON_CLI)) {
    Write-Host "[ERROR] Electron cli.js 不存在: $ELECTRON_CLI" -ForegroundColor Red
    Write-Host "  跑 fix-electron-wrapper.ps1 修复" -ForegroundColor Yellow
    Read-Host "按 Enter 退出"
    exit 1
}

# 启 electron
& $NODE_EXE $ELECTRON_CLI .
