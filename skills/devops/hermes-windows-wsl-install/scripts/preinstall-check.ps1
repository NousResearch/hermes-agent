# preinstall-check.ps1 - Windows 端 pre-install 状态一键检查（7 探针）
# 用法: powershell -NoProfile -ExecutionPolicy Bypass -File preinstall-check.ps1
# 输出: 7 探针 PASS/WARN/FAIL + 修法

$ErrorActionPreference = 'Continue'

$Pass = 0
$Warn = 0
$Fail = 0

function Check-Pass {
    param([string]$Msg)
    Write-Host "[PASS] $Msg" -ForegroundColor Green
    $script:Pass++
}
function Check-Warn {
    param([string]$Msg)
    Write-Host "[WARN] $Msg" -ForegroundColor Yellow
    $script:Warn++
}
function Check-Fail {
    param([string]$Msg)
    Write-Host "[FAIL] $Msg" -ForegroundColor Red
    $script:Fail++
}

Write-Host "=== Hermes Pre-Install 状态检查（Windows 端）==="
Write-Host ""

# 探针 1: Node.js 已装
Write-Host "1. Node.js ... " -NoNewline
$nodeVer = node --version 2>$null
if ($nodeVer -match "v(\d+)\.") {
    $major = [int]$Matches[1]
    if ($major -ge 20) {
        Check-Pass "Node.js $nodeVer"
    } else {
        Check-Fail "Node.js $nodeVer（需 20+）— 装 v22.10.0: https://nodejs.org/dist/v22.10.0/node-v22.10.0-x64.msi"
    }
} else {
    Check-Fail "Node.js 未装 — 装 v22.10.0"
}

# 探针 2: npm npmmirror
Write-Host "2. npm registry ... " -NoNewline
$npmRegistry = npm config get registry 2>$null
if ($npmRegistry -match "npmmirror.com") {
    Check-Pass "registry=$npmRegistry"
} else {
    Check-Warn "registry=$npmRegistry — 改: npm config set registry https://registry.npmmirror.com/"
}

# 探针 3: .npmrc audit=false
Write-Host "3. .npmrc audit=false ... " -NoNewline
$npmrc = "$env:USERPROFILE\.npmrc"
if (Test-Path $npmrc) {
    if (Select-String -Path $npmrc -Pattern "^audit=false" -Quiet) {
        Check-Pass "$npmrc 已配 audit=false"
    } else {
        Check-Warn "$npmrc 缺 audit=false — 修法: Add-Content $npmrc 'audit=false'"
    }
} else {
    Check-Warn "$npmrc 不存在 — 跑: New-Item $npmrc"
}

# 探针 4: electron binary
Write-Host "4. Electron binary ... " -NoNewline
$electronExe = "C:\Users\lujun\AppData\Local\hermes\hermes-agent\node_modules\electron\dist\electron.exe"
if (Test-Path $electronExe) {
    $size = (Get-Item $electronExe).Length / 1MB
    if ($size -gt 200) {
        Check-Pass "electron.exe $size MB"
    } else {
        Check-Warn "electron.exe 只 $size MB（应 213MB）— 重跑 fix-electron-wrapper.ps1"
    }
} else {
    Check-Fail "electron.exe 不存在 — 跑 templates/fix-electron-wrapper.ps1"
}

# 探针 5: HERMES_HOME
Write-Host "5. HERMES_HOME (env var) ... " -NoNewline
if ($env:HERMES_HOME -match "\\\\wsl\\\$") {
    Check-Pass "HERMES_HOME=$env:HERMES_HOME"
} else {
    Check-Warn "HERMES_HOME 未设或不是 WSL 路径（当前='$env:HERMES_HOME'）"
}

# 探针 6: HERMES_DESKTOP_REMOTE_URL
Write-Host "6. HERMES_DESKTOP_REMOTE_URL ... " -NoNewline
if ($env:HERMES_DESKTOP_REMOTE_URL) {
    Check-Pass "URL=$env:HERMES_DESKTOP_REMOTE_URL"
} else {
    Check-Warn "未设 — start-desktop.ps1 会自动设"
}

# 探针 7: token 文件可读
Write-Host "7. dashboard.token 可读 ... " -NoNewline
$tokenPath = "\\wsl$\Ubuntu\home\lujun\.hermes\dashboard.token"
if (Test-Path $tokenPath) {
    $token = (Get-Content $tokenPath -Raw).Trim()
    if ($token.Length -ge 24) {
        Check-Pass "token 长度 $($token.Length)B"
    } else {
        Check-Warn "token 长度只 $($token.Length)B（应 24+）— 重设: echo 'hermes-shared-2026-06-16' | wsl tee ~/.hermes/dashboard.token"
    }
} else {
    Check-Fail "token 文件不存在 — 在 WSL 跑: echo 'hermes-shared-2026-06-16' > ~/.hermes/dashboard.token"
}

Write-Host ""
Write-Host "=== 总结 ==="
Write-Host "  PASS: $Pass / 7"
Write-Host "  WARN: $Warn / 7"
Write-Host "  FAIL: $Fail / 7"
Write-Host ""

if ($Fail -gt 0) {
    Write-Host "X  有 $Fail 项必修复，按上面 [FAIL] 行的修法跑" -ForegroundColor Red
    exit 1
}

if ($Warn -gt 0) {
    Write-Host "!  有 $Warn 项建议优化，不影响启动" -ForegroundColor Yellow
    exit 0
}

Write-Host "OK  7/7 PASS，pre-install 环境完美！" -ForegroundColor Green
exit 0
