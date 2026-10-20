# prompt-optimizer 一键导入脚本（幂等）
# 用途：Hermes 每次更新后，运行本脚本即可重新部署/重新加载提示词优化插件。
# 说明：
#   - 将 plugin.js 从源码主副本复制到 <hermes home>/desktop-plugins/prompt-optimizer/
#     桌面 App 自动 watch desktop-plugins/ 目录，文件落地数秒内自动热加载，
#     无需重启应用；若按钮未出现，在 App 中按 Ctrl+K -> "Reload desktop plugins"。
#   - 后端 API（dashboard/manifest.json + plugin_api.py）由 web_server 在启动时
#     从 <hermes home>/plugins/prompt-optimizer/dashboard/ 挂载（无需部署，
#     主副本即该目录；如被移动则复制一份保险）。
#   - 确保 plugins.enabled 包含 prompt-optimizer（后端路由挂载的必要条件）。
#   - 脚本可重复运行，不会造成重复安装或残留。
# 用法：右键 -> 使用 PowerShell 运行；或 PowerShell 中执行 .\install.ps1

$ErrorActionPreference = 'Stop'

# 源码主副本目录（本脚本所在目录）
$SourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceFile = Join-Path $SourceDir 'plugin.js'

if (-not (Test-Path $SourceFile)) {
    Write-Host "[ERROR] 未找到 $SourceFile，请确认脚本与 plugin.js 在同一目录。" -ForegroundColor Red
    exit 1
}

# 目标：$HERMES_HOME/desktop-plugins/prompt-optimizer/
$HermesHome = $env:HERMES_HOME
if (-not $HermesHome -or -not (Test-Path $HermesHome)) {
    $HermesHome = Join-Path $env:USERPROFILE '.hermes'
}
$DestDir = Join-Path $HermesHome 'desktop-plugins\prompt-optimizer'
New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
Copy-Item -Force $SourceFile (Join-Path $DestDir 'plugin.js')

# 后端 dashboard/ 保险复制（主副本即 <hermes home>/plugins/prompt-optimizer 时可跳过）
$DashboardSrc = Join-Path $SourceDir 'dashboard'
if (Test-Path $DashboardSrc) {
    $BackendDest = Join-Path $HermesHome "plugins\prompt-optimizer\dashboard"
    if ((Resolve-Path $BackendDest -ErrorAction SilentlyContinue) -ne (Resolve-Path $DashboardSrc)) {
        New-Item -ItemType Directory -Force -Path $BackendDest | Out-Null
        Copy-Item -Force (Join-Path $DashboardSrc 'manifest.json') (Join-Path $BackendDest 'manifest.json')
        Copy-Item -Force (Join-Path $DashboardSrc 'plugin_api.py') (Join-Path $BackendDest 'plugin_api.py')
        Write-Host "[OK] 已部署后端 API: $BackendDest"
    }
}

$DestFile = Join-Path $DestDir 'plugin.js'
$Size = (Get-Item $DestFile).Length
Write-Host "[OK] 已部署: $DestFile ($Size bytes)" -ForegroundColor Green
Write-Host "     桌面 App 将在数秒内自动加载该插件。"
Write-Host "     若输入框右侧未出现「优化提示词」按钮，请按 Ctrl+K 执行 Reload desktop plugins。"
Write-Host "     后端 API 路由需重启桌面 App（hermes serve 启动时挂载）后生效。"
