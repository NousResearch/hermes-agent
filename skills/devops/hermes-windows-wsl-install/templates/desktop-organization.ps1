# desktop-organization.ps1 - 桌面文件治理 4 步
# 用法: powershell -NoProfile -ExecutionPolicy Bypass -File desktop-organization.ps1
# 作用: 建分类目录 + 移调试 log + 移早期脚本 + 移核心脚本

$ErrorActionPreference = 'Continue'

$base = 'C:\Users\lujun\Desktop'
$logsDir = "$base\hermes-setup\_archive\logs"
$oldDir = "$base\hermes-setup\_archive\scripts-old"

Write-Host "=== 桌面文件治理 4 步 ==="
Write-Host "  Base: $base"
Write-Host ""

# Step 1: 建目录
Write-Host "[1/4] 建分类目录 ..."
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
New-Item -ItemType Directory -Path $oldDir -Force | Out-Null
New-Item -ItemType Directory -Path "$base\hermes-setup" -Force | Out-Null
Write-Host "  OK: $logsDir"
Write-Host "  OK: $oldDir"

# Step 2: 移 .err/.out/.log
Write-Host ""
Write-Host "[2/4] 移调试 log/err/out ..."
$logs = Get-ChildItem -Path $base -File | Where-Object {
    $_.Extension -in @('.err','.out','.log') -and
    $_.Name -ne 'desktop_hermes_eval_report.md'
}
$count = 0
foreach ($f in $logs) {
    Move-Item -Path $f.FullName -Destination $logsDir -Force
    $count++
}
Write-Host "  移动 $count 个文件"

# Step 3: 移早期迭代脚本
Write-Host ""
Write-Host "[3/4] 移早期迭代脚本 ..."
$oldScripts = @(
    'start-desktop-ps1.ps1','start-desktop-v2.ps1','start-desktop-v3.ps1',
    'fix_desktop_ebusy.ps1','fix_desktop_v2.ps1',
    'find_electron.ps1','find_electron3.ps1',
    'test_hermes_desktop.ps1','test_hermes_desktop2.ps1',
    'manual_install_electron.ps1','manual_install_electron_v2.ps1',
    'manual_install_electron_v3.ps1','manual_install_electron_v4.ps1',
    'manual_install_electron_v5.ps1'
)
$count = 0
foreach ($n in $oldScripts) {
    $p = Join-Path $base $n
    if (Test-Path $p) {
        Move-Item -Path $p -Destination $oldDir -Force
        $count++
    }
}
Write-Host "  移动 $count 个文件"

# Step 4: 移核心脚本到 hermes-setup/
Write-Host ""
Write-Host "[4/4] 移核心脚本到 hermes-setup\ ..."
$keep = @('start-desktop.cmd','start-hermes.ps1','manual_install_electron_v6.ps1')
$count = 0
foreach ($n in $keep) {
    $p = Join-Path $base $n
    if (Test-Path $p) {
        Move-Item -Path $p -Destination "$base\hermes-setup" -Force
        $count++
    }
}
Write-Host "  移动 $count 个文件"

# 总结
Write-Host ""
Write-Host "=== 治理完成 ==="
Write-Host "桌面根目录："
Get-ChildItem -Path $base -File | Where-Object { $_.Name -match 'hermes|Hermes' } | ForEach-Object {
    Write-Host "  $($_.Name)"
}
Write-Host ""
Write-Host "$base\hermes-setup\："
Get-ChildItem -Path "$base\hermes-setup" -File | ForEach-Object {
    Write-Host "  $($_.Name)"
}
