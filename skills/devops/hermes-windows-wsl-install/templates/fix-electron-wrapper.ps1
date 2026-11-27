# fix-electron-wrapper.ps1 - 修复 Electron wrapper（5 文件恢复）
# 用法: powershell -NoProfile -ExecutionPolicy Bypass -File fix-electron-wrapper.ps1
# 作用: 下 electron binary zip + 解压 + 写 5 个 wrapper + 写 .npmrc

$ErrorActionPreference = 'Continue'

$ver = "40.9.3"
$electronRoot = "C:\Users\lujun\AppData\Local\hermes\hermes-agent\node_modules\electron"
$dist = "$electronRoot\dist"
$npmrc = "C:\Users\lujun\AppData\Local\hermes\hermes-agent\.npmrc"

Write-Host "=== Electron Wrapper 修复 ==="
Write-Host "  Version: $ver"
Write-Host "  Root: $electronRoot"
Write-Host ""

# Step 1: 下载 zip
$zip = "$env:TEMP\electron-v$ver.zip"
$url = "https://npmmirror.com/mirrors/electron/v$ver/electron-v$ver-win32-x64.zip"

Write-Host "[1/5] 下载 electron-v$ver-win32-x64.zip ..."
if (-not (Test-Path $zip) -or (Get-Item $zip).Length -lt 100MB) {
    Write-Host "  URL: $url"
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing -TimeoutSec 600
    $zipSize = (Get-Item $zip).Length / 1MB
    Write-Host "  下载完成: $([math]::Round($zipSize, 1))MB"
} else {
    Write-Host "  已存在 (cache)"
}

# Step 2: 解压
Write-Host ""
Write-Host "[2/5] 解压到 $dist ..."
if (Test-Path $dist) {
    Remove-Item -Recurse -Force $dist
}
New-Item -ItemType Directory -Path $dist -Force | Out-Null
Expand-Archive -Path $zip -DestinationPath $dist -Force
$exeSize = (Get-Item "$dist\electron.exe").Length / 1MB
Write-Host "  electron.exe: $([math]::Round($exeSize, 1))MB"

# Step 3: 写 5 个 wrapper 文件
Write-Host ""
Write-Host "[3/5] 写 5 个 wrapper 文件 ..."

# 3.1 package.json
@"
{
  "name": "electron",
  "version": "$ver",
  "main": "index.js",
  "bin": {
    "electron": "cli.js"
  }
}
"@ | Out-File "$electronRoot\package.json" -Encoding UTF8

# 3.2 cli.js
@'
#!/usr/bin/env node
const path = require("path")
const proc = require("child_process")
const fs = require("fs")
const version = fs.readFileSync(path.join(__dirname, "version.txt"), "utf-8").trim()
const args = process.argv.slice(2)
if (args[0] === "--version" || args[0] === "-v") { console.log(version); process.exit(0) }
const electron = path.join(__dirname, "dist", process.platform === "win32" ? "electron.exe" : "electron")
const child = proc.spawn(electron, args, { stdio: "inherit" })
child.on("exit", code => process.exit(code))
'@ | Out-File "$electronRoot\cli.js" -Encoding UTF8

# 3.3 index.js
"module.exports = require('./dist/electron');" | Out-File "$electronRoot\index.js" -Encoding ASCII

# 3.4 path.txt
"v$ver" | Out-File "$electronRoot\path.txt" -Encoding ASCII

# 3.5 version.txt
"v$ver" | Out-File "$electronRoot\version.txt" -Encoding ASCII

Write-Host "  package.json / cli.js / index.js / path.txt / version.txt"

# Step 4: 写 .npmrc
Write-Host ""
Write-Host "[4/5] 写 .npmrc ..."
$npmrcContent = @"
registry=https://registry.npmmirror.com/
audit=false
fund=false
fetch-retries=2
maxsockets=2
electron_mirror=https://npmmirror.com/mirrors/electron/
"@
$npmrcContent | Out-File $npmrc -Encoding ASCII -Force
Write-Host "  $npmrc"

# Step 5: 验证
Write-Host ""
Write-Host "[5/5] 验证 ..."
$files = @("package.json", "cli.js", "index.js", "path.txt", "version.txt", "dist\electron.exe")
$allOK = $true
foreach ($f in $files) {
    $path = Join-Path $electronRoot $f
    if (Test-Path $path) {
        $size = (Get-Item $path).Length
        Write-Host "  OK: $f ($size bytes)"
    } else {
        Write-Host "  X: $f 缺失"
        $allOK = $false
    }
}

Write-Host ""
if ($allOK) {
    Write-Host "=== 修复完成 ===" -ForegroundColor Green
    Write-Host "现在可以双击 Hermes Desktop.lnk 启动桌面"
} else {
    Write-Host "=== 修复失败 ===" -ForegroundColor Red
    Write-Host "看上面 [X] 行"
    exit 1
}
