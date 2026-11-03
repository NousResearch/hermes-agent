# 手动装 Electron 40.9.3 详细步骤

> 适用：v0.16.0+ hermes-agent 桌面端，绕开 npm 11.11.0 bug
> 替代：`npm install electron`（**不要用**）

## 为什么手动装

`npm install electron` 在中国大陆/公司代理环境下**几乎必失败**：

| 失败点 | 报错 | 修法 |
|-------|------|------|
| npm 11.11.0 自身 bug | `Exit handler never called! error code 1` | 手动装绕过 |
| npmmirror 代理 ECONNRESET | `fetch error` | 浏览器下载 zip |
| npm audit 404 | `404 /security/advisories/bulk` | `.npmrc` 加 `audit=false` |
| Windows EBUSY | `EBUSY: rmdir 'node_modules\electron'` | 杀进程 + 删目录 + 重试 |

**4 步手动装 30 分钟搞定**（vs `npm install` 各种失败循环 2 小时+）。

## Step 1：下载 electron binary zip（npmmirror 镜像）

```powershell
$ver = "40.9.3"
$url = "https://npmmirror.com/mirrors/electron/v$ver/electron-v$ver-win32-x64.zip"
$zip = "$env:TEMP\electron-v$ver.zip"

Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing -TimeoutSec 600
# 138MB / 30-60s
```

**验证**：
```powershell
Get-Item $zip | Select Name, Length
# Name                       Length
# ----                       ------
# electron-v40.9.3-...zip   144824627  (138MB)
```

## Step 2：解压到 `node_modules/electron/dist/`

```powershell
$dist = "C:\Users\lujun\AppData\Local\hermes\hermes-agent\node_modules\electron\dist"

# 清旧（如有）
if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
New-Item -ItemType Directory -Path $dist -Force | Out-Null

# 解压
Expand-Archive -Path $zip -DestinationPath $dist -Force
# 213MB electron.exe + Chromium 资源
```

**验证**：
```powershell
Get-Item "$dist\electron.exe" | Select Name, Length
# Name          Length
# ----          ------
# electron.exe  223395840  (213MB)
```

## Step 3：写 5 个 wrapper 文件

`node_modules/electron/` 目录下需要 5 个 wrapper 文件（npm 包结构）：

```powershell
$root = "C:\Users\lujun\AppData\Local\hermes\hermes-agent\node_modules\electron"
$ver = "40.9.3"

# 1. package.json
@"
{
  "name": "electron",
  "version": "$ver",
  "main": "index.js",
  "bin": {
    "electron": "cli.js"
  }
}
"@ | Out-File "$root\package.json" -Encoding UTF8

# 2. cli.js
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
'@ | Out-File "$root\cli.js" -Encoding UTF8

# 3. index.js
"module.exports = require('./dist/electron');" | Out-File "$root\index.js" -Encoding ASCII

# 4. path.txt
"v$ver" | Out-File "$root\path.txt" -Encoding ASCII

# 5. version.txt
"v$ver" | Out-File "$root\version.txt" -Encoding ASCII
```

**验证**：
```powershell
Test-Path "$root\package.json"  # True
Test-Path "$root\cli.js"        # True
Test-Path "$root\index.js"      # True
Test-Path "$root\path.txt"      # True
Test-Path "$root\version.txt"   # True
Test-Path "$root\dist\electron.exe"  # True (213MB)
```

## Step 4：写 .npmrc（绕开 audit 404）

**`C:\Users\lujun\AppData\Local\hermes\hermes-agent\.npmrc`**：
```ini
registry=https://registry.npmmirror.com/
audit=false
fund=false
fetch-retries=2
maxsockets=2
electron_mirror=https://npmmirror.com/mirrors/electron/
```

## 验证 Electron 可启动

```powershell
cd "C:\Users\lujun\AppData\Local\hermes\hermes-agent"
& "C:\Program Files\nodejs\node.exe" `
  "C:\Users\lujun\AppData\Local\hermes\hermes-agent\node_modules\electron\cli.js" `
  "C:\Users\lujun\AppData\Local\hermes\hermes-agent\apps\desktop"
# 应在 5-10s 内弹 Electron 窗口
```

## 一键脚本

`templates/fix-electron-wrapper.ps1` 集成上面 4 步，一次跑完。

## 故障排除

| 症状 | 修法 |
|------|------|
| `Test-Path electron.exe` 返 False | 重跑 Step 2 解压 |
| 启 electron 报 "missing dist" | 5 个 wrapper 文件缺一不可 |
| 启 electron 报 "no such file" | `path.txt` 内容错了，应写 `v40.9.3` 不是 `40.9.3` |
| 启 electron 后黑屏 | 缺 `apps/desktop/dist/`（Vite build 产物），需跑 `cd apps/desktop && npm run build` |

## 升级 Electron 版本

```powershell
# 1. 改 $ver = "40.10.0"（最新）
# 2. 跑 Step 1-3（重新下 + 解压 + 写 wrapper）
# 3. 验证
```

**注意**：hermes-agent 内置 Electron 版本可能与桌面版不兼容——升级前看 `hermes-agent/apps/desktop/package.json` 的 `devDependencies.electron` 字段。
