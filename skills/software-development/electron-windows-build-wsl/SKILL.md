---
name: electron-windows-build-wsl
description: 在 WSL/Linux 环境下构建 Electron Windows 应用 - 解决网络下载失败、TypeScript 兼容性和 NSIS 需要 wine 的问题
---

# Electron Windows Build on WSL

## 常见问题与解决方案

### 1. TypeScript `baseUrl` deprecated 错误

**问题**: `tsconfig.json` 中 `baseUrl` 选项在 TypeScript 6+ 被弃用，编译报错退出

**解决**: 在 `tsconfig.json` 的 `compilerOptions` 中添加:
```json
"ignoreDeprecations": "6.0"
```

### 2. npm install 超时

**问题**: `npm install` 反复超时，无法完成

**解决**: 使用 `--ignore-scripts --no-audit --no-fund` 跳过脚本执行和额外检查:
```bash
npm install --ignore-scripts --no-audit --no-fund
```

### 3. electron-builder 下载 Electron 二进制失败（网络问题）

**问题**: electron-builder 反复下载 Electron zip 但始终损坏，且每次重试耗时很长

**排查步骤**:
```bash
# 验证 zip 文件是否真的损坏
unzip -t ~/.cache/electron/electron-v*.zip

# 如果显示 "No errors detected"，zip 本身是好的
# 问题在 electron-builder 的缓存索引损坏或下载被截断
```

**注意**: `~/.cache/electron/` 中可能有多个 partial 文件（如 `*.zip.part1`, `*.zip.part2`），表示之前下载被截断过。需要清理这些 partial 文件后再解压。

**解决方案 A — 手动缓存法**:
```bash
# 1. 手动下载 Electron zip（用其他工具如 curl）
# 2. 解压到指定目录
unzip -o electron-v28.3.3-win32-x64.zip -d electron-v28.3.3-win32-x64/

# 3. 在 package.json build 配置中指定 electronDist
{
  "build": {
    "electronDist": "/path/to/electron-v28.3.3-win32-x64",
    "win": { "target": "dir" }
  }
}
```

**解决方案 B — 预下载缓存位置**:
- electron-builder 缓存: `~/.cache/electron/`
- 但其内部索引可能损坏，需确保目录结构符合预期

### 4. NSIS 安装程序需要 wine

**问题**: electron-builder 的 NSIS target 需要 wine，但 WSL 通常没有安装

**解决**: 改用 `dir` target（生成便携版目录结构，无需安装）:
```json
{
  "build": {
    "win": {
      "target": [
        {
          "target": "dir",
          "arch": ["x64"]
        }
      ]
    }
  }
}
```

**输出**: `release/win-unpacked/Harness Desktop.exe` (完整便携版，176MB)

### 5. npm install 后的 electron 包只有占位符

**问题**: `npm install --ignore-scripts` 后 `node_modules/electron/` 目录只有 `package.json`、`cli.js` 等文件，缺少 `dist/` 子目录（electron 二进制从未被下载）

**原因**: `install.js` 下载脚本被 `--ignore-scripts` 跳过了

**解决**: 不需要修复。electron-builder 会自己处理下载，只需要解决网络下载损坏问题（用 `electronDist` 方案）

### 6. 完整构建命令（跳过 tsc）

tsc 可能有大量类型错误（隐式 any、React 类型缺失等），但 Vite 的 esbuild 能正常构建。可跳过 tsc 直接构建:
```bash
# 构建 Web 部分
node_modules/.bin/vite build

# 打包 Electron（使用预缓存的 electronDist）
    node_modules/.bin/electron-builder --win --x64
```

### 验证构建产物

```bash
ls release/win-unpacked/*.exe  # 确认 exe 存在
ls -lh release/win-unpacked/   # 确认文件大小
```

## 关键配置示例

```json
{
  "build": {
    "appId": "com.example.app",
    "productName": "My App",
    "electronDist": "/home/hermes/.cache/electron/electron-v28.3.3-win32-x64",
    "directories": {
      "output": "release"
    },
    "files": [
      "dist/**/*",
      "dist-electron/**/*"
    ],
    "win": {
      "signAndEditExecutable": false,
      "target": [
        {
          "target": "dir",
          "arch": ["x64"]
        }
      ]
    }
  }
}
```

### 7. GitHub Release 上传（gh CLI 未登录时）

**问题**: `gh auth login` token 无效，无法使用 gh CLI

**解决**: 用 Python + requests 直接调用 GitHub API:
```python
import urllib.request, json, base64

TOKEN = "ghp_YOUR_TOKEN"
REPO = "owner/repo"
RELEASE_ID = "123456789"

# 上传小文件（< 100MB）
req = urllib.request.Request(
    f"https://uploads.github.com/repos/{REPO}/releases/{RELEASE_ID}/assets?name=file.exe",
    data=open("file.exe", "rb"),
    headers={"Authorization": f"token {TOKEN}", "Content-Type": "application/octet-stream"},
    method="POST"
)
with urllib.request.urlopen(req, timeout=300) as resp:
    asset = json.loads(resp.read())
    print(asset["browser_download_url"])
```

**超时处理**: 巨文件（>100MB）建议先创建 Release，通过 GitHub 网页手动上传，或使用 git LFS

### 8. electron-builder native module rebuild CI 失败（canvas optional dep）

**问题**: CI 中 electron-builder 尝试 rebuild `canvas` 等 native 模块，但缺少构建工具（GTK/Python/native toolchain）导致失败。`canvas` 是 `pdfjs-dist` 的 optional peer dependency，app 代码中并无直接引用。

**现象**: CI 日志出现 `gyp ERR!` 或 `canvas` rebuild 超时，即使设置了 `npmRebuild: false` 也无效。

**解决方案 — 清理 optional native deps**:
```yaml
# GitHub Actions workflow 中，构建前删除不需要的 optional native 模块
- name: Remove unnecessary native modules
  run: |
    Remove-Item -Recurse -Force node_modules/canvas -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force node_modules/@img -ErrorAction SilentlyContinue
```

**解决方案 — electron-builder 配置**:
```json
{
  "build": {
    "npmRebuild": false,
    "buildDependenciesFromSource": false,
    "nativeRebuilder": false,
    "publish": null
  }
}
```

**注意**: `nativeRebuilder: "skip"` 和 `nativeRebuilder: null` 均无效（electron-builder 对字符串/null 值的处理有 bug），真正起作用的是 `npmRebuild: false` + CI 清理步骤。

**`publish: null`** 防止 electron-builder 尝试自动发布（需要 GH_TOKEN），如果没有有效的发布 token 会导致 CI 失败。

### 9. electron-builder NSIS + GitHub Actions CI 完整配置

```json
{
  "build": {
    "appId": "com.pixelpal.app",
    "productName": "PixelPal",
    "npmRebuild": false,
    "buildDependenciesFromSource": false,
    "publish": null,
    "directories": {
      "output": "release"
    },
    "files": [
      "dist/**/*",
      "dist-electron/**/*"
    ],
    "win": {
      "target": [
        {
          "target": "nsis",
          "arch": ["x64"]
        }
      ]
    },
    "nsis": {
      "oneClick": false,
      "allowToChangeInstallationDirectory": true,
      "createDesktopShortcut": true,
      "createStartMenuShortcut": true
    }
  }
}
```

```yaml
# .github/workflows/build-electron.yml
name: Build Electron

on:
  push:
    branches: [master]

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci --legacy-peer-deps

      - name: Remove unnecessary native modules
        run: |
          Remove-Item -Recurse -Force node_modules/canvas -ErrorAction SilentlyContinue

      - name: Build web
        run: npm run build

      - name: Package Electron
        run: npx electron-builder --win --x64

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: PixelPal-Windows
          path: release/**/*.exe
```

---

- `dir` target 只生成便携版，无安装程序
- 如需 NSIS 安装程序，需要在有 wine 的环境（原生 Linux 或 macOS）打包
- electron-builder 版本 24.13.3 在 WSL2 下网络行为异常
- `npmRebuild: false` 配合 CI 清理 optional native deps 是跳过 rebuild 的有效组合
- `publish: null` 是禁用 auto-release 的正确方式（不是 `false` 或空字符串）
