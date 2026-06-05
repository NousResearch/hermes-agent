# Hermes Agent 离线部署指南

将 Hermes Agent 部署到完全无网络的 Linux x86_64 服务器。

## 前置条件

### 开发机（联网 Windows）

| 工具 | 用途 | 安装方法 |
|------|------|---------|
| Git | 源码打包 | https://git-scm.com |
| uv | Python 包管理 | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| Node.js | npm 离线包 + Playwright | https://nodejs.org |
| Docker (可选) | 自动下载 deb 系统依赖 | https://docker.com |

### 目标机（无网络 Linux）

| 工具 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.11 | 必须。3.11-3.13 均可 |
| Git | 任意 | 必须。源码包已包含 .git 信息 |
| root 权限 | - | 安装二进制和系统依赖需要 |

## 快速开始

### 第一步：在开发机准备离线包

```powershell
cd C:\project\hermes-agent
.\scripts\prepare-offline.ps1
```

这会在 `hermes-offline-bundle/` 目录下生成所有离线资源。预计大小 800MB - 1.2GB。

**常用参数：**

```powershell
# 指定输出目录
.\scripts\prepare-offline.ps1 -OutputDir D:\offline-bundle

# 跳过 Playwright Chromium（节省 ~150MB，不使用 browser 工具时）
.\scripts\prepare-offline.ps1 -SkipBrowser

# 跳过 npm 离线包（不使用 web dashboard / TUI 时）
.\scripts\prepare-offline.ps1 -SkipNpm

# 跳过 deb 系统依赖（目标机已有这些库）
.\scripts\prepare-offline.ps1 -SkipDebs
```

### 第二步：传输到目标机

将整个 `hermes-offline-bundle` 目录传输到目标机。方法不限：

- USB 驱动器
- 内网文件共享 (SMB/NFS)
- SCP/SFTP（如果目标机有临时网络）
- 离线介质（光盘、移动硬盘）

```bash
# 示例: 通过 SCP 传输
scp -r hermes-offline-bundle/ user@target:/tmp/
```

### 第三步：在目标机安装

```bash
cd /tmp/hermes-offline-bundle
chmod +x install-offline.sh
sudo ./install-offline.sh
```

**常用参数：**

```bash
# 指定安装目录
sudo ./install-offline.sh --install-dir /opt/hermes

# 指定 Python 版本
sudo ./install-offline.sh --python-version 3.12

# 跳过 Playwright Chromium
sudo ./install-offline.sh --skip-browser

# 跳过前端构建（不使用 dashboard / TUI 时）
sudo ./install-offline.sh --skip-deps-build
```

### 第四步：配置 API Key

```bash
vim ~/.hermes/.env
# 添加至少一个 API key，例如:
#   OPENAI_API_KEY=sk-xxx
#   ANTHROPIC_API_KEY=sk-ant-xxx
```

### 第五步：验证安装

```bash
hermes --help
hermes --version
```

## 离线包目录结构

```
hermes-offline-bundle/
├── binaries/                   # Linux 二进制工具
│   ├── node-v22.x.x-linux-x64.tar.xz
│   ├── ripgrep-*-x86_64-unknown-linux-musl.tar.gz
│   ├── ffmpeg-release-amd64-static.tar.xz
│   └── uv-x86_64-unknown-linux-gnu.tar.gz
├── python-wheels/              # Python wheel 包（Linux x86_64）
│   ├── openai-*.whl
│   ├── anthropic-*.whl
│   └── ... (数百个 .whl 文件)
├── hermes-agent.tar.gz         # 源码打包
├── playwright-browsers/        # Playwright Chromium 浏览器
│   └── chromium-*/
├── npm-offline/                # npm 离线 tarball
│   ├── *.tgz
├── deb-packages/               # Playwright 系统依赖 deb 包
│   ├── libnss3_*.deb
│   ├── libatk1.0-0_*.deb
│   └── ...
├── install-offline.sh          # 目标机安装脚本
└── MANIFEST.txt                # 离线包清单
```

## 安装脚本做了什么

`install-offline.sh` 执行以下步骤：

1. **检查前置条件** — Python 版本、Git、磁盘空间（至少 5GB）
2. **安装 Node.js** — 解压到 `/usr/local/`
3. **安装 ripgrep** — 复制到 `/usr/local/bin/`
4. **安装 ffmpeg** — 复制到 `/usr/local/bin/`
5. **安装 uv** — 复制到 `/usr/local/bin/`
6. **安装 Playwright 系统依赖** — `dpkg -i deb-packages/*.deb`
7. **解压源码** — 到 `/opt/hermes-agent/`
8. **创建 Python venv** — `uv venv venv --python 3.11`
9. **安装 Python 依赖** — `uv pip install --no-index --find-links python-wheels`
10. **安装 npm 依赖** — 从离线 tarball
11. **构建前端** — web dashboard + TUI
12. **安装 Playwright Chromium** — 复制到 `~/.cache/ms-playwright/`
13. **创建 hermes 命令** — `/usr/local/bin/hermes` shim 脚本
14. **初始化配置** — `~/.hermes/.env` + `config.yaml`

## 已安装的功能

离线安装后，以下功能**全部可用**：

| 功能类别 | 功能 | 说明 |
|---------|------|------|
| **推理** | OpenAI, Anthropic, Bedrock, Azure | 所有 provider SDK 已安装 |
| **搜索** | Exa, Firecrawl, Parallel Web | 所有搜索后端 |
| **TTS** | Edge TTS, ElevenLabs, Mistral | 文字转语音 |
| **STT** | Faster Whisper, Mistral | 语音转文字 |
| **图像** | Fal AI | 图像生成 |
| **记忆** | Honcho, Hindsight | 记忆提供商 |
| **消息** | Telegram, Discord, Slack, Matrix, DingTalk, Feishu, WeCom | 所有消息平台 |
| **沙箱** | Modal, Daytona | 终端沙箱后端 |
| **技能** | Google Workspace, YouTube | 集成技能 |
| **工具** | ACP (VS Code), Dashboard, MCP | 所有工具 |
| **浏览器** | Playwright Chromium | 浏览器自动化（需安装 Chromium） |
| **Web** | FastAPI Dashboard | Web 管理界面 |

## 手动补充 deb 包

如果 `prepare-offline.ps1` 未能自动下载 deb 包（没有 Docker），需要在有网络的 Linux 机器上手动下载：

```bash
# 在有网络的 Debian/Ubuntu 机器上执行
mkdir -p /tmp/deb-packages && cd /tmp/deb-packages

apt-get download \
  libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
  libxkbcommon0 libgbm1 libpango-1.0-0 libcairo2 libasound2 \
  libatspi2.0-0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
  libwayland-client0 libwayland-cursor0 libwayland-egl1 \
  libnspr4 libfontconfig1 libfreetype6 libxshmfence1

# 传递依赖（可选但推荐）
apt-cache depends --recurse --no-recommends --no-suggests \
  --no-conflicts --no-breaks --no-replaces --no-enhances \
  libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
  libxkbcommon0 libgbm1 libpango-1.0-0 libcairo2 libasound2 \
  2>/dev/null | grep '^\w' | sort -u | while read dep; do
    apt-get download "$dep" 2>/dev/null || true
  done

# 复制到离线包目录
cp *.deb /path/to/hermes-offline-bundle/deb-packages/
```

## 故障排除

### Python 版本不匹配

```
错误: 需要 Python >= 3.11
```

安装 Python 3.11+。如果系统源中没有：

```bash
# Debian/Ubuntu
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.11

# 或使用 uv 安装
uv python install 3.11
```

### 磁盘空间不足

```
错误: 磁盘空间不足: 可用 2GB, 需要至少 5GB
```

清理磁盘空间或使用 `--install-dir` 指向更大的分区。

### Playwright Chromium 无法运行

```
缺少 libnss3 或 libgbm1
```

确保 deb 包已安装。手动安装：

```bash
sudo apt-get install -y libnss3 libatk1.0-0 libatk-bridge2.0-0 \
  libcups2 libdrm2 libxkbcommon0 libgbm1 libpango-1.0-0 \
  libcairo2 libasound2
```

### npm 安装失败

如果 npm 离线包不完整，可以跳过：

```bash
sudo ./install-offline.sh --skip-deps-build
```

仅影响 web dashboard 和 TUI，核心 CLI 功能不受影响。

### Lazy deps 安装失败

某些 wheel 可能不兼容当前系统（如 `python-olm` 需要特定 glibc 版本）。
查看具体错误：

```bash
uv pip install --no-index --find-links python-wheels mautrix[encryption]==0.21.0
```

如果某个包无法安装，对应的平台功能将不可用（通过 `tools/lazy_deps.py` 按需安装机制），
其他功能不受影响。

## 更新 Hermes

离线环境更新需要重新准备离线包：

1. 在开发机拉取最新代码
2. 重新运行 `.\scripts\prepare-offline.ps1`
3. 传输新的离线包到目标机
4. 运行 `sudo ./install-offline.sh`（会自动备份旧安装）

## 目录说明

| 路径 | 说明 |
|------|------|
| `/opt/hermes-agent/` | 安装目录（源码 + venv） |
| `/opt/hermes-agent/venv/` | Python 虚拟环境 |
| `~/.hermes/` | 用户配置目录 |
| `~/.hermes/.env` | API 密钥配置 |
| `~/.hermes/config.yaml` | 运行时配置 |
| `~/.hermes/skills/` | 用户技能 |
| `~/.hermes/plugins/` | 用户插件 |
| `~/.hermes/logs/` | 日志文件 |
| `~/.cache/ms-playwright/` | Playwright 浏览器 |
| `/usr/local/bin/hermes` | 命令入口 |
| `/usr/local/bin/node` | Node.js |
| `/usr/local/bin/rg` | ripgrep |
| `/usr/local/bin/ffmpeg` | ffmpeg |
| `/usr/local/bin/uv` | uv 包管理器 |
