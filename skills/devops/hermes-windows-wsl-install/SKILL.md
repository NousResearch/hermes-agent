---
name: hermes-windows-wsl-install
description: "Windows + WSL2 + Hermes Agent 桌面端完整安装与互通指南。覆盖 pre-install 必备包（PyPI 7.7MB hermes-agent + 15 个直接依赖 + 9 个 native 间接依赖）、网络报错 7 大坑（npmmirror block / npm 11.11.0 bug / audit 404 / EBUSY / vite build 30s / raw.githubusercontent / WSL 2 NAT）、10 步安装 SOP、共同一个记忆配置（HERMES_HOME env var + X-Hermes-Session-Token + 9119 loopback dashboard）、国内镜像清单（PyPI 清华/阿里/USTC + npmmirror + gh-proxy + gitee）。适用于 hermes-agent v0.16.0+，Windows 10 22H2+ / Windows 11 + WSL 2 Ubuntu 24.04。"
version: 1.0.0
author: lujun2508
license: MIT
platforms: [windows]
metadata:
  hermes:
    tags: [windows, wsl, wsl2, install, pre-install, desktop, network, mirror, gitee, npmmirror, pypi, hermes-home, shared-memory, x-hermes-session-token, loopback-mode, 9119, electron, npm-bug, corporate-proxy]
    related_skills: [devops/hermes-dashboard-remote-access, devops/hermes-update-troubleshooting, devops/nas-z4s-ops, software-development/hermes-agent-skill-authoring]
---

# Hermes Windows + WSL2 完整安装与互通

> 🇨🇳 **为中国大陆 / 公司代理环境避坑而写** —— 在受限网络下从 0 到 1 装好 Hermes 桌面端 + 跟 WSL 互通，**避免各种失败**。

> 适用版本：hermes-agent v0.16.0+（2026.6.5 后）
> 适用环境：Windows 10 22H2+ / Windows 11 + WSL 2 Ubuntu 24.04
> 适用人群：在中国大陆 / 公司代理环境下从 0 到 1 部署 hermes 桌面端 + 跟 WSL 互通的所有人
> **总占用**：≈ 1.2GB（WSL 850 + Windows 340），**7 大网络报错提前绕开**

本 skill 是 2026-06-16 实战沉淀——把"装上去 + 互通 + 共同一个记忆"的全部真实坑位都写清楚。新环境/重装/迁移时**按本文 10 步走**，中途任何一步失败回头查 `references/network-errors.md`。

---

## 一、核心决策（必看，决定后续所有步骤）

| # | 决策 | 为什么 |
|---|------|------|
| 1 | **HERMES_HOME 走 WSL 路径**（`\\wsl$\Ubuntu\home\<user>\.hermes`）| 同一份 config.yaml / .env / sessions / memory / skills |
| 2 | **不用目录软链**（`%LOCALAPPDATA%\hermes` → `\\wsl$...\.hermes`）| WSL 端 hermes-agent 是 git clone + venv，Windows 端是 pilotdeck 装的 .venv，软链会破坏 venv |
| 3 | **WSL dashboard 用 loopback 模式**（`--host 127.0.0.1`）| 0.0.0.0 模式触发 OAuth gate，强制 basic auth 不认 `X-Hermes-Session-Token` |
| 4 | **固定 token**（`HERMES_DASHBOARD_SESSION_TOKEN=...`）| 随机 token 每次启都要重配；固定 token 写 `dashboard.token` 文件 + env var |
| 5 | **手动装 Electron 40.9.3**（zip + wrapper + .npmrc）| 绕开 npm 11.11.0 `Exit handler never called!` bug + npmmirror 代理 ECONNRESET |
| 6 | **不升 Windows 端 hermes-agent**（用 pilotdeck 装的版本）| 同步 WSL 端会覆盖 .venv，得重建，**代价 > 收益** |

---

## 二、7.7MB 实际包清单（hermes-agent 真实依赖树）

**注意**：官方 PyPI 包 `hermes-agent-0.16.0-py3-none-any.whl` 是 **7.7MB**（不是 7.2MB）—— 7.2MB 是早期版本，0.16.0 涨到 7.7MB。装完所有依赖后总占用约 **45-55MB**（含 transitive deps）。

### 2.1 PyPI 端：15 个直接依赖

`pip show hermes-agent` 输出的 `Requires`：

| 包名 | 用途 | 大小约 | 镜像 |
|------|------|-------|------|
| `croniter` | 解析 cron 表达式 | ~50KB | PyPI 清华 |
| `fire` | CLI 参数解析 | ~100KB | PyPI 清华 |
| `httpx` | 异步 HTTP 客户端 | ~150KB | PyPI 清华 |
| `jinja2` | 模板引擎 | ~200KB | PyPI 清华 |
| `openai` | OpenAI 兼容 SDK | ~500KB | PyPI 清华 |
| `prompt_toolkit` | 交互式提示 | ~600KB | PyPI 清华 |
| `psutil` | 系统/进程信息 | ~500KB | PyPI 清华 |
| `pydantic` | 数据验证 | ~3MB | PyPI 清华 |
| `PyJWT` | JWT 编解码（**锁 2.12.1**，硬钉）| ~50KB | PyPI 清华 |
| `python-dotenv` | .env 解析 | ~30KB | PyPI 清华 |
| `pyyaml` | YAML 解析 | ~500KB | PyPI 清华 |
| `requests` | HTTP 客户端 | ~200KB | PyPI 清华 |
| `rich` | 富文本终端 | ~500KB | PyPI 清华 |
| `ruamel.yaml` | YAML（保留注释）| ~500KB | PyPI 清华 |
| `tenacity` | 重试逻辑 | ~50KB | PyPI 清华 |

### 2.2 WSL 端：5 个系统包

| 包名 | 用途 | apt 镜像 |
|------|------|----------|
| `build-essential` | 编译 native modules（node-pty 等）| 阿里云 mirror.aliyun.com |
| `python3.12 python3.12-venv python3-dev` | hermes-agent 运行环境 | 阿里云 |
| `libssl-dev libffi-dev` | web UI HTTPS | 阿里云 |
| `git` | 克隆 hermes-agent | 阿里云 |
| `curl` | 探针 + 下载 | 阿里云 |

### 2.3 Windows 端：2 个系统包

| 包名 | 用途 | 来源 |
|------|------|------|
| **Node.js 20+**（实测 v22.x，**不要** Windows Store 版本）| 跑 electron + desktop | 清华镜像 npmmirror（避免 npm 11 bug）|
| **PowerShell 5.1+** | 启动脚本 | Windows 自带 |

### 2.4 手动装：Electron 40.9.3（213MB）

**不走 npm**——完整内容见 `references/manual-electron-install.md`。总结：

| 文件 | 来源 | 大小 |
|------|------|------|
| `electron-v40.9.3-win32-x64.zip` | `https://npmmirror.com/mirrors/electron/v40.9.3/` | 138MB |
| 解压后 `node_modules/electron/dist/` | 浏览器下载 | 213MB |
| 5 个 wrapper 文件（package.json / cli.js / index.js / path.txt / version.txt）| 手写 | <1KB |

---

## 三、国内镜像清单（哪些走什么）

| 资源类型 | 推荐镜像 | 不推荐 / 走官网 | 原因 |
|---------|---------|----------------|------|
| **PyPI**（hermes-agent + 15 直接依赖）| `https://pypi.tuna.tsinghua.edu.cn/simple` 或 `https://mirrors.aliyun.com/pypi/simple/` | ❌ `https://pypi.org/simple`（实测 333ms 但下载慢）| 国内公司代理对 PyPI 不限速，但稳定性清华 > 阿里 > USTC |
| **npm registry**（electron 之外的 npm 包）| `https://registry.npmmirror.com/` | ❌ `https://registry.npmjs.org`（npm 11 bug 高发）| npmmirror 是淘宝镜像，国内 npm 唯一稳的 |
| **Electron binary** | `https://npmmirror.com/mirrors/electron/v40.9.3/electron-v40.9.3-win32-x64.zip` | ❌ `https://github.com/electron/electron/releases`（GFW block）| npmmirror 镜像 electron |
| **hermes-agent 源码** | `https://gh-proxy.com/https://github.com/NousResearch/hermes-agent.git` | ❌ `https://github.com/.../hermes-agent.git`（GFW block）| gh-proxy.com 是 GitHub 加速器，反代 raw.githubusercontent.com |
| **raw.githubusercontent.com** | `https://gh-proxy.com/https://raw.githubusercontent.com/...` | ❌ 直连（GFW block）| 同上 |
| **gitee 镜像**（可选，国内备用）| `https://gitee.com/mirrors/hermes-agent`（jun哥 6/16 实战创建）| ❌ 无 | Gitee 同步 GitHub 仓库，全文镜像 |

**镜像配置**（WSL 端 `/etc/pip.conf` + Windows 端 `%APPDATA%\pip\pip.ini`）：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60
```

**npm 配置**（`C:\Users\<user>\.npmrc`）：

```ini
registry=https://registry.npmmirror.com/
audit=false
fund=false
fetch-retries=2
maxsockets=2
```

---

## 四、Pre-Install 必备环境

| # | 必备项 | 验证命令 |
|---|-------|---------|
| 1 | Windows 10 22H2+ / Windows 11 | `winver` 看版本号 |
| 2 | WSL 2（不是 WSL 1）| `wsl --status` 看 "默认版本: 2" |
| 3 | Ubuntu 24.04 | `wsl -l -v` |
| 4 | 公司代理（如果在办公室）| `curl -x http://127.0.0.1:7897 https://pypi.org/ -I` |
| 5 | 端口 9119 未占用 | `netstat -an \| grep 9119` 应空 |
| 6 | `~/.hermes/.env` 必备字段：`SN_API_KEY` / `AGNES_API_KEY` / `SN_BASE_URL=https://token.sensenova.cn/v1` / `WECHAT_APPID` / `WECHAT_SECRET` | `cat ~/.hermes/.env` |
| 7 | `~/.local/bin/hermes` 可用（**正确路径**，不是 `~/.hermes/.venv/bin/hermes.exe`）| `which hermes` |

**WSL 2 关键**：`localhostForwarding=true`（默认开）让 Windows `127.0.0.1:9119` 直接通 WSL 9119。**不要**用 netsh portproxy（WSL 2 不需要）。

---

## 五、10 步安装 SOP

**完整内容见 `references/preinstall-sop.md`**，本节给概要：

```
Step 1: WSL 装 Ubuntu 24.04         (wsl --install -d Ubuntu-24.04)
Step 2: WSL 端装包                   (apt install build-essential + python3.12-venv + git + curl + libssl-dev)
Step 3: WSL 端配 pip 镜像            (/etc/pip.conf → tsinghua)
Step 4: WSL 端 git clone hermes-agent + venv + pip install -e
         (走 gh-proxy 镜像)
Step 5: WSL 端配 .env                (SN_API_KEY + AGNES_API_KEY + WECHAT_*)
Step 6: WSL 端验 hermes doctor       (hermes doctor 应全 ✓)
Step 7: WSL 端启 9119 dashboard      (loopback 模式 + 固定 token, 等 30s)
Step 8: Windows 装 Node.js 22+       (从 nodejs.org 官网, 不要 Windows Store)
Step 9: Windows 端手动装 Electron    (zip + wrapper + .npmrc, 绕开 npm 11 bug)
Step 10: Windows 端配 3 env vars + 启 desktop
```

**单步详细命令**：见 `references/preinstall-sop.md`。**7 大网络报错对应每一步可能撞的坑**：见 `references/network-errors.md`。

---

## 六、共同一个记忆配置（4 步）

**目标**：Windows 端 `hermes.exe` 跟 WSL 端 `hermes` 用**同一份** `~/.hermes/`（config.yaml / .env / sessions.db / memory / skills）。

**4 步**（在 WSL 端 + Windows 端各跑对应命令）：

```bash
# === WSL 端 ===
# 1. 选固定 token（24 字符+）
echo "hermes-shared-2026-06-16" > ~/.hermes/dashboard.token
chmod 600 ~/.hermes/dashboard.token

# 2. 杀旧 9119 + 重启 loopback 模式 + 固定 token
old=$(ps aux | grep "dashboard.*9119" | grep -v grep | awk '{print $2}')
[ -n "$old" ] && kill -9 $old
sleep 2
HERMES_DASHBOARD_SESSION_TOKEN=$(cat ~/.hermes/dashboard.token) \
  /home/lujun/.local/bin/hermes dashboard --no-open --host 127.0.0.1 --port 9119 \
  > /tmp/dashboard_9119.log 2>&1 &
disown
sleep 30  # web UI build 30s
curl -s -H "X-Hermes-Session-Token: $(cat ~/.hermes/dashboard.token)" \
  http://127.0.0.1:9119/api/status
# 期望 HTTP=200
```

```powershell
# === Windows 端 ===
# 1. 写 start-desktop.ps1（已沉淀在 `templates/start-desktop.ps1`）

# 2. 双击 Hermes Desktop.lnk → 自动设 3 env vars + 启 electron
# 3. GUI 5-10s 内弹出，MainWindowTitle=Hermes
```

**完整 4 步曲 + 验证清单**：见 `templates/setup-shared-memory.sh`（WSL 端一键脚本）+ `templates/start-desktop.ps1`（Windows 端启动器）。

---

## 七、启动 + 验证（4 步曲）

```bash
# Step 1: 探 WSL 9119 dashboard 状态
curl -s -o /dev/null -w "%{http_code}\n" \
  -H "X-Hermes-Session-Token: $(cat ~/.hermes/dashboard.token)" \
  http://127.0.0.1:9119/api/status
# HTTP=200 在跑，HTTP=000 没跑，HTTP=401 token 错
```

```bash
# Step 2: 如果 9119 没跑，按六.2 重启
# 等 30s（web UI build），再 Step 1 探针
```

```powershell
# Step 3: 启动 Windows desktop（设 3 env vars + node cli.js）
$env:HERMES_HOME = '\\wsl$\Ubuntu\home\lujun\.hermes'
$env:HERMES_DESKTOP_REMOTE_URL = 'http://127.0.0.1:9119'
$env:HERMES_DESKTOP_REMOTE_TOKEN = (Get-Content '\\wsl$\Ubuntu\home\lujun\.hermes\dashboard.token' -Raw).Trim()
Set-Location 'C:\Users\lujun\AppData\Local\hermes\hermes-agent\apps\desktop'
& 'C:\Program Files\nodejs\node.exe' 'C:\Users\lujun\AppData\Local\hermes\hermes-agent\node_modules\electron\cli.js' '.'
```

```powershell
# Step 4: 验证
Get-Process -Name electron | Select Id, MainWindowTitle
# 期望 5-6 个 electron 进程 + MainWindowTitle="Hermes"
# 若报 "404 /api/media" = 已连上 WSL，部分 API 缺失（不影响 chat/session/memory）
# 若报 "Couldn't start" / "Could not connect" = 9119 没跑，回 Step 1
```

---

## 八、关键文件位置

| 路径 | 作用 | 跨平台 |
|------|------|-------|
| `\\wsl$\Ubuntu\home\lujun\.hermes\` | **共同 HERMES_HOME** | WSL 唯一 / Windows 通过 `\\wsl$` 访问 |
| `\\wsl$\Ubuntu\home\lujun\.hermes\.env` | API keys（SN/AGNES/WECHAT 等）| 共用 |
| `\\wsl$\Ubuntu\home\lujun\.hermes\dashboard.token` | 24B 共享 token | 共用 |
| `\\wsl$\Ubuntu\home\lujun\.hermes\hermes-agent\` | WSL 端 hermes-agent 仓库（git clone）| WSL 唯一 |
| `/home/lujun/.local/bin/hermes` | hermes CLI 入口 | WSL 唯一 |
| `C:\Users\lujun\AppData\Local\hermes\hermes-agent\` | Windows 端 hermes-agent（pilotdeck 装）| Windows 唯一 |
| `C:\Users\lujun\AppData\Local\hermes\hermes-agent\.venv\` | Windows 端 venv | Windows 唯一 |
| `C:\Users\lujun\AppData\Local\hermes\hermes-agent\node_modules\electron\dist\electron.exe` | 手动装 Electron 40.9.3 (213MB) | Windows 唯一 |
| `C:\Users\lujun\AppData\Local\hermes\hermes-agent\.npmrc` | `audit=false, fund=false, registry=npmmirror` | Windows 唯一 |

---

## 九、6 个绝对禁忌

| ❌ 禁忌 | 触发后果 | 正确做法 |
|--------|---------|---------|
| `npm install electron` | npm 11.11.0 必崩（坑 2）+ npmmirror ECONNRESET（坑 1）+ audit 404（坑 3）+ EBUSY（坑 4）几乎必失败 | 手动装 Electron（zip + wrapper + .npmrc）|
| WSL 端用 `~/.hermes/.venv/bin/hermes.exe` | "No such file or directory"（路径不存在）| 用 `~/.local/bin/hermes`（**正确路径**）|
| WSL dashboard 用 `--host 0.0.0.0` | 触发 OAuth gate → main.cjs `X-Hermes-Session-Token` 不认 → 401 | 用 `--host 127.0.0.1`（loopback 模式）|
| 不设 `HERMES_DESKTOP_REMOTE_TOKEN` | main.cjs 抛 "HERMES_DESKTOP_REMOTE_TOKEN not provided" → GUI 报 "Couldn't start" | 3 env vars 必设齐（HERMES_HOME + REMOTE_URL + REMOTE_TOKEN）|
| 跳过 `hermes doctor` 验环境 | 缺 native module 静默失败，启动 4 分钟后才报错 | 装完必跑 `hermes doctor`，5 项全 ✓ 才算装好 |
| 不配公司代理就装 | WSL 端 `git clone` 必败 + `pip install` 必败 + npmmirror ECONNRESET | WSL 端 `export http_proxy=http://$(cat /etc/resolv.conf \| grep nameserver \| awk '{print $2}'):7897` |

---

## 十、故障排除

| 症状 | 真凶 | 修法 |
|------|------|------|
| `pip install hermes-agent` 报 `ECONNRESET` | 坑 1 | 配 pip 清华镜像 |
| `npm install` 报 `Exit handler never called!` | 坑 2 | 不用 `npm install electron`，手动装 |
| `npm audit` 报 `/security/advisories/bulk` 404 | 坑 3 | `.npmrc` 加 `audit=false` |
| `npm install` 报 `EBUSY: rmdir 'node_modules\electron'` | 坑 4 | 杀 electron 进程 + 删 `node_modules\electron` + 重试 |
| `hermes dashboard` 启 30s 没动（log 卡 `Building web UI...`）| 坑 5 | **不是**真卡！等满 30s |
| `git clone` 报 `Could not resolve host: raw.githubusercontent.com` | 坑 6 | 用 `gh-proxy.com` 镜像 |
| Windows 连不上 WSL 9119 | 坑 7 | WSL 2 默认 `localhostForwarding=true` 走通 |
| `hermes doctor` 报 SSL error | 缺 `libssl-dev` | `sudo apt install libssl-dev` |
| 桌面 GUI 弹 5-10s 后立刻退出 | Electron binary 没装 | 跑 Step 9 手动装 Electron |
| 桌面 GUI 报 `404 /api/media` | 部分 API 端点缺失 | **忽略**，不影响 chat/session/memory |

**完整 7 大网络报错 + 修法**：见 `references/network-errors.md`。

---

## 十一、关联 skill

- `devops/hermes-dashboard-remote-access` — 远程 dashboard 鉴权（Basic Auth + cookie 登录）+ **本 skill 来自该 skill P0-15 拆分**（2026-06-16）。loopback 模式 + `X-Hermes-Session-Token` 协议的详细技术细节见该 skill P0-7/P0-8。
- `devops/hermes-update-troubleshooting` — 升级 v0.16.0 → v0.17.0 时的常见坑
- `devops/nas-z4s-ops` — NAS 极空间 Z4S 上跑 hermes 也走相同 remote-backend 模式
- `software-development/hermes-skill-authoring` — 写新 skill 的 SOP（4 文档原则 + PR 流程 + gitee 镜像教程）
- `cross-platform-skill-install` — skill 跨平台安装（Hermes + OpenClaw + Claude Code）

---

## 十二、参考文档（references/）

- `references/package-list-and-mirrors.md` — **PyPI/npm/git 镜像清单 + 15 直接依赖大小表**
- `references/network-errors.md` — **7 大网络报错（真凶-症状-绕开-修法）**
- `references/preinstall-sop.md` — **10 步安装 SOP 完整命令**
- `references/manual-electron-install.md` — 手动装 Electron 40.9.3 详细步骤
- `references/troubleshooting.md` — 故障排除扩展（按错误码 + 场景分类）
- `references/gitee-mirror-setup.md` — gitee 镜像教程（给国内用户拉取用）

## 十三、脚本（scripts/）

- `scripts/preinstall-check.sh` — WSL 端 pre-install 状态一键检查（7 探针）
- `scripts/preinstall-check.ps1` — Windows 端 pre-install 状态一键检查
- `scripts/setup-shared-memory.sh` — 共同一个记忆配置一键脚本（WSL 端）
- `scripts/restart-9119.sh` — 重启 WSL 9119 dashboard（loopback 模式 + 固定 token）

## 十四、模板（templates/）

- `templates/start-desktop.ps1` — Windows 端 hermes desktop 启动器（设 3 env vars + 探针 + 启 electron）
- `templates/setup-shared-memory.sh` — 共同一个记忆配置脚本
- `templates/fix-electron-wrapper.ps1` — 修复 Electron wrapper（5 文件恢复）
- `templates/desktop-organization.ps1` — 桌面文件治理 4 步（hermes-setup/_archive/）

---

**本 skill 实战沉淀日期**：2026-06-16（hermes-agent v0.16.0, upstream c6b0eb4d）
**反馈渠道**：https://github.com/NousResearch/hermes-agent/issues
**国内用户拉取**：见 `references/gitee-mirror-setup.md`（gitee 镜像同步教程）
