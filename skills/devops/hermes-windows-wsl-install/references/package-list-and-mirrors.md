# 实际包清单 + 国内镜像清单

> 适用版本：hermes-agent v0.16.0+ (2026-6-5)
> 验证日期：2026-06-16

## 一、PyPI 包：hermes-agent 实际是 7.7MB（不是 7.2MB）

**实测命令**：
```bash
pip download hermes-agent --no-deps -d /tmp/hermes-pkg
# Downloading hermes_agent-0.16.0-py3-none-any.whl (7.7 MB)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.7/7.7 MB 11.0 MB/s  0:00:01
ls -la /tmp/hermes-pkg/
# -rw-r--r-- 1 lujun lujun 7709966 Jun 16 15:07 hermes_agent-0.16.0-py3-none-any.whl
```

`hermes-agent-0.16.0-py3-none-any.whl` = **7,709,966 bytes = 7.7MB**

**注**：jun哥之前说"7.2M"是早期版本（v0.14 / v0.15 早期），0.16.0 涨到 7.7MB。

**装完所有依赖后总占用**：约 **45-55MB**（含 transitive deps）。

## 二、PyPI 15 个直接依赖

`pip show hermes-agent` 输出：

```
Requires: croniter, fire, httpx, jinja2, openai, prompt_toolkit, psutil,
          pydantic, PyJWT, python-dotenv, pyyaml, requests, rich,
          ruamel.yaml, tenacity
```

| 包名 | 版本 | 大小约 | 用途 | 镜像 | 验证 |
|------|------|-------|------|------|------|
| `croniter` | 6.0.0 | ~50KB | 解析 cron 表达式（hermes cron 用）| 清华/阿里 | `pip show croniter` |
| `fire` | 0.7.1 | ~100KB | CLI 参数解析 | 清华 | 同上 |
| `httpx` | 0.28.1 | ~150KB | 异步 HTTP 客户端 | 清华 | 同上 |
| `httpx-sse` | 0.4.3 | ~30KB | httpx SSE 支持（transitive）| 清华 | 同上 |
| `Jinja2` | 3.1.6 | ~200KB | 模板引擎 | 清华 | 同上 |
| `openai` | 2.24.0 | ~500KB | OpenAI 兼容 SDK | 清华 | 同上 |
| `prompt_toolkit` | 3.0.52 | ~600KB | 交互式提示（TUI）| 清华 | 同上 |
| `psutil` | 7.2.2 | ~500KB | 系统/进程信息 | 清华 | 同上 |
| `pydantic` | 2.13.4 | ~3MB | 数据验证 | 清华 | 同上 |
| `pydantic_core` | 2.46.4 | ~3MB | pydantic 核心（transitive）| 清华 | 同上 |
| `pydantic-settings` | 2.14.1 | ~50KB | pydantic settings（transitive）| 清华 | 同上 |
| `PyJWT` | **2.12.1**（**硬钉，不要升 2.13.0**）| ~50KB | JWT 编解码 | 清华 | 同上 |
| `python-dotenv` | 1.2.2 | ~30KB | .env 解析 | 清华 | 同上 |
| `PyYAML` | 6.0.3 | ~500KB | YAML 解析 | 清华 | 同上 |
| `requests` | 2.33.0 | ~200KB | HTTP 客户端 | 清华 | 同上 |
| `requests-toolbelt` | 1.0.0 | ~100KB | requests 工具（transitive）| 清华 | 同上 |
| `rich` | 14.3.3 | ~500KB | 富文本终端 | 清华 | 同上 |
| `ruamel.yaml` | 0.18.17 | ~500KB | YAML（保留注释）| 清华 | 同上 |
| `ruamel.yaml.clib` | 0.2.15 | ~200KB | C 加速（transitive）| 清华 | 同上 |
| `tenacity` | 9.1.4 | ~50KB | 重试逻辑 | 清华 | 同上 |

## 三、WSL 端 5 个系统包

| 包名 | apt 名 | 用途 | 镜像 |
|------|-------|------|------|
| 1 | `build-essential` | 编译 native modules（node-pty 等）| mirror.aliyun.com |
| 2 | `python3.12 python3.12-venv python3-dev` | hermes-agent 运行环境 | mirror.aliyun.com |
| 3 | `libssl-dev libffi-dev` | web UI HTTPS / cffi 依赖 | mirror.aliyun.com |
| 4 | `git` | 克隆 hermes-agent + 后续升级 | mirror.aliyun.com |
| 5 | `curl` | 探针 + 下载 zip | mirror.aliyun.com |

**Ubuntu 24.04 装包命令**：

```bash
# 配 apt 镜像（mirror.aliyun.com）
sudo sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources
sudo apt update

# 装包
sudo apt install -y build-essential python3.12 python3.12-venv python3-dev \
                    libssl-dev libffi-dev git curl
```

## 四、Windows 端 2 个系统包

| # | 包 | 来源 | 备注 |
|---|----|------|------|
| 1 | **Node.js 20+**（实测 v22.10.0）| https://nodejs.org/dist/v22.10.0/node-v22.10.0-x64.msi | **不要**用 Windows Store 版本（路径权限问题）|
| 2 | **PowerShell 5.1+** | Windows 自带 | 启动脚本需 ExecutionPolicy Bypass |

**验证**：
```powershell
node --version
# v22.10.0
powershell -Command "$PSVersionTable.PSVersion"
# Major  Minor  Build  Revision
# 5      1      ...
```

## 五、Electron 40.9.3（手动装，213MB）

**完整步骤**：见 `manual-electron-install.md`

**摘要**：

| 文件 | 来源 | 大小 | 验证 |
|------|------|------|------|
| `electron-v40.9.3-win32-x64.zip` | `https://npmmirror.com/mirrors/electron/v40.9.3/electron-v40.9.3-win32-x64.zip` | 138MB | `ls -la` |
| `node_modules/electron/dist/electron.exe` | 解压 | 213MB | `Test-Path` |
| 5 个 wrapper 文件 | 手写 | <1KB | 完整 5 文件 |

## 六、国内镜像清单（哪些走什么）

| 资源类型 | 推荐镜像 | 走官网 | 实测延迟 | 备注 |
|---------|---------|--------|---------|------|
| **PyPI**（hermes-agent + 15 直接依赖）| `https://pypi.tuna.tsinghua.edu.cn/simple` | `https://pypi.org/simple` | 清华 ~0.6s / 官网 ~0.3s | 国内公司代理对 PyPI 不限速，但稳定性清华 > 阿里 > USTC |
| **PyPI 备选 1** | `https://mirrors.aliyun.com/pypi/simple/` | 同上 | 阿里 ~1.9s | 阿里有 `pypi/simple` 但部分包缺 |
| **PyPI 备选 2** | `https://pypi.mirrors.ustc.edu.cn/simple` | 同上 | USTC ~0.4s | USTC 最快但只返 301（需带路径）|
| **npm registry** | `https://registry.npmmirror.com/` | `https://registry.npmjs.org` | 镜像 ~0.1s | 国内 npm 唯一稳的镜像 |
| **Electron binary** | `https://npmmirror.com/mirrors/electron/v40.9.3/...` | `https://github.com/electron/electron/releases/download/v40.9.3/...` | 镜像 ~30s / 官网 GFW block | 必走镜像 |
| **hermes-agent 源码** | `https://gh-proxy.com/https://github.com/NousResearch/hermes-agent.git` | `https://github.com/NousResearch/hermes-agent.git` | gh-proxy ~0.5s / 官网 GFW block | gh-proxy.com 是 GitHub 加速器 |
| **raw.githubusercontent.com** | `https://gh-proxy.com/https://raw.githubusercontent.com/...` | 直连 | gh-proxy 0.3s / 直连 GFW block | gh-proxy 反代 |
| **gitee 镜像**（jun哥 6/16 实战）| `https://gitee.com/mirrors/hermes-agent` | 无 | 0.3s | Gitee 同步 GitHub，全文镜像 |

**镜像配置**：

```bash
# === WSL 端：/etc/pip.conf ===
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60
```

```ini
# === Windows 端：%APPDATA%\pip\pip.ini ===
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60
```

```ini
# === Windows 端：C:\Users\<user>\.npmrc ===
registry=https://registry.npmmirror.com/
audit=false
fund=false
fetch-retries=2
maxsockets=2
electron_mirror=https://npmmirror.com/mirrors/electron/
```

## 七、装完总占用

| 类别 | 大小 |
|------|------|
| PyPI hermes-agent + 19 个依赖 | ~15-20MB |
| WSL apt 5 个包（含 build-essential）| ~800MB |
| Windows Node.js 22.10.0 | ~80MB |
| 手动装 Electron 40.9.3 | 213MB |
| hermes-agent 仓库（git clone）| ~50MB |
| **合计 WSL 端** | **~850MB** |
| **合计 Windows 端** | **~340MB** |
| **共同一个记忆占用** | `~/.hermes/` ~10-100MB（视 sessions / skills / memory）|

## 八、卸载/清理命令

```bash
# WSL 端卸载 hermes-agent（保留 .hermes 数据）
pip uninstall hermes-agent
rm -rf ~/.hermes/hermes-agent

# 完整清理（连同 .hermes 数据，谨慎！）
pip uninstall -y -r <(pip show hermes-agent | grep ^Requires | sed 's/Requires: //;s/, /\n/g')
rm -rf ~/.hermes

# Windows 端
"C:\Users\lujun\AppData\Local\Programs\Python\Python311\python.exe" -m pip uninstall hermes-agent
# 手动删 C:\Users\lujun\AppData\Local\hermes\
```
