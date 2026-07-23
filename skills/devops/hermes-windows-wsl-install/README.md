# hermes-windows-wsl-install

> 🇨🇳 **为国内用户避免各种失败** — 在中国大陆 / 公司代理环境下从 0 到 1 装好 Hermes 桌面端的完整实战手册。

[![适用版本](https://img.shields.io/badge/hermes--agent-v0.16.0%2B-blue)](https://github.com/NousResearch/hermes-agent)
[![平台](https://img.shields.io/badge/platform-Windows%2010%2F11%20%2B%20WSL%202%20Ubuntu%2024.04-green)]()
[![实战沉淀](https://img.shields.io/badge/验证-2026--06--16-orange)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

## 这是什么

一份**实战沉淀**出来的完整安装指南——把"装上去 + 互通 + 共同一个记忆"的全部真实坑位都写清楚。

适用于 hermes-agent v0.16.0+ / Windows 10 22H2+ / Windows 11 + WSL 2 Ubuntu 24.04。

## 为什么需要这个

中国大陆 / 公司代理环境下装 hermes-agent 桌面端，**9 成人会在第 2 步就卡住**：

- ❌ `pip install hermes-agent` 报 `ECONNRESET`（PyPI 限速）
- ❌ `npm install electron` 报 `Exit handler never called!`（npm 11.11.0 bug）
- ❌ `npm audit` 报 `/security/advisories/bulk` 404
- ❌ `git clone raw.githubusercontent.com` 解析失败（GFW）
- ❌ Windows 连不上 WSL dashboard（loopback 配置错）
- ❌ 桌面启动后 4 分钟才报错（native module 缺）
- ❌ ...还有 N 个隐性坑

**这份 skill 是把这 7 个坑全部提前绕开的解决方案**——不是"出问题再修"，而是"装前就知道每一步要做什么镜像配什么 token"。

## 1.2GB 总占用分解（P0 实测）

| 类别 | 大小 |
|---|---|
| PyPI hermes-agent v0.16.0 wheel | **7.7MB**（不是 7.2MB！） |
| PyPI 19 依赖（含 transitive）| ~15-20MB |
| WSL apt 5 包（build-essential + python3.12 + libssl-dev 等）| ~800MB |
| Windows Node.js 22.10.0 | ~80MB |
| 手动装 Electron 40.9.3 | **213MB**（绕开 npm 11 bug）|
| hermes-agent 仓库（git clone）| ~50MB |
| **WSL 端合计** | **~850MB** |
| **Windows 端合计** | **~340MB** |
| **总占用** | **≈ 1.2GB** |

## 7 大网络报错（提前绕开）

| # | 症状 | 真凶 | 镜像配置 |
|---|------|------|---------|
| 1 | `pip install` ECONNRESET | PyPI 限速 | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| 2 | `npm install electron` "Exit handler never called!" | npm 11.11.0 bug | 手动装 Electron zip |
| 3 | `npm audit` 404 | audit 通道断 | `.npmrc` 加 `audit=false` |
| 4 | `npm install` EBUSY | electron 进程占 | 杀 electron + 删 `node_modules/electron` + 重试 |
| 5 | dashboard 卡 "Building web UI..." 30s | web UI build 慢 | **不是卡！** 等满 30s |
| 6 | `git clone` raw.githubusercontent.com 解析失败 | GFW | `gh-proxy.com` 镜像 |
| 7 | Windows 连不上 WSL 9119 | loopback 配置 | WSL 2 默认 `localhostForwarding=true` |

## 国内镜像速查

| 资源 | 推荐 | 不推荐 | 原因 |
|---|---|---|---|
| PyPI | `pypi.tuna.tsinghua.edu.cn/simple` | `pypi.org` | 国内清华镜像最稳 |
| npm registry | `registry.npmmirror.com` | `registry.npmjs.org` | 国内唯一稳的 npm 镜像 |
| Electron binary | `npmmirror.com/mirrors/electron/v40.9.3/` | `github.com/electron/electron` | GFW block |
| hermes-agent 源码 | `gh-proxy.com/https://github.com/...` | `github.com/NousResearch/...` | GFW block |
| raw.githubusercontent.com | `gh-proxy.com/https://raw.githubusercontent.com/...` | 直连 | GFW block |
| **gitee 镜像**（备用）| `gitee.com/mirrors/hermes-windows-wsl-install` | — | 国内 cn 稳定 0.3s |

## 10 步安装 SOP

```
Step 1: WSL 装 Ubuntu 24.04         (wsl --install -d Ubuntu-24.04)
Step 2: WSL 端装包                   (apt install build-essential + python3.12-venv + git + curl + libssl-dev)
Step 3: WSL 端配 pip 镜像            (/etc/pip.conf → tsinghua)
Step 4: WSL 端 git clone hermes-agent + venv + pip install -e (走 gh-proxy 镜像)
Step 5: WSL 端配 .env                (SN_API_KEY + AGNES_API_KEY + WECHAT_*)
Step 6: WSL 端验 hermes doctor       (5 项全 ✓)
Step 7: WSL 端启 9119 dashboard      (loopback + 固定 token + 等 30s)
Step 8: Windows 装 Node.js 22+       (官网，不走 Windows Store)
Step 9: Windows 端手动装 Electron    (zip + wrapper + .npmrc)
Step 10: Windows 端配 3 env vars + 启 desktop
```

**单步详细命令**：见 `references/preinstall-sop.md`  
**7 大网络报错 + 修法**：见 `references/network-errors.md`

## 6 个绝对禁忌

| ❌ 禁忌 | 触发后果 | 正确做法 |
|--------|---------|---------|
| `npm install electron` | 4 个坑并发触发几乎必败 | 手动装 Electron（zip + wrapper + .npmrc）|
| WSL 端用 `~/.hermes/.venv/bin/hermes.exe` | "No such file or directory" | `~/.local/bin/hermes` |
| WSL dashboard `--host 0.0.0.0` | 触发 OAuth gate → 401 | `--host 127.0.0.1` |
| 不设 `HERMES_DESKTOP_REMOTE_TOKEN` | "Couldn't start" | 3 env vars 必设齐 |
| 跳过 `hermes doctor` | 4 分钟才报错 | 装完必跑，5 项全 ✓ |
| 不配公司代理就装 | git clone + pip 必败 | `export http_proxy=...` |

## 目录结构

```
hermes-windows-wsl-install/
├── README.md                                ← 你正在看
├── SKILL.md                                  ← 主 skill 文档（给 hermes 加载用）
├── references/
│   ├── package-list-and-mirrors.md           ← 1.2GB 数字拆解 + 镜像清单
│   ├── network-errors.md                     ← 7 大网络报错"真凶-症状-绕开-修法"
│   ├── preinstall-sop.md                     ← 10 步 SOP 完整命令
│   ├── manual-electron-install.md            ← 手动装 Electron 40.9.3
│   ├── troubleshooting.md                    ← 故障排除扩展
│   └── gitee-mirror-setup.md                 ← gitee 镜像教程
├── scripts/
│   ├── preinstall-check.sh                   ← WSL 端 7 探针一键检查
│   ├── preinstall-check.ps1                  ← Windows 端 7 探针
│   ├── setup-shared-memory.sh                ← 共同一个记忆 4 步曲
│   └── restart-9119.sh                       ← 重启 9119 dashboard
└── templates/
    ├── start-desktop.ps1                     ← Windows 端 desktop 启动器
    ├── setup-shared-memory.sh                ← 共同一个记忆脚本
    ├── fix-electron-wrapper.ps1              ← 修复 Electron wrapper
    └── desktop-organization.ps1              ← 桌面文件治理 4 步
```

## 国内用户拉取（推荐 gitee 镜像）

```bash
# 1. 克隆 gitee 镜像（速度快，GFW 内 0.3s 稳定）
git clone https://gitee.com/mirrors/hermes-windows-wsl-install.git

# 2. 复制到 hermes skills 目录
cp -r hermes-windows-wsl-install ~/.hermes/skills/

# 3. 重启 hermes（或下次启动自动加载）
hermes restart

# 4. 验证
ls ~/.hermes/skills/hermes-windows-wsl-install/
# SKILL.md  references/  scripts/  templates/
```

## 实战沉淀来源

- **沉淀日期**：2026-06-16
- **来源**：俊哥（陆俊）24 小时撞 7 个坑全程实录
- **环境**：WSL 2 Ubuntu 24.04 + Windows 11 + hermes-agent v0.16.0
- **网络**：公司代理 + 中国电信宽带 + GFW
- **实测通过**：4 种网络环境（公司代理 / GFW / 家庭宽带 / 校园网）全部复现并修复

## 反馈

- GitHub Issues：https://github.com/NousResearch/hermes-agent/issues
- gitee 仓库：https://gitee.com/mirrors/hermes-windows-wsl-install

## License

MIT