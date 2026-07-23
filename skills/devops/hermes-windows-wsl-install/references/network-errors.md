# 7 大网络报错（真凶-症状-绕开-修法）

> 实战沉淀：2026-06-16 安装 hermes-agent v0.16.0 + Electron 40.9.3 + WSL dashboard 9119
> 适用人群：在中国大陆 / 公司代理环境下从 0 到 1 部署的人

## 报错 1：PyPI ECONNRESET（公司代理 block）

**真凶**：公司代理（Clash Verge 7897）对 `pypi.org` 的特殊处理——TCP 握手成功但数据流被拦。

**症状**：
```bash
pip install hermes-agent
# ERROR: Could not install packages due to an OSError: [Errno 104] Connection reset by peer
```

**验证**：
```bash
curl -v https://pypi.org/simple/hermes-agent/ 2>&1 | head -20
# 看返回是 200 还是 ECONNRESET
```

**绕开**：配 pip 清华/阿里镜像
```ini
# /etc/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

**验证修法成功**：
```bash
pip install hermes-agent --dry-run
# 应输出 "Would install hermes-agent-0.16.0" 等
```

## 报错 2：npm 11.11.0 `Exit handler never called! error code 1`

**真凶**：npm 11.11.0 与 electron postinstall 的兼容性 bug。npm 11 系列都有，但 11.11.0 触发率最高。

**症状**：
```bash
npm install electron
# ... 下载 electron binary 后 ...
npm error Exit handler never called!
npm error code 1
```

**验证**：
```bash
npm --version
# 输出 11.11.0 = 100% 必触发
```

**绕开**：**完全不用** `npm install electron` —— 手动 zip + 手写 wrapper（见 `manual-electron-install.md`）

**为什么不升级 npm 修**：
- npm 12 改动大，hermes-agent 内部脚本可能不兼容
- 升 npm 11.12+ 还是同 bug
- 改用 pnpm / yarn 会引入新依赖

## 报错 3：npmmirror 镜像没 `/security/advisories/bulk`

**真凶**：npmmirror 是淘宝镜像，**没有** npm 官方的 security advisories endpoint。

**症状**：
```bash
npm install
# npm http fetch GET https://registry.npmmirror.com/-/security/advisories/bulk
# npm ERR! 404 Not Found - GET https://registry.npmmirror.com/-/security/advisories/bulk
```

**绕开**：`.npmrc` 加 `audit=false, fund=false`

**完整 `.npmrc`**：
```ini
registry=https://registry.npmmirror.com/
audit=false
fund=false
fetch-retries=2
maxsockets=2
electron_mirror=https://npmmirror.com/mirrors/electron/
```

## 报错 4：Windows `EBUSY: rmdir 'node_modules\electron'`

**真凶**：electron.exe 还在跑，文件锁（Windows 资源管理器或别的进程）。

**症状**：
```bash
npm install electron
# npm error code EBUSY
# npm error syscall rmdir
# npm error path C:\Users\lujun\AppData\Local\hermes\hermes-agent\node_modules\electron
# npm error EBUSY: resource busy or locked, rmdir 'C:\...\node_modules\electron'
```

**修法 3 步**：
```powershell
# Step 1: 杀残留 electron 进程
Get-Process -Name electron, node | Where-Object { $_.Path -like "*hermes-agent*" } | Stop-Process -Force
Start-Sleep -Seconds 2

# Step 2: 删被锁目录
Remove-Item -Recurse -Force "C:\Users\lujun\AppData\Local\hermes\hermes-agent\node_modules\electron" -ErrorAction SilentlyContinue

# Step 3: 重试
cd "C:\Users\lujun\AppData\Local\hermes\hermes-agent"
npm install
```

## 报错 5：`hermes dashboard` `Building web UI...` 卡 30s+

**真凶**：dashboard 启动时强制跑 `_build_web_ui()`（main.py:4750），先 `tsc -b` 再 `vite build`。v8.0.16 vite 实际编译 ~660ms，但前置 npm install + rollup 总耗时 ~30s。

**症状**：
```bash
hermes dashboard --host 127.0.0.1 --port 9119
# 输出：
# → Building web UI...
# ... 30s 没动静 ...
```

**绕开**：**不是真卡死**！等满 30s 后探针 HTTP=200 即正常。

**验证**：
```bash
# 看 log 进度
tail -f /tmp/dashboard_9119.log
# 看到 "→ Building web UI..." → 等
# 看到 "HERMES_DASHBOARD_READY port=9119" → 起来了
# 看到 "✓ Web UI built" → vite build 完成

# 端口探针
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:9119/api/status
# HTTP=200 正常，HTTP=000 还没起来（继续等）
```

**生产环境优化**（未来）：
- 加 `--no-build` 标志（**hermes CLI 当前不支持**，需提 issue）

## 报错 6：`Can't reach raw.githubusercontent.com`（WSL 拉代码）

**真凶**：GFW 屏蔽 `raw.githubusercontent.com`。

**症状**：
```bash
git clone https://github.com/NousResearch/hermes-agent.git
# fatal: unable to access 'https://github.com/...': Could not resolve host: github.com
```

**或**：
```bash
curl https://raw.githubusercontent.com/.../file
# curl: (6) Could not resolve host: raw.githubusercontent.com
```

**绕开**：用 `gh-proxy.com` 镜像
```bash
# git clone 走 gh-proxy
git clone https://gh-proxy.com/https://github.com/NousResearch/hermes-agent.git

# curl raw 走 gh-proxy
curl https://gh-proxy.com/https://raw.githubusercontent.com/NousResearch/hermes-agent/main/README.md
```

**或**：gitee 镜像（jun哥 6/16 实战）
```bash
git clone https://gitee.com/mirrors/hermes-agent
```

## 报错 7：WSL 2 NAT 网络——Windows 连不上 WSL 服务

**真凶**：WSL 2 走 NAT 转换。WSL 内部服务绑 127.0.0.1 时，Windows 端 `127.0.0.1:<port>` 默认通（WSL 2 `localhostForwarding=true`），但**如果**改用 WSL IP `172.x.x.x:<port>` 或自定义端口转发，可能撞 firewall。

**症状**：
```powershell
# Windows 端
curl http://127.0.0.1:9119/api/status
# curl : Unable to connect to the remote server
```

**或**：
```powershell
# 想用 WSL IP
wsl hostname -I
# 172.20.100.50
curl http://172.20.100.50:9119/api/status
# connection refused / timeout
```

**绕开 1（推荐）**：**用 `127.0.0.1` 不要用 WSL IP**
```bash
# WSL 端
hermes dashboard --host 127.0.0.1 --port 9119
# Windows 端
curl http://127.0.0.1:9119/api/status  # 走通
```

**绕开 2（如果用 WSL IP）**：Windows firewall 放行
```powershell
New-NetFirewallRule -DisplayName "WSL 9119" -Direction Inbound -LocalPort 9119 -Protocol TCP -Action Allow
```

**绝对禁忌**：
- ❌ **不要用** `netsh portproxy`（WSL 2 不需要，且会引入新坑）
- ❌ **不要用** WSL 1（无 `localhostForwarding`）
- ❌ **不要在 WSL 端** `sudo ufw allow 9119`（WSL 2 无 ufw）

## 七大坑速查表

| # | 报错 | 触发步骤 | 修法 |
|---|------|---------|------|
| 1 | pip ECONNRESET | Step 4 `pip install -e` | 配 pip 清华镜像 |
| 2 | npm Exit handler | Step 9 `npm install electron` | 手动装 Electron（zip + wrapper）|
| 3 | npm audit 404 | Step 9 `npm install` | `.npmrc` 加 `audit=false` |
| 4 | npm EBUSY | Step 9 `npm install electron` | 杀进程 + 删 `node_modules\electron` + 重试 |
| 5 | dashboard 30s 卡 | Step 7 `hermes dashboard` | 等 30s 后探针 |
| 6 | raw.githubusercontent.com | Step 4 `git clone` | 用 gh-proxy / gitee 镜像 |
| 7 | Windows 连不上 9119 | Step 10 启 desktop | 用 `127.0.0.1` 不用 WSL IP |

## 诊断脚本（一键检查 7 大坑）

```bash
# 跑 scripts/preinstall-check.sh 会输出 7 探针 PASS/WARN/FAIL
bash scripts/preinstall-check.sh

# 输出示例：
# [PASS] pip 清华镜像已配（/etc/pip.conf）
# [PASS] npm registry 走 npmmirror（.npmrc）
# [PASS] gh-proxy.com 可达
# [WARN] npm 11.11.0 - 必触发 bug 2，建议手动装 Electron
# [FAIL] raw.githubusercontent.com 不可达 - 必走 gh-proxy
# [PASS] WSL 2 + localhostForwarding=true
# [PASS] 端口 9119 未占用
```

## 工具脚本

- `scripts/preinstall-check.sh` — WSL 端 7 探针
- `scripts/preinstall-check.ps1` — Windows 端 7 探针
- `templates/fix-electron-wrapper.ps1` — 修复 Electron wrapper（5 文件恢复）
