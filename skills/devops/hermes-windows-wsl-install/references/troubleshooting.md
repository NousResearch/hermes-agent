# 故障排除扩展（按错误码 + 场景分类）

> 实战沉淀：2026-06-16 hermes-agent v0.16.0 + WSL 2 + Windows 11
> 配合主 SKILL.md § 十故障排除 + `references/network-errors.md` 一起看

## 一、按错误码分类

### ECONNRESET / Connection reset

| 触发 | 真凶 | 修法 |
|------|------|------|
| `pip install` | 公司代理对 PyPI 拦截 | 配 pip 清华镜像 |
| `npm install` | 公司代理对 npm registry 拦截 | 配 npm npmmirror |
| `git clone` | GFW block github.com | 用 gh-proxy 镜像 |
| `curl raw.githubusercontent.com` | GFW block | 用 gh-proxy 镜像 |

### 401 Unauthorized

| 触发 | 真凶 | 修法 |
|------|------|------|
| Windows desktop 连 WSL 9119 报 401 | main.cjs `X-Hermes-Session-Token` 不被 WSL dashboard 认 | 改用 loopback 模式（`--host 127.0.0.1`）|
| `curl /api/sessions` 报 401（basic auth 模式）| 没带 cookie | 先 POST `/auth/password-login` 拿 cookie |

### 404 Not Found

| 触发 | 真凶 | 修法 |
|------|------|------|
| `npm audit` 报 `/security/advisories/bulk` | npmmirror 没这个 endpoint | `.npmrc` 加 `audit=false` |
| desktop 报 `404 /api/media` | WSL dashboard 167 endpoints 不含 `/api/media` | **忽略**，不影响 chat/session/memory |
| `gh-proxy.com` 报 404 | URL 格式错 | 必须用 `https://gh-proxy.com/https://github.com/...`（双 https）|

### 403 Forbidden

| 触发 | 真凶 | 修法 |
|------|------|------|
| `curl https://pypi.tuna.tsinghua.edu.cn/simple/...` 报 403 | 缺 `trusted-host` | `/etc/pip.conf` 加 `trusted-host = pypi.tuna.tsinghua.edu.cn` |

### EBUSY

| 触发 | 真凶 | 修法 |
|------|------|------|
| `npm install electron` 报 EBUSY | electron.exe 还在跑 | 杀进程 + 删 `node_modules\electron` + 重试 |

### Exit handler never called!

| 触发 | 真凶 | 修法 |
|------|------|------|
| `npm install` 报 Exit handler | npm 11.11.0 自身 bug | 不用 `npm install electron`，手动装 |

### Address already in use

| 触发 | 真凶 | 修法 |
|------|------|------|
| `hermes dashboard` 报端口占用 | 旧 dashboard 进程没杀干净 | `pkill -f "dashboard.*9119"` + 等 2s + 重启 |
| `hermes.exe desktop` 报 9119 占用 | 同上 | 同上 |

## 二、按场景分类

### 场景 1：装完后 `hermes doctor` 报某项 FAIL

| FAIL 项 | 修法 |
|---------|------|
| Security | `chmod 600 ~/.hermes/.env` |
| Python | `sudo apt install python3.12-venv` |
| SSL | `sudo apt install libssl-dev` |
| OpenAI SDK | `pip install --upgrade openai` |
| Packages | `pip install -e ~/.hermes/hermes-agent` |

### 场景 2：WSL 9119 dashboard 启不来

```bash
# 看 log
tail -50 /tmp/dashboard_9119.log
```

| log 关键行 | 真凶 | 修法 |
|----------|------|------|
| `→ Building web UI...` | 正常 30s | 等 30s |
| `✗ Web UI npm install failed` | 公司代理 block npm | 配 `.npmrc`（在 hermes-agent 根目录）|
| `Address already in use` | 旧进程没杀 | `pkill -f dashboard` + 等 2s |
| `ModuleNotFoundError: No module named 'xxx'` | hermes-agent 没装全 | `pip install -e ~/.hermes/hermes-agent` |
| `OSError: [Errno 98] Address already in use` | 端口 9119 被别的程序占 | `lsof -i:9119` 找 + 杀 |

### 场景 3：Windows desktop 弹不出 / 立刻退出

| 症状 | 真凶 | 修法 |
|------|------|------|
| 双击 .lnk 没反应 | electron binary 缺失 | 跑 `templates/fix-electron-wrapper.ps1` |
| 弹窗 5s 后退出 | backend 连不上（9119 没跑）| 跑 Step 1 探针 + Step 2 重启 9119 |
| 弹窗报 "HERMES_DESKTOP_REMOTE_TOKEN not provided" | 3 env vars 没设齐 | 检查 `start-desktop.ps1` 是否在 PowerShell 跑（不是 cmd）|
| 弹窗报 "Couldn't start" | desktop 启动失败 | 看 Windows Event Viewer / `%APPDATA%\hermes\logs\` |
| 5-6 个 electron 进程跑但 GUI 看不见 | GUI 最小化到托盘 | 点系统托盘找 |
| GUI 文字乱码 | 字体问题 | 装 Microsoft YaHei UI 字体 |

### 场景 4：共同一个记忆不生效

| 症状 | 真凶 | 修法 |
|------|------|------|
| Windows 端 `hermes doctor` 走自己 `%LOCALAPPDATA%\hermes` | `HERMES_HOME` 没设 | `start-desktop.ps1` 跑在 PowerShell（不是 cmd）|
| WSL 端看 Windows 端配置 | Windows 配置没同步到 WSL | Windows 端 `hermes` 不写 WSL 端 config（除非同步）|
| 双方 sessions 不一致 | 同一份 `~/.hermes/sessions/` 验证 | `ls -la \\wsl$\Ubuntu\home\lujun\.hermes\sessions\` |

### 场景 5：升级 v0.16.0 → v0.17.0 撞坑

| 症状 | 修法 |
|------|------|
| `hermes --version` 后 `Update available: X commits behind` | 跑 `pip install --upgrade hermes-agent`（**WSL 端**，不要 Windows 端）|
| WSL 端升级后 Windows 端报错 | Windows 端仍是旧版，差异太大——同步 WSL 仓库到 Windows（参考 hermes-update-troubleshooting）|
| 升级后 dashboard 报 API 错 | 重启 dashboard：`pkill -f dashboard` + `hermes dashboard` |

## 三、最常用的 5 条命令

```bash
# 1. 探 WSL 9119
curl -s -o /dev/null -w "%{http_code}\n" -H "X-Hermes-Session-Token: $(cat ~/.hermes/dashboard.token)" http://127.0.0.1:9119/api/status

# 2. 重启 WSL 9119
bash scripts/restart-9119.sh

# 3. 配置共同一个记忆
bash scripts/setup-shared-memory.sh

# 4. 探 Windows Electron 进程
powershell.exe -NoProfile -Command "Get-Process -Name electron | Select Id, MainWindowTitle"

# 5. 看 dashboard log
tail -50 /tmp/dashboard_9119.log
```

## 四、日志位置速查

| 日志 | 路径 | 看什么 |
|------|------|-------|
| WSL dashboard | `/tmp/dashboard_9119.log` | Building web UI / npm install / 启动进度 |
| Windows electron stdout | `%APPDATA%\hermes\logs\desktop-*.log` | electron 启动信息 |
| Windows hermes CLI | `%LOCALAPPDATA%\hermes\logs\hermes-*.log` | hermes.exe 报错 |
| WSL hermes CLI | `~/.hermes/logs/hermes-*.log` | hermes CLI 报错 |

## 五、性能调优

| 慢在哪 | 优化 |
|--------|------|
| `hermes dashboard` 启动 30s | 等 30s（设计如此），不可优化 |
| `hermes doctor` 跑 5s | 正常 |
| Windows desktop 弹窗 8-10s | 手动装 electron 启动慢，pilotdeck 装会快但会撞 npm 11 bug |
| `pip install` 慢 | 已配清华镜像，但 hermes-agent 依赖多，要 30-60s |
| `git clone hermes-agent` 慢 | gh-proxy 镜像 0.5s，应该 5s 内下完 50MB |
