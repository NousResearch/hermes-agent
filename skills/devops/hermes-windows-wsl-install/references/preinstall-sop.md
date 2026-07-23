# 10 步安装 SOP（完整命令）

> 适用：hermes-agent v0.16.0+, Windows 10 22H2+ / Windows 11, WSL 2 Ubuntu 24.04
> 网络：中国大陆 / 公司代理

## Step 1: 装 WSL Ubuntu 24.04

**Windows PowerShell（管理员）**：
```powershell
wsl --install -d Ubuntu-24.04
# 重启后自动弹 Ubuntu 终端，设用户名密码
```

**验证**：
```powershell
wsl -l -v
#  NAME            STATE           VERSION
#  Ubuntu-24.04    Running         2
```

## Step 2: WSL 端装系统包

**WSL Ubuntu 24.04**：
```bash
# 1. 配 apt 阿里镜像
sudo sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources
sudo apt update

# 2. 装 5 类包
sudo apt install -y build-essential python3.12 python3.12-venv python3-dev \
                    libssl-dev libffi-dev git curl
```

**验证**：
```bash
python3.12 --version  # Python 3.12.x
git --version         # git version 2.x
curl --version        # curl 7.x
```

## Step 3: WSL 端配 pip 镜像

```bash
# 配清华镜像
sudo tee /etc/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60
EOF

# 验证
pip config list
# global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'
```

## Step 4: WSL 端 git clone hermes-agent + venv + pip install

```bash
# 1. 走 gh-proxy 镜像（GFW 绕开）
git clone https://gh-proxy.com/https://github.com/NousResearch/hermes-agent.git \
  ~/.hermes/hermes-agent

# 2. 创建 venv
python3.12 -m venv ~/.hermes/hermes-agent/venv
source ~/.hermes/hermes-agent/venv/bin/activate

# 3. 升级 pip + 装 hermes-agent
pip install --upgrade pip
pip install -e ~/.hermes/hermes-agent

# 完后 hermes 命令在 ~/.local/bin/hermes（pip 安装的 entry point）
```

**验证**：
```bash
which hermes
# /home/lujun/.local/bin/hermes
hermes --version
# Hermes Agent v0.16.0 (2026.6.5) · upstream c6b0eb4d
```

## Step 5: WSL 端配 .env

```bash
# 1. 创建 .env
cat > ~/.hermes/.env <<'EOF'
# === Hermes Agent .env (WSL 端 2026-06-16) ===

# 模型 providers
SN_API_KEY=sk-...
SN_BASE_URL=https://token.sensenova.cn/v1
AGNES_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...

# 微信公众号
WECHAT_APPID=wx...
WECHAT_SECRET=...

# Gateway 默认端口
HERMES_GATEWAY_PORT=8888
HERMES_DASHBOARD_PORT=9119
EOF

# 2. 权限 600
chmod 600 ~/.hermes/.env

# 3. 验证
ls -la ~/.hermes/.env
# -rw------- 1 lujun lujun ... .env
```

## Step 6: WSL 端验 hermes doctor

```bash
hermes doctor
# 期望输出（关键项）：
# ✓ Security  : API keys safely stored
# ✓ Python    : 3.12.3
# ✓ SSL       : certifi / openssl
# ✓ OpenAI SDK: 2.24.0
# ✓ Packages  : all required packages installed
```

**如果某项 FAIL**：
- `Python` FAIL → 装 `python3.12-venv`
- `SSL` FAIL → 装 `libssl-dev`
- `Packages` FAIL → 跑 `pip install -e ~/.hermes/hermes-agent` 重试

## Step 7: WSL 端启 9119 dashboard

```bash
# 1. 写固定 token
echo "hermes-shared-2026-06-16" > ~/.hermes/dashboard.token
chmod 600 ~/.hermes/dashboard.token

# 2. 杀旧 9119
old=$(ps aux | grep "dashboard.*9119" | grep -v grep | awk '{print $2}')
[ -n "$old" ] && kill -9 $old
sleep 2

# 3. 后台启 dashboard（loopback 模式 + 固定 token）
HERMES_DASHBOARD_SESSION_TOKEN=$(cat ~/.hermes/dashboard.token) \
  /home/lujun/.local/bin/hermes dashboard --no-open --host 127.0.0.1 --port 9119 \
  > /tmp/dashboard_9119.log 2>&1 &
disown

# 4. 等 30s（web UI build）
sleep 30

# 5. 探针
curl -s -H "X-Hermes-Session-Token: $(cat ~/.hermes/dashboard.token)" \
  http://127.0.0.1:9119/api/status
# 期望 HTTP=200
```

**完整脚本**：见 `scripts/setup-shared-memory.sh`

## Step 8: Windows 装 Node.js 22+

**浏览器下载**（清华镜像）：
```powershell
# 清华镜像
Invoke-WebRequest -Uri "https://mirrors.tuna.tsinghua.edu.cn/nodejs-release/v22.10.0/node-v22.10.0-x64.msi" `
  -OutFile "$env:TEMP\node-v22.10.0-x64.msi"

# 双击安装（勾 "Add to PATH"）
Start-Process "$env:TEMP\node-v22.10.0-x64.msi"
```

**或官网**：
- https://nodejs.org/dist/v22.10.0/node-v22.10.0-x64.msi

**验证**：
```powershell
node --version  # v22.10.0
npm --version   # 10.x.x（Node.js 22 自带 npm 10.x，不会是 11.11.0）
```

## Step 9: Windows 端手动装 Electron 40.9.3

**完整步骤**：见 `manual-electron-install.md`

**摘要**：
```powershell
# 1. 下载 electron binary zip（npmmirror 镜像）
$ver = "40.9.3"
$url = "https://npmmirror.com/mirrors/electron/v$ver/electron-v$ver-win32-x64.zip"
$zip = "$env:TEMP\electron-v$ver.zip"
Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing -TimeoutSec 600
# 138MB / 30-60s

# 2. 解压
$dist = "C:\Users\lujun\AppData\Local\hermes\hermes-agent\node_modules\electron\dist"
if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
New-Item -ItemType Directory -Path $dist -Force | Out-Null
Expand-Archive -Path $zip -DestinationPath $dist -Force
# 213MB electron.exe

# 3. 写 5 个 wrapper 文件
"v$ver" | Out-File "C:\Users\lujun\AppData\Local\hermes\hermes-agent\node_modules\electron\path.txt" -Encoding ASCII
"v$ver" | Out-File "C:\Users\lujun\AppData\Local\hermes\hermes-agent\node_modules\electron\version.txt" -Encoding ASCII

# 完整 wrapper 5 文件 + .npmrc
# 见 templates/fix-electron-wrapper.ps1
```

**验证**：
```powershell
Test-Path "C:\Users\lujun\AppData\Local\hermes\hermes-agent\node_modules\electron\dist\electron.exe"
# True
```

## Step 10: Windows 端配 3 env vars + 启 desktop

**完整启动器**：见 `templates/start-desktop.ps1`

**手动**：
```powershell
$env:HERMES_HOME = '\\wsl$\Ubuntu\home\lujun\.hermes'
$env:HERMES_DESKTOP_REMOTE_URL = 'http://127.0.0.1:9119'
$env:HERMES_DESKTOP_REMOTE_TOKEN = (Get-Content '\\wsl$\Ubuntu\home\lujun\.hermes\dashboard.token' -Raw).Trim()
Set-Location 'C:\Users\lujun\AppData\Local\hermes\hermes-agent\apps\desktop'
& 'C:\Program Files\nodejs\node.exe' 'C:\Users\lujun\AppData\Local\hermes\hermes-agent\node_modules\electron\cli.js' '.'
```

**验证**：
```powershell
Start-Sleep -Seconds 8
Get-Process -Name electron | Select Id, MainWindowTitle
# 期望 5-6 个 electron 进程 + MainWindowTitle="Hermes"
```

## 安装完成检查清单

- [ ] `hermes doctor` 5 项全 ✓
- [ ] WSL 9119 dashboard HTTP=200
- [ ] Windows `node --version` 输出 v22.x
- [ ] `electron.exe` 存在（213MB）
- [ ] 双击 Hermes Desktop.lnk 弹 GUI
- [ ] 5-6 个 electron 进程 + MainWindowTitle="Hermes"
- [ ] 同一份 `~/.hermes/config.yaml`（WSL + Windows 都看得到）
- [ ] 同一份 `~/.hermes/sessions/`（WSL + Windows 都看得到）

## 工具脚本

- `scripts/preinstall-check.sh` — Step 0 环境一键检查（7 探针）
- `scripts/setup-shared-memory.sh` — Step 7 共同一个记忆配置（一键脚本）
- `scripts/restart-9119.sh` — 9119 dashboard 重启
- `templates/start-desktop.ps1` — Step 10 Windows 端启动器
