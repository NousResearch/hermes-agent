#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Hermes Agent 离线安装脚本
# 在完全无网络的 Linux x86_64 目标机上运行
# 前置条件: Python 3.11+, Git, root 权限
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 颜色输出 ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

step()   { echo -e "\n${CYAN}==> $*${NC}"; }
sub()    { echo -e "    -> $*"; }
ok()     { echo -e "    ${GREEN}[OK] $*${NC}"; }
warn()   { echo -e "    ${YELLOW}[WARN] $*${NC}"; }
err()    { echo -e "    ${RED}[ERROR] $*${NC}"; }
die()    { err "$*"; exit 1; }

# ── 变量 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/hermes-agent"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PYTHON_VERSION="3.11"
VENV_DIR="$INSTALL_DIR/venv"
NODE_INSTALL_DIR="/usr/local"
MIN_DISK_GB=5

# 可选参数
SKIP_BROWSER=false
SKIP_DEPS_BUILD=false

# ── 参数解析 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --install-dir)    INSTALL_DIR="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        --skip-browser)   SKIP_BROWSER=true; shift ;;
        --skip-deps-build) SKIP_DEPS_BUILD=true; shift ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --install-dir DIR      安装目录 (默认: /opt/hermes-agent)"
            echo "  --python-version VER   Python 版本 (默认: 3.11)"
            echo "  --skip-browser         跳过 Playwright Chromium 安装"
            echo "  --skip-deps-build      跳过前端构建"
            echo "  -h, --help             显示帮助"
            exit 0
            ;;
        *) die "未知参数: $1" ;;
    esac
done

VENV_DIR="$INSTALL_DIR/venv"

# ── 前置检查 ─────────────────────────────────────────────────────────────────
step "检查前置条件"

# root 权限
if [[ $EUID -ne 0 ]]; then
    die "需要 root 权限运行此脚本。请使用: sudo $0"
fi
ok "root 权限确认"

# Python - 自动检测 >= 3.11 的 Python
PYTHON_BIN=""
for py in "python${PYTHON_VERSION}" "python3.13" "python3.12" "python3.11" "python3" "python" \
          "/usr/local/bin/python3.13" "/usr/local/bin/python3.12" "/usr/local/bin/python3.11" \
          "/usr/local/bin/python3" "/usr/bin/python3.13" "/usr/bin/python3.12" "/usr/bin/python3.11" "/usr/bin/python3"; do
    if command -v "$py" &>/dev/null || [[ -x "$py" ]]; then
        PY_MAJOR=$("$py" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)
        PY_MINOR=$("$py" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)
        if [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -ge 11 ]] && [[ "$PY_MINOR" -lt 14 ]]; then
            PYTHON_BIN="$py"
            break
        fi
    fi
done
[[ -z "$PYTHON_BIN" ]] && die "需要 Python 3.11-3.13。请先安装 python3.11/3.12/3.13"
PYTHON_FULL_VER=$("$PYTHON_BIN" --version 2>&1)
ok "$PYTHON_FULL_VER ($PYTHON_BIN)"

# Git
command -v git &>/dev/null || die "需要 git。请先安装 git。"
ok "git 已安装"

# 磁盘空间
AVAIL_GB=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | tr -d 'G')
if [[ "$AVAIL_GB" -lt "$MIN_DISK_GB" ]]; then
    die "磁盘空间不足: 可用 ${AVAIL_GB}GB, 需要至少 ${MIN_DISK_GB}GB"
fi
ok "磁盘空间: ${AVAIL_GB}GB 可用"

# 检查离线包完整性
[[ -d "$SCRIPT_DIR/binaries" ]]       || die "缺少 binaries/ 目录"
[[ -d "$SCRIPT_DIR/python-wheels" ]]  || die "缺少 python-wheels/ 目录"
[[ -f "$SCRIPT_DIR/hermes-agent.tar.gz" ]] || die "缺少 hermes-agent.tar.gz"
ok "离线包完整性检查通过"

# ── 1. 安装 Node.js ─────────────────────────────────────────────────────────
step "1/10 安装 Node.js"

NODE_TAR=$(find "$SCRIPT_DIR/binaries" -name "node-v*-linux-x64.tar.xz" | head -1)
if [[ -n "$NODE_TAR" ]]; then
    sub "解压 Node.js: $(basename "$NODE_TAR")"
    tar -xJf "$NODE_TAR" -C "$NODE_INSTALL_DIR" --strip-components=1
    NODE_VER=$(node --version 2>/dev/null || echo "unknown")
    ok "Node.js $NODE_VER 已安装到 $NODE_INSTALL_DIR"
else
    if command -v node &>/dev/null; then
        ok "Node.js 已存在: $(node --version)"
    else
        die "未找到 Node.js 安装包且系统未安装 Node.js"
    fi
fi

# ── 2. 安装 ripgrep ─────────────────────────────────────────────────────────
step "2/10 安装 ripgrep"

RG_TAR=$(find "$SCRIPT_DIR/binaries" -name "ripgrep-*-x86_64-unknown-linux-musl.tar.gz" | head -1)
if [[ -n "$RG_TAR" ]]; then
    sub "解压 ripgrep: $(basename "$RG_TAR")"
    RG_TMP=$(mktemp -d)
    tar -xzf "$RG_TAR" -C "$RG_TMP"
    cp "$RG_TMP"/*/rg /usr/local/bin/
    chmod +x /usr/local/bin/rg
    rm -rf "$RG_TMP"
    ok "ripgrep $(rg --version | head -1) 已安装"
elif command -v rg &>/dev/null; then
    ok "ripgrep 已存在: $(rg --version | head -1)"
else
    warn "未找到 ripgrep 安装包。hermes 将使用 grep 作为回退。"
fi

# ── 3. 安装 ffmpeg ──────────────────────────────────────────────────────────
step "3/10 安装 ffmpeg"

FFMPEG_TAR=$(find "$SCRIPT_DIR/binaries" -name "ffmpeg-*-amd64-static.tar.*" | head -1)
if [[ -n "$FFMPEG_TAR" ]]; then
    sub "解压 ffmpeg: $(basename "$FFMPEG_TAR")"
    FF_TMP=$(mktemp -d)
    tar -xf "$FFMPEG_TAR" -C "$FF_TMP"
    cp "$FF_TMP"/*/ffmpeg "$FF_TMP"/*/ffprobe /usr/local/bin/ 2>/dev/null || true
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe 2>/dev/null || true
    rm -rf "$FF_TMP"
    ok "ffmpeg $(ffmpeg -version 2>&1 | head -1) 已安装"
elif command -v ffmpeg &>/dev/null; then
    ok "ffmpeg 已存在: $(ffmpeg -version 2>&1 | head -1)"
else
    warn "未找到 ffmpeg。TTS 语音功能将受限。"
fi

# ── 4. 安装 uv ──────────────────────────────────────────────────────────────
step "4/10 安装 uv (Python 包管理器)"

UV_TAR=$(find "$SCRIPT_DIR/binaries" -name "uv-*-unknown-linux-gnu.tar.gz" | head -1)
if [[ -n "$UV_TAR" ]]; then
    sub "解压 uv: $(basename "$UV_TAR")"
    UV_TMP=$(mktemp -d)
    tar -xzf "$UV_TAR" -C "$UV_TMP"
    cp "$UV_TMP"/*/uv /usr/local/bin/ 2>/dev/null || cp "$UV_TMP"/uv /usr/local/bin/ 2>/dev/null || true
    cp "$UV_TMP"/*/uvx /usr/local/bin/ 2>/dev/null || cp "$UV_TMP"/uvx /usr/local/bin/ 2>/dev/null || true
    chmod +x /usr/local/bin/uv /usr/local/bin/uvx 2>/dev/null || true
    rm -rf "$UV_TMP"
    ok "uv $(uv --version 2>/dev/null || echo 'installed') 已安装"
elif command -v uv &>/dev/null; then
    ok "uv 已存在: $(uv --version)"
else
    die "未找到 uv 安装包且系统未安装 uv"
fi

# ── 5. 安装 Playwright 系统依赖 ─────────────────────────────────────────────
step "5/10 安装 Playwright 系统依赖"

DEB_DIR="$SCRIPT_DIR/deb-packages"
if [[ -d "$DEB_DIR" ]] && ls "$DEB_DIR"/*.deb &>/dev/null 2>&1; then
    DEB_COUNT=$(ls "$DEB_DIR"/*.deb 2>/dev/null | wc -l)
    sub "安装 $DEB_COUNT 个 deb 包 ..."
    dpkg -i "$DEB_DIR"/*.deb 2>&1 | tail -5 || true
    # 修复依赖关系
    apt-get install -f -y 2>/dev/null || true
    ok "Playwright 系统依赖已安装"
else
    warn "未找到 deb 包目录。Playwright Chromium 可能无法运行。"
    warn "如需手动安装: apt-get install -y libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libgbm1 libpango-1.0-0 libcairo2 libasound2"
fi

# ── 6. 解压源码 ─────────────────────────────────────────────────────────────
step "6/10 解压 Hermes 源码"

if [[ -d "$INSTALL_DIR" ]]; then
    sub "备份旧安装 ..."
    BACKUP="${INSTALL_DIR}.bak.$(date +%Y%m%d%H%M%S)"
    mv "$INSTALL_DIR" "$BACKUP"
    sub "旧安装已备份到: $BACKUP"
fi

mkdir -p "$INSTALL_DIR"
tar -xzf "$SCRIPT_DIR/hermes-agent.tar.gz" -C "$INSTALL_DIR"
ok "源码已解压到 $INSTALL_DIR"

# ── 7. 创建 Python venv ─────────────────────────────────────────────────────
step "7/10 创建 Python 虚拟环境"

cd "$INSTALL_DIR"

# 禁止 uv 自动下载 Python，使用系统已安装的版本
export UV_PYTHON_DOWNLOADS=never

# 找到系统 Python 完整路径
SYSTEM_PYTHON=$(command -v "$PYTHON_BIN")
sub "使用系统 Python: $SYSTEM_PYTHON"

uv venv venv --python "$SYSTEM_PYTHON"
export VIRTUAL_ENV="$INSTALL_DIR"
export UV_PYTHON="$VENV_DIR/bin/python"
ok "虚拟环境已创建: $VENV_DIR"

# ── 8. 安装 Python 依赖 ─────────────────────────────────────────────────────
step "8/10 安装 Python 依赖 (离线模式)"

# 确保 uv 不会尝试联网下载任何东西
export UV_PYTHON_DOWNLOADS=never
export UV_NO_SYNC=1

WHEEL_DIR="$SCRIPT_DIR/python-wheels"
SITE_PACKAGES="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages"

if [[ ! -d "$WHEEL_DIR" ]]; then
    die "python-wheels/ 目录不存在"
fi

WHEEL_COUNT=$(find "$WHEEL_DIR" -name "*.whl" 2>/dev/null | wc -l)
EXTRACTED_COUNT=$(find "$WHEEL_DIR" -name "*.py" -maxdepth 2 2>/dev/null | wc -l)

if [[ "$WHEEL_COUNT" -gt 0 ]]; then
    # 方式1: .whl 文件，用 --find-links 安装
    sub "从 $WHEEL_COUNT 个 wheel 包安装 ..."
    uv pip install --no-index --find-links "$WHEEL_DIR" -e ".[all]" 2>&1 | tail -3
elif [[ "$EXTRACTED_COUNT" -gt 0 ]]; then
    # 方式2: 已解压的包，直接复制到 site-packages
    sub "从已解压的包安装 ($EXTRACTED_COUNT 个 .py 文件) ..."
    cp -r "$WHEEL_DIR"/* "$SITE_PACKAGES/" 2>/dev/null || true
    sub "已复制到 $SITE_PACKAGES"
else
    die "python-wheels/ 目录为空"
fi

# 安装 lazy deps（所有可选后端）
sub "安装 lazy deps（可选后端） ..."

LAZY_PACKAGES=(
    # Inference providers
    "anthropic==0.87.0"
    "boto3==1.42.89"
    "azure-identity==1.25.3"
    # Web search
    "exa-py==2.10.2"
    "firecrawl-py==4.17.0"
    "parallel-web==0.4.2"
    # TTS
    "mistralai==2.4.8"
    "edge-tts==7.2.7"
    "elevenlabs==1.59.0"
    # STT
    "faster-whisper==1.2.1"
    "sounddevice==0.5.5"
    "numpy==2.4.3"
    # Image
    "fal-client==0.13.1"
    # Memory
    "honcho-ai==2.0.1"
    "hindsight-client==0.6.1"
    # Messaging
    "python-telegram-bot[webhooks]==22.6"
    "discord.py[voice]==2.7.1"
    "brotlicffi==1.2.0.1"
    "slack-bolt==1.27.0"
    "slack-sdk==3.40.1"
    "aiohttp==3.13.4"
    "mautrix[encryption]==0.21.0"
    "Markdown==3.10.2"
    "aiosqlite==0.22.1"
    "asyncpg==0.31.0"
    "aiohttp-socks==0.11.0"
    "dingtalk-stream==0.24.3"
    "alibabacloud-dingtalk==2.2.42"
    "qrcode==7.4.2"
    "lark-oapi==1.5.3"
    "defusedxml==0.7.1"
    # Terminal backends
    "modal==1.3.4"
    "daytona==0.155.0"
    # Skills
    "google-api-python-client==2.194.0"
    "google-auth-oauthlib==1.3.1"
    "google-auth-httplib2==0.3.1"
    "youtube-transcript-api==1.2.4"
    # Tools
    "agent-client-protocol==0.9.0"
)

# 分批安装（仅在使用 .whl 文件时需要，已解压的包已在上面复制完成）
if [[ "$WHEEL_COUNT" -gt 0 ]]; then
    BATCH_SIZE=10
    for ((i = 0; i < ${#LAZY_PACKAGES[@]}; i += BATCH_SIZE)); do
        BATCH=("${LAZY_PACKAGES[@]:i:BATCH_SIZE}")
        sub "安装批次 $((i / BATCH_SIZE + 1)): ${BATCH[*]} ..."
        uv pip install --no-index --find-links "$WHEEL_DIR" "${BATCH[@]}" 2>&1 | tail -1 || true
    done
else
    sub "已解压的包已复制，跳过 lazy deps 单独安装"
fi

# 统计已安装包数
INSTALLED_COUNT=$(uv pip list --format=columns 2>/dev/null | tail -n +3 | wc -l || echo "?")
ok "Python 依赖安装完成 ($INSTALLED_COUNT 个包)"

# ── 9. 安装 Node 依赖 + 构建前端 ───────────────────────────────────────────
step "9/10 安装 Node 依赖"

NPM_OFFLINE_DIR="$SCRIPT_DIR/npm-offline"
cd "$INSTALL_DIR"

if [[ -d "$NPM_OFFLINE_DIR" ]] && ls "$NPM_OFFLINE_DIR"/*.tgz &>/dev/null 2>&1; then
    NPM_COUNT=$(ls "$NPM_OFFLINE_DIR"/*.tgz 2>/dev/null | wc -l)
    sub "从 $NPM_COUNT 个离线包安装 ..."

    # 先安装 workspace 内部依赖（file: 引用）
    npm install --prefer-offline --ignore-scripts 2>&1 | tail -3

    # 从离线 tarball 安装外部依赖
    for tgz in "$NPM_OFFLINE_DIR"/*.tgz; do
        npm install --no-save "$tgz" 2>/dev/null || true
    done

    ok "npm 依赖已安装"
else
    warn "未找到 npm-offline/ 目录。尝试直接 npm install ..."
    npm install --ignore-scripts 2>&1 | tail -3 || warn "npm install 失败"
fi

# 构建前端
if [[ "$SKIP_DEPS_BUILD" != "true" ]]; then
    step "构建前端"

    if [[ -d "$INSTALL_DIR/web" ]]; then
        sub "构建 web dashboard ..."
        cd "$INSTALL_DIR/web"
        npm run build 2>&1 | tail -3 || warn "web 构建失败（非致命）"
        cd "$INSTALL_DIR"
    fi

    if [[ -d "$INSTALL_DIR/ui-tui" ]]; then
        sub "构建 TUI ..."
        cd "$INSTALL_DIR/ui-tui"
        npm install --prefer-offline --ignore-scripts 2>&1 | tail -3 || true
        npm run build 2>&1 | tail -3 || warn "TUI 构建失败（非致命）"
        cd "$INSTALL_DIR"
    fi

    ok "前端构建完成"
fi

# ── 10. 安装 Playwright Chromium ────────────────────────────────────────────
if [[ "$SKIP_BROWSER" != "true" ]]; then
    step "10/10 安装 Playwright Chromium"

    PW_DIR="$SCRIPT_DIR/playwright-browsers"
    PW_CACHE_DIR="${HOME}/.cache/ms-playwright"

    if [[ -d "$PW_DIR" ]] && ls "$PW_DIR"/chromium-* &>/dev/null 2>&1; then
        mkdir -p "$PW_CACHE_DIR"
        cp -r "$PW_DIR"/chromium-* "$PW_CACHE_DIR/"
        ok "Playwright Chromium 已安装到 $PW_CACHE_DIR"
    else
        sub "未找到离线 Chromium 包"
        sub "如有网络可用: npx playwright install chromium"
        sub "或使用系统浏览器: 设置 AGENT_BROWSER_EXECUTABLE_PATH 环境变量"
    fi
else
    step "10/10 跳过 Playwright Chromium"
fi

# ── 创建命令入口 ─────────────────────────────────────────────────────────────
step "创建 hermes 命令入口"

cat > /usr/local/bin/hermes << 'HERMES_SHIM'
#!/usr/bin/env bash
HERMES_DIR="/opt/hermes-agent"
VENV_DIR="$HERMES_DIR/venv"

if [[ ! -f "$VENV_DIR/bin/python" ]]; then
    echo "错误: Hermes 未正确安装。缺少 $VENV_DIR/bin/python" >&2
    exit 1
fi

export VIRTUAL_ENV="$HERMES_DIR"
export PATH="$VENV_DIR/bin:$PATH"
export UV_PYTHON="$VENV_DIR/bin/python"

cd "$HERMES_DIR"
exec "$VENV_DIR/bin/python" -m hermes_cli.main "$@"
HERMES_SHIM

chmod +x /usr/local/bin/hermes
ok "hermes 命令已创建: /usr/local/bin/hermes"

# ── 初始化配置 ───────────────────────────────────────────────────────────────
step "初始化用户配置"

mkdir -p "$HERMES_HOME"/{skills,plugins,skins,logs,cron}

if [[ ! -f "$HERMES_HOME/.env" ]]; then
    if [[ -f "$INSTALL_DIR/.env.example" ]]; then
        cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
        ok ".env 已复制到 $HERMES_HOME/.env"
    else
        touch "$HERMES_HOME/.env"
        ok ".env 已创建（空文件）"
    fi
else
    ok ".env 已存在，跳过"
fi

if [[ ! -f "$HERMES_HOME/config.yaml" ]]; then
    if [[ -f "$INSTALL_DIR/config.yaml.example" ]]; then
        cp "$INSTALL_DIR/config.yaml.example" "$HERMES_HOME/config.yaml"
        ok "config.yaml 已复制"
    fi
fi

# ── 完成 ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Hermes Agent 离线安装完成！                      ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  安装目录: $INSTALL_DIR${NC}"
echo -e "${GREEN}║  配置目录: $HERMES_HOME${NC}"
echo -e "${GREEN}║  命令入口: /usr/local/bin/hermes                             ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  下一步:                                                     ║${NC}"
echo -e "${GREEN}║  1. 编辑 $HERMES_HOME/.env                                  ║${NC}"
echo -e "${GREEN}║     添加你的 API key (OPENAI_API_KEY 等)                     ║${NC}"
echo -e "${GREEN}║  2. 运行: hermes --help                                      ║${NC}"
echo -e "${GREEN}║  3. 启动: hermes                                              ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
