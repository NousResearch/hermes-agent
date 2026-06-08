#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Hermes Agent 离线部署包准备脚本 (Debian/Ubuntu Linux)
# 在有网络的 Linux 机器上运行，准备离线包后传输到无网络的目标机
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

step()   { echo -e "\n${CYAN}==> $*${NC}"; }
sub()    { echo -e "    -> $*"; }
ok()     { echo -e "    ${GREEN}[OK] $*${NC}"; }
warn()   { echo -e "    ${YELLOW}[WARN] $*${NC}"; }
err()    { echo -e "    ${RED}[ERROR] $*${NC}"; }
die()    { err "$*"; exit 1; }

# ── 参数 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$PROJECT_ROOT/hermes-offline-bundle}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
NODE_MAJOR="${NODE_MAJOR:-18}"
SKIP_BROWSER="${SKIP_BROWSER:-false}"
SKIP_NPM="${SKIP_NPM:-false}"
SKIP_DEBS="${SKIP_DEBS:-false}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        --node-major)     NODE_MAJOR="$2"; shift 2 ;;
        --skip-browser)   SKIP_BROWSER=true; shift ;;
        --skip-npm)       SKIP_NPM=true; shift ;;
        --skip-debs)      SKIP_DEBS=true; shift ;;
        -h|--help)
            echo "用法: $0 [选项] [输出目录]"
            echo ""
            echo "选项:"
            echo "  --output-dir DIR      输出目录 (默认: hermes-offline-bundle)"
            echo "  --python-version VER  Python 版本 (默认: 3.11)"
            echo "  --node-major VER      Node.js 主版本 (默认: 22)"
            echo "  --skip-browser        跳过 Playwright Chromium"
            echo "  --skip-npm            跳过 npm 离线缓存"
            echo "  --skip-debs           跳过 deb 系统依赖"
            echo "  -h, --help            显示帮助"
            exit 0
            ;;
        -*) die "未知参数: $1" ;;
        *)  OUTPUT_DIR="$1"; shift ;;
    esac
done

# ── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Hermes Agent Offline Bundle Builder${NC}"
echo -e "${CYAN}  Platform: Debian/Ubuntu x86_64${NC}"
echo -e "${CYAN}  Python: $PYTHON_VERSION | Node: v${NODE_MAJOR}.x${NC}"
echo -e "${CYAN}========================================${NC}"

# ── 前置检查 ─────────────────────────────────────────────────────────────────
step "检查前置条件"

command -v git     &>/dev/null || die "需要 git"
ok "git $(git --version | awk '{print $3}')"

command -v curl    &>/dev/null || die "需要 curl"
ok "curl 已安装"

# uv (仅检查，不安装到本机)
if ! command -v uv &>/dev/null; then
    die "需要 uv。安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"

# Python
PYTHON_BIN=""
PYTHON_MAJOR="${PYTHON_VERSION%%.*}"
PYTHON_MINOR="${PYTHON_VERSION#*.}"
for py in "python${PYTHON_VERSION}" "python3" "python"; do
    if command -v "$py" &>/dev/null; then
        PY_MAJOR=$("$py" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)
        PY_MINOR=$("$py" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)
        if [[ "$PY_MAJOR" -gt "$PYTHON_MAJOR" ]] || \
           { [[ "$PY_MAJOR" -eq "$PYTHON_MAJOR" ]] && [[ "$PY_MINOR" -ge "$PYTHON_MINOR" ]]; }; then
            PYTHON_BIN="$py"
            break
        fi
    fi
done
[[ -z "$PYTHON_BIN" ]] && die "需要 Python >= $PYTHON_VERSION"
ok "$($PYTHON_BIN --version)"

# ── 创建目录 ─────────────────────────────────────────────────────────────────
step "创建目录结构: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"/{binaries,python-wheels,playwright-browsers,npm-offline,deb-packages}
ok "目录已创建"

# ============================================================
# 1/6 下载 Linux 二进制
# ============================================================
step "1/6 下载 Linux 二进制"

# --- Node.js ---
NODE_EXIST=$(find "$OUTPUT_DIR/binaries" -name "node-v*-linux-x64.tar.xz" 2>/dev/null | head -1)
if [[ -n "$NODE_EXIST" ]]; then
    ok "Node.js 已存在: $(basename "$NODE_EXIST")，跳过"
else
    sub "下载 Node.js v${NODE_MAJOR}.x ..."
    NODE_HTML=$(curl -sL --connect-timeout 10 --max-time 30 "https://nodejs.org/dist/latest-v${NODE_MAJOR}.x/" 2>/dev/null || true)
    NODE_VER=$(echo "$NODE_HTML" | grep -oP "node-v(\d+\.\d+\.\d+)-linux-x64\.tar\.xz" | head -1 | grep -oP '\d+\.\d+\.\d+')
    if [[ -n "$NODE_VER" ]]; then
        NODE_FILE="node-v${NODE_VER}-linux-x64.tar.xz"
        NODE_URL="https://nodejs.org/dist/latest-v${NODE_MAJOR}.x/$NODE_FILE"
        if curl -fSL --connect-timeout 15 --max-time 300 --progress-bar -o "$OUTPUT_DIR/binaries/$NODE_FILE" "$NODE_URL"; then
            ok "Node.js v$NODE_VER ($(du -m "$OUTPUT_DIR/binaries/$NODE_FILE" | cut -f1) MB)"
        else
            warn "Node.js 下载失败"
        fi
    else
        warn "无法解析 Node.js 版本，跳过"
    fi
fi

# --- ripgrep ---
RG_FILE="ripgrep-15.1.0-x86_64-unknown-linux-musl.tar.gz"
if [[ -f "$OUTPUT_DIR/binaries/$RG_FILE" ]]; then
    ok "ripgrep 已存在，跳过"
else
    sub "下载 ripgrep ..."
    RG_URL="https://github.com/BurntSushi/ripgrep/releases/latest/download/$RG_FILE"
    if curl -fSL --http1.1 --connect-timeout 15 --max-time 120 --progress-bar -o "$OUTPUT_DIR/binaries/$RG_FILE" "$RG_URL"; then
        ok "ripgrep 15.1.0 ($(du -m "$OUTPUT_DIR/binaries/$RG_FILE" | cut -f1) MB)"
    else
        warn "ripgrep 下载失败，跳过"
    fi
fi

# --- ffmpeg ---
FFMPEG_FILE="ffmpeg-release-amd64-static.tar.xz"
if [[ -f "$OUTPUT_DIR/binaries/$FFMPEG_FILE" ]]; then
    ok "ffmpeg 已存在，跳过"
else
    sub "下载 ffmpeg (静态构建) ..."
    FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    if curl -fSL --connect-timeout 15 --max-time 300 --progress-bar -o "$OUTPUT_DIR/binaries/$FFMPEG_FILE" "$FFMPEG_URL"; then
        ok "ffmpeg ($(du -m "$OUTPUT_DIR/binaries/$FFMPEG_FILE" | cut -f1) MB)"
    else
        warn "ffmpeg 下载失败"
    fi
fi

# --- uv (Linux 二进制) ---
UV_FILE="uv-x86_64-unknown-linux-gnu.tar.gz"
if [[ -f "$OUTPUT_DIR/binaries/$UV_FILE" ]]; then
    ok "uv 已存在，跳过"
else
    sub "下载 uv (Linux 二进制) ..."
    UV_URL="https://github.com/astral-sh/uv/releases/latest/download/$UV_FILE"
    if curl -fSL --http1.1 --connect-timeout 15 --max-time 120 --progress-bar -o "$OUTPUT_DIR/binaries/$UV_FILE" "$UV_URL"; then
        ok "uv ($(du -m "$OUTPUT_DIR/binaries/$UV_FILE" | cut -f1) MB)"
    else
        warn "uv 下载失败"
    fi
fi

# ============================================================
# 2/6 下载 Python wheels
# ============================================================
step "2/6 下载 Python wheels"

WHEEL_DIR="$OUTPUT_DIR/python-wheels"
EXISTING_WHEELS=$(find "$WHEEL_DIR" -name "*.whl" 2>/dev/null | wc -l)

if [[ "$EXISTING_WHEELS" -gt 50 ]]; then
    ok "已有 $EXISTING_WHEELS 个 wheel 包，跳过下载"
else
    # 导出 requirements
    sub "从 uv.lock 导出依赖列表 ..."
    REQ_FILE="$OUTPUT_DIR/requirements.txt"
    cd "$PROJECT_ROOT"
    if uv export --extra all --no-hashes -o "$REQ_FILE" 2>/dev/null; then
        # 移除 -e . (editable install，不能和 --target 一起用)
        grep -v '^\-e \.' "$REQ_FILE" > "$REQ_FILE.tmp" && mv "$REQ_FILE.tmp" "$REQ_FILE"
        PKG_COUNT=$(grep -c '==' "$REQ_FILE" || echo 0)
        ok "导出 $PKG_COUNT 个包"
    else
        warn "uv export 失败，使用 hermes-agent[all]"
        echo "hermes-agent[all]" > "$REQ_FILE"
    fi

    # 下载 [all] extras wheels (本机平台 = Linux)
    sub "下载 hermes-agent[all] wheels ..."
    uv pip install \
        --python-version "$PYTHON_VERSION" \
        --only-binary :all: \
        --target "$WHEEL_DIR" \
        -r "$REQ_FILE" 2>&1 | tail -5 || {
        warn "uv pip install --target 失败，重试不限制二进制 ..."
        uv pip install \
            --python-version "$PYTHON_VERSION" \
            --target "$WHEEL_DIR" \
            -r "$REQ_FILE" 2>&1 | tail -5 || true
    }

    # Lazy deps
    sub "下载 lazy deps ..."
    LAZY_REQ="$OUTPUT_DIR/requirements-lazy.txt"
    cat > "$LAZY_REQ" << 'EOF'
anthropic==0.87.0
boto3==1.42.89
azure-identity==1.25.3
exa-py==2.10.2
firecrawl-py==4.17.0
parallel-web==0.4.2
mistralai==2.4.8
edge-tts==7.2.7
elevenlabs==1.59.0
faster-whisper==1.2.1
sounddevice==0.5.5
numpy==2.4.3
fal-client==0.13.1
honcho-ai==2.0.1
hindsight-client==0.6.1
python-telegram-bot[webhooks]==22.6
discord.py[voice]==2.7.1
brotlicffi==1.2.0.1
slack-bolt==1.27.0
slack-sdk==3.40.1
aiohttp==3.13.4
mautrix[encryption]==0.21.0
Markdown==3.10.2
aiosqlite==0.22.1
asyncpg==0.31.0
aiohttp-socks==0.11.0
dingtalk-stream==0.24.3
alibabacloud-dingtalk==2.2.42
qrcode==7.4.2
lark-oapi==1.5.3
defusedxml==0.7.1
modal==1.3.4
daytona==0.155.0
google-api-python-client==2.194.0
google-auth-oauthlib==1.3.1
google-auth-httplib2==0.3.1
youtube-transcript-api==1.2.4
agent-client-protocol==0.9.0
EOF

uv pip install \
    --python-version "$PYTHON_VERSION" \
    --only-binary :all: \
    --target "$WHEEL_DIR" \
    -r "$LAZY_REQ" 2>&1 | tail -5 || {
    warn "部分 lazy deps 下载失败（可能无 Linux wheel），重试不限制二进制 ..."
    uv pip install \
        --python-version "$PYTHON_VERSION" \
        --target "$WHEEL_DIR" \
        -r "$LAZY_REQ" 2>&1 | tail -5 || true
}

WHEEL_COUNT=$(find "$WHEEL_DIR" -name "*.whl" 2>/dev/null | wc -l)
WHEEL_SIZE=$(du -sm "$WHEEL_DIR" 2>/dev/null | cut -f1)
ok "Python wheels: $WHEEL_COUNT 个, ${WHEEL_SIZE} MB"
fi

# ============================================================
# 3/6 打包源码
# ============================================================
step "3/6 打包源码"

cd "$PROJECT_ROOT"
git archive --format=tar.gz -o "$OUTPUT_DIR/hermes-agent.tar.gz" HEAD
ok "源码打包完成 ($(du -m "$OUTPUT_DIR/hermes-agent.tar.gz" | cut -f1) MB)"

# ============================================================
# 4/6 下载 Playwright Chromium (Linux 原生)
# ============================================================
if [[ "$SKIP_BROWSER" != "true" ]]; then
    if ls "$OUTPUT_DIR/playwright-browsers/chromium-"* &>/dev/null 2>&1; then
        step "4/6 Playwright Chromium 已存在，跳过"
    else
        step "4/6 下载 Playwright Chromium (Linux 原生)"

        export PLAYWRIGHT_BROWSERS_PATH="$OUTPUT_DIR/playwright-browsers"

        # 先安装 node_modules (agent-browser 需要)
        cd "$PROJECT_ROOT"
        sub "安装 npm 依赖 (Playwright 需要) ..."
        npm install --silent 2>/dev/null || true

        sub "下载 Chromium ..."
        npx -y playwright install chromium 2>&1 | tail -3 || {
            warn "Playwright 下载失败"
            unset PLAYWRIGHT_BROWSERS_PATH
        }

        if [[ -d "$OUTPUT_DIR/playwright-browsers/chromium-"* ]]; then
            ok "Playwright Chromium 已下载"
        fi

        unset PLAYWRIGHT_BROWSERS_PATH
    fi
else
    step "4/6 跳过 Playwright Chromium"
fi

# ============================================================
# 5/6 创建 npm 离线缓存
# ============================================================
if [[ "$SKIP_NPM" != "true" ]]; then
    NPM_OFFLINE="$OUTPUT_DIR/npm-offline"
    NPM_EXISTING=$(find "$NPM_OFFLINE" -name "*.tgz" 2>/dev/null | wc -l)
    if [[ "$NPM_EXISTING" -gt 10 ]]; then
        step "5/6 npm 离线缓存已存在 ($NPM_EXISTING 个包)，跳过"
    else
        step "5/6 创建 npm 离线缓存"

        cd "$PROJECT_ROOT"

        # 确保 node_modules 完整
        sub "安装 npm 依赖 ..."
        npm install --silent 2>/dev/null || true

        sub "打包 npm 依赖 ..."
        PACKED=0

        # 从 package.json 收集依赖名
        for pkg_json in package.json web/package.json ui-tui/package.json; do
            if [[ -f "$pkg_json" ]]; then
                # 提取非 file: 的依赖名
                deps=$(node -e "
                    const pkg = require('./$pkg_json');
                    const deps = pkg.dependencies || {};
                    for (const [name, ver] of Object.entries(deps)) {
                        if (!ver.startsWith('file:')) console.log(name);
                    }
                " 2>/dev/null || true)

                while IFS= read -r dep; do
                    [[ -z "$dep" ]] && continue
                    dep_dir="node_modules/$dep"
                    if [[ -d "$dep_dir" ]]; then
                        tarball=$(cd "$dep_dir" && npm pack --silent 2>/dev/null || true)
                        if [[ -n "$tarball" && -f "$dep_dir/$tarball" ]]; then
                            mv "$dep_dir/$tarball" "$NPM_OFFLINE/" 2>/dev/null || true
                            PACKED=$((PACKED + 1))
                        fi
                    fi
                done <<< "$deps"
            fi
        done

        NPM_SIZE=$(du -sm "$NPM_OFFLINE" 2>/dev/null | cut -f1)
        ok "npm 离线包: $PACKED 个, ${NPM_SIZE} MB"
    fi
else
    step "5/6 跳过 npm 离线缓存"
fi

# ============================================================
# 6/6 下载 Playwright 系统依赖 (deb 包)
# ============================================================
if [[ "$SKIP_DEBS" != "true" ]]; then
    DEB_DIR="$OUTPUT_DIR/deb-packages"
    DEB_EXISTING=$(find "$DEB_DIR" -name "*.deb" 2>/dev/null | wc -l)
    if [[ "$DEB_EXISTING" -gt 20 ]]; then
        step "6/6 deb 包已存在 ($DEB_EXISTING 个)，跳过"
    else
        step "6/6 下载 Playwright 系统依赖 (deb 包)"
    )

    sub "下载 ${#DEB_PKGS[@]} 个 deb 包及其依赖 ..."
    cd "$DEB_DIR"

    # 先下载指定包
    apt-get download "${DEB_PKGS[@]}" 2>/dev/null || true

    # 下载传递依赖
    for pkg in "${DEB_PKGS[@]}"; do
        apt-cache depends --recurse --no-recommends --no-suggests \
            --no-conflicts --no-breaks --no-replaces --no-enhances \
            "$pkg" 2>/dev/null | grep '^\w' | sort -u | while read dep; do
                apt-get download "$dep" 2>/dev/null || true
            done
    done

    DEB_COUNT=$(find "$DEB_DIR" -name "*.deb" 2>/dev/null | wc -l)
    DEB_SIZE=$(du -sm "$DEB_DIR" 2>/dev/null | cut -f1)
    ok "deb 包: $DEB_COUNT 个, ${DEB_SIZE} MB"
else
    step "6/6 跳过 deb 包下载"
fi

# ============================================================
# 复制安装脚本 + 生成清单
# ============================================================
step "收尾"

cp "$SCRIPT_DIR/install-offline.sh" "$OUTPUT_DIR/" 2>/dev/null || true
chmod +x "$OUTPUT_DIR/install-offline.sh" 2>/dev/null || true

# 清理临时文件
rm -f "$OUTPUT_DIR/requirements.txt" "$OUTPUT_DIR/requirements-lazy.txt"

# 生成清单
cat > "$OUTPUT_DIR/MANIFEST.txt" << EOF
# Hermes Agent 离线部署包
# 生成时间: $(date '+%Y-%m-%d %H:%M:%S')
# 生成机器: $(hostname) ($(uname -m))
# 目标平台: Linux x86_64 / Python $PYTHON_VERSION

## binaries/
$(ls -lh "$OUTPUT_DIR/binaries/" 2>/dev/null | tail -n +2 | awk '{print "- " $NF " (" $5 ")"}')

## python-wheels/
- $WHEEL_COUNT 个 wheel 包, ${WHEEL_SIZE} MB

## playwright-browsers/
$(ls -d "$OUTPUT_DIR/playwright-browsers/chromium-"* 2>/dev/null | head -1 | xargs -I{} basename {} || echo "- 未下载")

## npm-offline/
- $PACKED 个 npm 包, ${NPM_SIZE:-0} MB

## deb-packages/
- $DEB_COUNT 个 deb 包, ${DEB_SIZE:-0} MB

## 安装方法
传输 hermes-offline-bundle/ 到目标机后:
  chmod +x install-offline.sh
  sudo ./install-offline.sh
EOF

ok "清单已写入 MANIFEST.txt"

# ── 完成 ─────────────────────────────────────────────────────────────────────
TOTAL_SIZE=$(du -sm "$OUTPUT_DIR" | cut -f1)

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  离线包准备完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  输出目录: $OUTPUT_DIR"
echo -e "  总大小:   ${TOTAL_SIZE} MB"
echo ""
echo -e "  下一步:"
echo -e "  1. 传输 hermes-offline-bundle/ 到目标机"
echo -e "  2. 在目标机: chmod +x install-offline.sh"
echo -e "  3. 在目标机: sudo ./install-offline.sh"
echo -e "${GREEN}========================================${NC}"
