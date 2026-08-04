#!/bin/bash
#
# install_memory_tencentdb.sh
#
# 在 install_hermes_ubuntu.sh 之后执行，用于：
#   1. 通过 npm 下载 @tencentdb-agent-memory/memory-tencentdb@latest 到
#      $MEMORY_TENCENTDB_ROOT/tdai-memory-openclaw-plugin（默认 ~/.memory-tencentdb/tdai-memory-openclaw-plugin）
#   2. 安装 Gateway 的 Node.js 依赖（npm install）
#   3. 配置 hermes config.yaml 使用 memory_tencentdb 记忆提供者
#   4. 设置 Gateway 自动启动环境变量
#
# 路径约定（全部位于 ~/.memory-tencentdb/ 之下，可通过环境变量覆盖）：
#   $MEMORY_TENCENTDB_ROOT     默认 ~/.memory-tencentdb
#   $TDAI_INSTALL_DIR          默认 $MEMORY_TENCENTDB_ROOT/tdai-memory-openclaw-plugin
#   $TDAI_DATA_DIR             默认 $MEMORY_TENCENTDB_ROOT/memory-tdai
#
# 旧版本（<= 0.3.x）使用 ~/tdai-memory-openclaw-plugin 与 ~/memory-tdai；
# 本脚本会在执行前自动迁移这两个旧目录到新位置（见 Step 0）。
#
# 使用方式：
#   以目标用户身份执行（推荐）：
#     su - <username> -c "bash ~/install_memory_tencentdb.sh"
#     # 或直接以该用户登录后执行
#     bash ~/install_memory_tencentdb.sh
#
#   以 root 身份执行（镜像构建场景）：
#     bash ~/install_memory_tencentdb.sh
#     # root 会自动 su 切换到目标用户执行，完成后修复权限
#
# 前置条件：
#   - install_hermes_ubuntu.sh 已执行完成（hermes-agent 已安装）
#   - Node.js >= 22 已安装

set -e

# 动态获取目标安装用户及其 HOME 目录。
# 优先级：
#   1. 显式 ``INSTALL_AS_USER`` 环境变量（管理员脚本场景：root 跑安装但
#      想为另一个用户配置）
#   2. ``SUDO_USER``（被 ``sudo`` 调用时，切回原用户而不是 root）
#   3. ``whoami`` —— 当前 EUID 对应的用户
#
# 注意：当 root 直接 ssh 登录跑（非 sudo）时，前两个都不会被设置，
# ``whoami`` 返回 ``root``。下面的 ``id -u`` == 0 分支会识别这种"目标
# 就是 root"的情况、跳过 ``su - root`` 递归。
USERNAME="${INSTALL_AS_USER:-${SUDO_USER:-$(whoami)}}"
USER_HOME=$(eval echo ~$USERNAME)

# npm 包名
NPM_PACKAGE="@tencentdb-agent-memory/memory-tencentdb@latest"

# Hermes 路径
HERMES_HOME="$USER_HOME/.hermes"
# HERMES_AGENT_DIR（fix: issue #18）
# 用户通过环境变量传什么就用什么；未设置时 fallback 到传统路径。
# 如果目录不存在，后续前置检查会统一报错。
HERMES_AGENT_DIR="${HERMES_AGENT_DIR:-$HERMES_HOME/hermes-agent}"
HERMES_CONFIG="$HERMES_HOME/config.yaml"

# memory-tencentdb 统一根目录（所有 tdai 相关数据/代码都收纳在此）
# 可通过环境变量 MEMORY_TENCENTDB_ROOT 覆盖
MEMORY_TENCENTDB_ROOT="${MEMORY_TENCENTDB_ROOT:-$USER_HOME/.memory-tencentdb}"

# tdai 解压目标目录（位于统一根目录下）
# 可通过环境变量 TDAI_INSTALL_DIR 覆盖
TDAI_INSTALL_DIR="${TDAI_INSTALL_DIR:-$MEMORY_TENCENTDB_ROOT/tdai-memory-openclaw-plugin}"

# tdai 数据目录（Gateway baseDir，位于统一根目录下）
# 可通过环境变量 TDAI_DATA_DIR 覆盖
TDAI_DATA_DIR="${TDAI_DATA_DIR:-$MEMORY_TENCENTDB_ROOT/memory-tdai}"

# 旧路径（仅用于自动迁移）
LEGACY_INSTALL_DIR="$USER_HOME/tdai-memory-openclaw-plugin"
LEGACY_DATA_DIR="$USER_HOME/memory-tdai"

# ==================== root → 自动切换到目标用户 ====================
# 与 install_hermes_ubuntu.sh 保持一致：如果以 root 执行且目标用户不是
# root，自动 su 切到目标用户运行实际安装逻辑。
#
# 如果当前是 root 且目标用户也是 root（``USERNAME=root``，例如直接 ssh
# 登录 root 跑安装），跳过 ``su - root`` —— 否则会无限递归（``su - root``
# 进入的仍是 root，又走到这个分支，再次 su，永远停不下来）。见 issue #20。

if [ "$(id -u)" -eq 0 ] && [ "$USERNAME" != "root" ]; then
    echo "[memory-tencentdb] Running as root, switching to $USERNAME for installation..."

    # 验证前置条件
    if [ ! -d "$HERMES_AGENT_DIR" ]; then
        echo "[ERROR] Hermes agent not found at $HERMES_AGENT_DIR"
        echo "[ERROR] Please run install_hermes_ubuntu.sh first."
        exit 1
    fi

    # 切换到目标用户执行
    TEMP_SCRIPT=$(mktemp /tmp/memory-tencentdb-install-XXXXXX.sh)
    cp "${BASH_SOURCE[0]}" "$TEMP_SCRIPT"
    chmod 755 "$TEMP_SCRIPT"
    su - $USERNAME -c "bash $TEMP_SCRIPT" </dev/null

    # 修复权限
    echo "[memory-tencentdb] Fixing permissions..."
    chown -R $USERNAME:$USERNAME "$USER_HOME"

    rm -f "$TEMP_SCRIPT"
    echo "[memory-tencentdb] Installation completed successfully"
    exit 0
elif [ "$(id -u)" -eq 0 ]; then
    # 当前是 root 且目标用户也是 root：直接以 root 跑后续安装逻辑，
    # 不再走 ``su -`` 切换（避免 #20 的递归）。
    echo "[memory-tencentdb] Running as root; target user is also root — installing in place."
fi

# ==================== 用户阶段（核心安装逻辑） ====================

echo "[memory-tencentdb] Installing memory-tencentdb plugin (user: $(whoami))..."

# 验证前置条件
if [ ! -d "$HERMES_AGENT_DIR" ]; then
    echo "[ERROR] Hermes agent not found at $HERMES_AGENT_DIR"
    echo "[ERROR] Please run install_hermes_ubuntu.sh first."
    exit 1
fi

# 加载 hermes 环境（PATH 中需要 node/npm）
if [ -f /etc/profile.d/hermes-env.sh ]; then
    source /etc/profile.d/hermes-env.sh
fi

# 确保统一根目录存在
mkdir -p "$MEMORY_TENCENTDB_ROOT"

# ---------- Step 0: 自动迁移旧路径（向后兼容） ----------
#
# 历史版本把 tdai 解压到 ~/tdai-memory-openclaw-plugin、数据放在 ~/memory-tdai。
# 现在统一收纳到 ~/.memory-tencentdb/ 之下，这里做一次性自动迁移。
# 已经在新位置时跳过；新旧都存在时打印警告并保留新位置不动。

migrate_legacy_dir() {
    local legacy="$1"
    local target="$2"
    local label="$3"
    if [ ! -e "$legacy" ]; then
        return 0
    fi
    if [ -L "$legacy" ]; then
        # 旧位置是 symlink，直接清掉
        echo "[memory-tencentdb] Removing legacy symlink: $legacy"
        rm -f "$legacy"
        return 0
    fi
    if [ -e "$target" ]; then
        echo "[memory-tencentdb] WARN: legacy $label dir exists at $legacy but new location $target also exists." >&2
        echo "[memory-tencentdb] WARN: keeping new location; please review and remove $legacy manually if obsolete." >&2
        return 0
    fi
    echo "[memory-tencentdb] Migrating legacy $label dir: $legacy -> $target"
    mkdir -p "$(dirname "$target")"
    mv "$legacy" "$target"
}

migrate_legacy_dir "$LEGACY_INSTALL_DIR" "$TDAI_INSTALL_DIR" "install"
migrate_legacy_dir "$LEGACY_DATA_DIR"    "$TDAI_DATA_DIR"    "data"

# ---------- Step 1: 通过 npm 下载包并提取到 $TDAI_INSTALL_DIR ----------

echo "[memory-tencentdb] Step 1: Downloading $NPM_PACKAGE via npm..."

# 清理旧安装
rm -rf "$TDAI_INSTALL_DIR"

# 使用临时目录通过 npm install 下载包
TEMP_DOWNLOAD=$(mktemp -d /tmp/memory-tencentdb-download-XXXXXX)
cd "$TEMP_DOWNLOAD"
npm init -y --silent > /dev/null 2>&1
npm install "$NPM_PACKAGE" --omit=dev 2>&1 | tail -5

# 包安装后位于 node_modules/@tencentdb-agent-memory/memory-tencentdb
PACK_DIR="$TEMP_DOWNLOAD/node_modules/@tencentdb-agent-memory/memory-tencentdb"

if [ ! -d "$PACK_DIR" ]; then
    echo "[ERROR] Downloaded package directory not found at $PACK_DIR"
    rm -rf "$TEMP_DOWNLOAD"
    exit 1
fi

# 将包内容移动到目标安装目录
mkdir -p "$(dirname "$TDAI_INSTALL_DIR")"
cp -r "$PACK_DIR" "$TDAI_INSTALL_DIR"

echo "[memory-tencentdb] Package downloaded and extracted to $TDAI_INSTALL_DIR"

# ---------- Step 2: 安装 Gateway Node.js 依赖 ----------

echo "[memory-tencentdb] Step 2: Installing Gateway dependencies..."

cd "$TDAI_INSTALL_DIR"

echo "[memory-tencentdb] Running npm install (this may take a while)..."
npm install --omit=dev 2>&1 | tail -5

# 安装 tsx（Gateway 启动需要），优先本地安装
if ! npx tsx --version &>/dev/null; then
    npm install tsx 2>&1 | tail -2
fi

echo "[memory-tencentdb] Gateway dependencies installed"

# ---------- Step 2.5: 将插件链接到 hermes 插件目录 ----------

echo "[memory-tencentdb] Step 2.5: Linking plugin into hermes plugins directory..."

HERMES_PLUGIN_DIR="$HERMES_AGENT_DIR/plugins/memory/memory_tencentdb"
PLUGIN_SRC_DIR="$TDAI_INSTALL_DIR/hermes-plugin/memory/memory_tencentdb"

# 移除旧链接/目录
rm -rf "$HERMES_PLUGIN_DIR"

# 创建 symlink 使 hermes 能发现插件
ln -sf "$PLUGIN_SRC_DIR" "$HERMES_PLUGIN_DIR"

echo "[memory-tencentdb] Plugin linked: $HERMES_PLUGIN_DIR -> $PLUGIN_SRC_DIR"

# ---------- Step 3: 提示用户手动开启 memory_tencentdb（不自动修改 config） ----------

echo "[memory-tencentdb] Step 3: Checking hermes config..."

# 插件已链接到 hermes 插件目录，但默认不自动启用，仅提示
if [ -f "$HERMES_CONFIG" ]; then
    if sed -n '/^memory:/,/^[a-zA-Z]/p' "$HERMES_CONFIG" | grep -q "provider: memory_tencentdb"; then
        echo "[memory-tencentdb] memory.provider already set to memory_tencentdb"
    else
        echo "[memory-tencentdb] Plugin installed but NOT enabled by default."
        echo "[memory-tencentdb] To enable tdai memory, add/edit in $HERMES_CONFIG:"
        echo ""
        echo "    memory:"
        echo "      provider: memory_tencentdb"
        echo ""
    fi
else
    echo "[memory-tencentdb] WARN: $HERMES_CONFIG not found, please run install_hermes_ubuntu.sh first"
fi

# ---------- Step 4: 配置 Gateway 环境变量 ----------

echo "[memory-tencentdb] Step 4: Setting up Gateway environment..."

# 构建 Gateway 启动命令
# 使用 sh -c 包裹，先 cd 到插件目录再启动 Gateway（ESM 解析需要）
#
# 解析 node 绝对路径写入 GATEWAY_CMD（fix: issue #19）
# 当 Hermes 或独立 Gateway 以 systemd service 运行时，systemd 不会
# source 任何 user shell rc 文件，nvm/asdf 注入的 PATH 不存在。
# 用 `command -v node` 在 install 时解析绝对路径，并改用 Node 原生
# `--import tsx/esm`（Node >= 20.6 stable）替代 `npx tsx`，
# 让最终命令完全不依赖运行时 PATH。
NODE_BIN="$(command -v node || true)"
if [ -z "$NODE_BIN" ]; then
    echo "[ERROR] 'node' not found in PATH; cannot generate Gateway start command." >&2
    echo "[ERROR] If you installed Node via nvm/asdf, source the loader script first:" >&2
    echo "[ERROR]   source ~/.bashrc   # or 'nvm use <version>'" >&2
    exit 1
fi
echo "[memory-tencentdb] Resolved node: $NODE_BIN"

GATEWAY_CMD="sh -c 'cd $TDAI_INSTALL_DIR && exec \"$NODE_BIN\" --import tsx/esm src/gateway/server.ts'"

# ── 4a: /etc/profile.d（SSH 交互式登录场景） ──
# 写入 /etc/profile.d 持久化环境变量，供 SSH 手动执行 `hermes` 时使用。
# 注意：LLM 相关变量（API key、model 等）需要用户后续手动配置
ENVFILE="/etc/profile.d/memory-tencentdb-env.sh"
cat << ENVEOF | sudo tee "$ENVFILE" > /dev/null
# memory-tencentdb Gateway 环境变量
export MEMORY_TENCENTDB_GATEWAY_CMD="$GATEWAY_CMD"
export MEMORY_TENCENTDB_GATEWAY_HOST="127.0.0.1"
export MEMORY_TENCENTDB_GATEWAY_PORT="8420"
# LLM 配置（按需修改）
# export MEMORY_TENCENTDB_LLM_API_KEY="sk-..."
# export MEMORY_TENCENTDB_LLM_BASE_URL="https://api.openai.com/v1"
# export MEMORY_TENCENTDB_LLM_MODEL="gpt-4o"
ENVEOF

echo "[memory-tencentdb] Environment variables written to $ENVFILE"

# ── 4b: ~/.hermes/.env（systemd service 场景） ──
# hermes-gateway 通过 systemd user service 启动时不会 source /etc/profile.d/*.sh，
# 但 hermes 的 run.py 启动时会 load_dotenv("~/.hermes/.env")。
# 因此必须将关键变量同步写入 .env，否则 systemd 场景下 Gateway 无法自动启动。
HERMES_ENV="$HERMES_HOME/.env"

_append_or_update_env() {
    local key="$1"
    local value="$2"
    local file="$3"
    if [ ! -f "$file" ]; then
        touch "$file"
    fi
    # 移除已有的同名变量行（含注释掉的和带引号的），再追加
    sed -i "/^${key}=/d" "$file"
    sed -i "/^# *${key}=/d" "$file"
    # python-dotenv 要求含空格/引号/特殊字符的值用双引号包裹
    echo "${key}=\"${value}\"" >> "$file"
}

_append_or_update_env "MEMORY_TENCENTDB_GATEWAY_CMD" "$GATEWAY_CMD" "$HERMES_ENV"
_append_or_update_env "MEMORY_TENCENTDB_GATEWAY_HOST" "127.0.0.1"   "$HERMES_ENV"
_append_or_update_env "MEMORY_TENCENTDB_GATEWAY_PORT" "8420"         "$HERMES_ENV"

echo "[memory-tencentdb] Gateway env vars also written to $HERMES_ENV (for systemd service)"

# ---------- 清理 ----------

rm -rf "$TEMP_DOWNLOAD"

# ---------- 验证安装 ----------

echo ""
echo "=========================================="
echo "[memory-tencentdb] Installation Summary"
echo "=========================================="
echo "  Root dir:       $MEMORY_TENCENTDB_ROOT"
echo "  tdai source:    $TDAI_INSTALL_DIR"
echo "  tdai data dir:  $TDAI_DATA_DIR"
echo "  Hermes config:  $HERMES_CONFIG"
echo "  Env file:       $ENVFILE"
echo ""
echo "  Installed files in tdai dir:"
ls -la "$TDAI_INSTALL_DIR"/ 2>/dev/null | head -20 || echo "  (none)"
echo ""

# 验证 hermes 插件文件存在（在解压目录中）
PLUGIN_SRC="$TDAI_INSTALL_DIR/hermes-plugin/memory/memory_tencentdb"
MISSING=0
for f in __init__.py plugin.yaml client.py supervisor.py; do
    if [ ! -f "$PLUGIN_SRC/$f" ]; then
        echo "  [WARN] Missing: $PLUGIN_SRC/$f"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 0 ]; then
    echo "  [OK] All hermes plugin files present"
fi

# 验证 Gateway 入口存在
if [ -f "$TDAI_INSTALL_DIR/src/gateway/server.ts" ]; then
    echo "  [OK] Gateway entry point found"
else
    echo "  [WARN] Gateway server.ts not found at $TDAI_INSTALL_DIR/src/gateway/server.ts"
fi

# 验证 node_modules 已安装
if [ -d "$TDAI_INSTALL_DIR/node_modules" ]; then
    echo "  [OK] Gateway node_modules installed"
else
    echo "  [WARN] Gateway node_modules not found"
fi

echo ""
echo "[memory-tencentdb] Done!"
echo ""
echo "  NOTE: Before using the memory plugin, configure LLM credentials in ~/.hermes/.env:"
echo "    MEMORY_TENCENTDB_LLM_API_KEY=your-api-key"
echo "    MEMORY_TENCENTDB_LLM_BASE_URL=https://api.openai.com/v1"
echo "    MEMORY_TENCENTDB_LLM_MODEL=gpt-4o"
echo ""
echo "  (For systemd-managed hermes-gateway, ~/.hermes/.env is the authoritative config."
echo "   /etc/profile.d/ is only used for interactive SSH sessions.)"
echo ""
