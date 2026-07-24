#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────
# Hermes Agent WebSocket Server Plugin - 一键安装脚本
# ───────────────────────────────────────────────────────────
# 用法：
#   bash docs/install-ws-server.sh
#
# 从仓库根目录运行，将插件文件安装到默认路径。
# ───────────────────────────────────────────────────────────

set -euo pipefail

# ── 颜色 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ── 检测 hermes-agent 仓库根目录 ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$REPO_DIR/run_agent.py" ] && [ ! -f "$REPO_DIR/cli.py" ]; then
    log_warn "当前目录不是 hermes-agent 仓库根目录（未找到 run_agent.py）"
    log_info "尝试从脚本位置推导..."

    # 尝试向上查找
    CANDIDATE="$SCRIPT_DIR"
    for _ in $(seq 1 5); do
        if [ -f "$CANDIDATE/run_agent.py" ] || [ -f "$CANDIDATE/cli.py" ]; then
            REPO_DIR="$CANDIDATE"
            break
        fi
        CANDIDATE="$(cd "$CANDIDATE/.." && pwd)"
    done

    if [ ! -f "$REPO_DIR/run_agent.py" ]; then
        log_error "无法定位 hermes-agent 仓库根目录"
        log_error "请将此脚本放在仓库的 docs/ 目录下运行"
        exit 1
    fi
fi

log_info "Hermes Agent 仓库: $REPO_DIR"

# ── 源文件 ──
SRC_DIR="$REPO_DIR/plugins/platforms/ws_server"
if [ ! -d "$SRC_DIR" ]; then
    log_error "插件源目录不存在: $SRC_DIR"
    log_error "请确保在 feature/ws-server 分支上运行此脚本"
    exit 1
fi

# ── 目标路径 ──
# 默认安装到 ~/.hermes/plugins/platforms/ws_server/
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
INSTALL_DIR="$HERMES_HOME/plugins/platforms/ws_server"

log_info "安装目标: $INSTALL_DIR"

# ── 创建目录 ──
mkdir -p "$INSTALL_DIR"

# ── 复制文件 ──
cp "$SRC_DIR/__init__.py" "$INSTALL_DIR/__init__.py"
cp "$SRC_DIR/adapter.py"  "$INSTALL_DIR/adapter.py"
cp "$SRC_DIR/plugin.yaml" "$INSTALL_DIR/plugin.yaml"

log_ok "文件已复制到 $INSTALL_DIR"
log_info "  ├── __init__.py"
log_info "  ├── adapter.py"
log_info "  └── plugin.yaml"

# ── 验证 ──
if [ -f "$INSTALL_DIR/__init__.py" ] && [ -f "$INSTALL_DIR/adapter.py" ] && [ -f "$INSTALL_DIR/plugin.yaml" ]; then
    log_ok "插件安装完成"
else
    log_error "插件安装不完整，请检查 $INSTALL_DIR"
    exit 1
fi

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo -e "  下一步"
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo ""
echo "  1. 配置认证密钥："
echo "     hermes config set gateway.platforms.ws_server.extra.api_key 'your-secret-key'"
echo ""
echo "  2. 重启网关："
echo "     supervisorctl restart hermes"
echo "     # 或: hermes gateway restart"
echo ""
echo "  3. 验证："
echo "     curl http://127.0.0.1:8765/health"
echo ""

# ── 可选：自动配置 ──
if command -v hermes &>/dev/null; then
    echo -ne "${YELLOW}是否自动配置 ws_server 并重启网关？(y/N): ${NC}"
    read -r answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        log_info "正在配置..."

        # 生成随机密钥
        if command -v openssl &>/dev/null; then
            KEY=$(openssl rand -hex 32)
        else
            KEY="ws-server-$(date +%s)-$$"
        fi

        # 通过 hermes config 命令配置
        hermes config set gateway.platforms.ws_server.enabled true 2>/dev/null || true
        hermes config set gateway.platforms.ws_server.extra.api_key "$KEY" 2>/dev/null || true
        hermes config set gateway.platforms.ws_server.extra.host "0.0.0.0" 2>/dev/null || true
        hermes config set gateway.platforms.ws_server.extra.port "8765" 2>/dev/null || true

        log_ok "配置已完成 (api_key: $KEY)"

        # 重启网关
        log_info "正在重启网关..."
        if command -v supervisorctl &>/dev/null && supervisorctl status hermes &>/dev/null 2>&1; then
            supervisorctl restart hermes
            log_ok "网关已重启 (supervisor)"
        else
            log_warn "未检测到 supervisor，请手动重启网关"
        fi
    fi
fi

echo -e "${GREEN}✅ 安装完成！${NC}"