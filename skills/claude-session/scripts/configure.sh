#!/usr/bin/env bash
# claude-session 安装配置钩子
# 用途：检查依赖 + 自动配置环境变量 + 权限预设
# 使用：bash ~/.hermes/skills/claude-session/scripts/configure.sh
# 触发：首次启用 claude_session toolset 时由 tools_config.py 调用

set -euo pipefail

HERMES_ENV="${HERMES_HOME:-$HOME/.hermes}/.env"
VAR_NAME="HERMES_STREAM_STALE_TIMEOUT"
RECOMMENDED_VALUE="300"
ISSUES=()
ALL_OK=true

echo "=== 🤖 Claude Code Session — 安装配置 ==="
echo ""

# ── 1. 依赖检查 ──────────────────────────────────────────────────────────────

# tmux
echo "Checking tmux..."
if command -v tmux >/dev/null 2>&1; then
    TMUX_VER=$(tmux -V 2>/dev/null || echo "unknown")
    echo "  ✅ tmux: $TMUX_VER"
else
    echo "  ❌ tmux: 未安装"
    ISSUES+=("tmux — Install: apt install tmux / brew install tmux")
    ALL_OK=false
fi

# Claude Code CLI
echo "Checking Claude Code CLI..."
if command -v claude >/dev/null 2>&1; then
    CLAUDE_VER=$(claude --version 2>/dev/null | head -1 || echo "unknown")
    echo "  ✅ Claude Code: $CLAUDE_VER"
else
    echo "  ❌ Claude Code CLI: 未安装"
    ISSUES+=("Claude Code CLI — Install: npm install -g @anthropic-ai/claude-code")
    ALL_OK=false
fi

echo ""

# ── 2. 环境变量配置 ──────────────────────────────────────────────────────────

echo "Configuring environment variables..."

if grep -q "^${VAR_NAME}=" "$HERMES_ENV" 2>/dev/null; then
    current=$(grep "^${VAR_NAME}=" "$HERMES_ENV" | cut -d'=' -f2)
    if [ "$current" -lt "$RECOMMENDED_VALUE" ] 2>/dev/null; then
        echo "  ⚠️  ${VAR_NAME}=${current} (recommend ≥ ${RECOMMENDED_VALUE})"
    else
        echo "  ✅ ${VAR_NAME}=${current}"
    fi
else
    echo "" >> "$HERMES_ENV"
    echo "# Claude Session 优化 - 防止 Stream Stalled 中断" >> "$HERMES_ENV"
    echo "${VAR_NAME}=${RECOMMENDED_VALUE}" >> "$HERMES_ENV"
    echo "  ✅ Auto-configured ${VAR_NAME}=${RECOMMENDED_VALUE}"
    echo "     (Prevents 'Stream Stalled' errors during long tasks)"
fi

echo ""

# ── 3. Claude Code 权限预设（可选） ──────────────────────────────────────────

CLAUDE_DIR="$HOME/.claude"
if [ -d "$CLAUDE_DIR" ]; then
    echo "Detected Claude Code config directory..."
    SETTINGS="$CLAUDE_DIR/settings.json"

    if [ ! -f "$SETTINGS" ]; then
        echo "  ℹ️  No settings.json yet — will be created on first claude run"
    else
        echo "  ✅ Claude Code settings.json exists"
    fi
    echo ""
fi

# ── 4. 结果汇总 ──────────────────────────────────────────────────────────────

if [ "$ALL_OK" = true ]; then
    echo "✅ All dependencies met — claude_session is ready to use."
else
    echo "⚠️  Missing dependencies:"
    for issue in "${ISSUES[@]}"; do
        echo "    • $issue"
    done
    echo ""
    echo "  claude_session will not work until these are installed."
fi

echo ""
echo "⚠️  Important: Restart Hermes Gateway for env changes to take effect!"
echo "   Ctrl+C the gateway, then run: hermes gateway run"
echo ""
echo "Configuration complete ✅"
