#!/usr/bin/env bash
# setup-shared-memory.sh - 共同一个记忆配置一键脚本（WSL 端）
# 用法: bash setup-shared-memory.sh
# 作用: 1. 写固定 token  2. 杀旧 9119  3. 启 loopback dashboard  4. 等 HTTP 200

set -e

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
TOKEN_FILE="$HERMES_HOME/dashboard.token"
TOKEN_VALUE="${HERMES_SHARED_TOKEN:-hermes-shared-2026-06-16}"
DASHBOARD_LOG="/tmp/dashboard_9119.log"

echo "=== Hermes 共同一个记忆配置 ==="
echo "  HERMES_HOME: $HERMES_HOME"
echo "  Token file: $TOKEN_FILE"
echo "  Token: $TOKEN_VALUE (24 字符)"
echo ""

# Step 1: 写固定 token
echo "[1/4] 写固定 token ..."
mkdir -p "$HERMES_HOME"
echo -n "$TOKEN_VALUE" > "$TOKEN_FILE"
chmod 600 "$TOKEN_FILE"
echo "  OK: $TOKEN_FILE (24 字符)"

# Step 2: 杀旧 9119
echo ""
echo "[2/4] 杀旧 9119 ..."
old=$(ps aux | grep "dashboard.*9119" | grep -v grep | awk '{print $2}')
if [ -n "$old" ]; then
    echo "  旧 PID: $old"
    kill -9 $old 2>/dev/null || true
    sleep 2
    echo "  OK: 旧进程已杀"
else
    echo "  无旧进程（首次跑）"
fi

# Step 3: 启 loopback dashboard
echo ""
echo "[3/4] 启 loopback dashboard (PID $(date +%s))..."
HERMES_DASHBOARD_SESSION_TOKEN="$TOKEN_VALUE" \
  "$(which hermes)" dashboard --no-open --host 127.0.0.1 --port 9119 \
  > "$DASHBOARD_LOG" 2>&1 &
DASH_PID=$!
disown
echo "  启动 PID: $DASH_PID"
echo "  log: $DASH_LOG"

# Step 4: 等 HTTP 200（轮询 60s）
echo ""
echo "[4/4] 等 HTTP 200 ..."
for i in {1..30}; do
    sleep 2
    code=$(curl -s -o /dev/null -w "%{http_code}" -H "X-Hermes-Session-Token: $TOKEN_VALUE" http://127.0.0.1:9119/api/status 2>/dev/null)
    if [ "$code" = "200" ]; then
        echo "  ✓ attempt $i: HTTP 200 (耗时 $((i*2))s)"
        echo ""
        echo "=== 配置完成！==="
        echo ""
        echo "Windows 端启动 desktop："
        echo "  powershell -NoProfile -ExecutionPolicy Bypass -File templates/start-desktop.ps1"
        echo ""
        echo "或双击桌面 'Hermes Desktop.lnk'"
        exit 0
    fi
    if [ $((i % 5)) -eq 0 ]; then
        echo "  attempt $i: HTTP $code (继续等)"
    fi
done

# 60s 后还没起来，看 log
echo "  X  60s 内没起来，看 log："
echo ""
tail -30 "$DASH_LOG"
exit 1
