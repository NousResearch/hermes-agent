#!/usr/bin/env bash
# restart-9119.sh - 重启 WSL 9119 dashboard（loopback 模式 + 固定 token）
# 用法: bash restart-9119.sh
# 作用: 杀旧进程 + 启新 dashboard + 轮询等 HTTP 200

set -e

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
TOKEN_FILE="$HERMES_HOME/dashboard.token"
DASHBOARD_LOG="/tmp/dashboard_9119.log"

# 检查 token
if [ ! -f "$TOKEN_FILE" ]; then
    echo "X  $TOKEN_FILE 不存在 — 跑 setup-shared-memory.sh 先"
    exit 1
fi

TOKEN_VALUE=$(cat "$TOKEN_FILE")
echo "=== 重启 9119 dashboard ==="
echo "  Token: $TOKEN_VALUE"
echo ""

# Step 1: 杀旧
echo "[1/3] 杀旧 9119 ..."
old=$(ps aux | grep "dashboard.*9119" | grep -v grep | awk '{print $2}')
if [ -n "$old" ]; then
    kill -9 $old 2>/dev/null || true
    sleep 2
    echo "  OK"
else
    echo "  无旧进程"
fi

# Step 2: 启新
echo ""
echo "[2/3] 启新 dashboard ..."
HERMES_DASHBOARD_SESSION_TOKEN="$TOKEN_VALUE" \
  "$(which hermes)" dashboard --no-open --host 127.0.0.1 --port 9119 \
  > "$DASHBOARD_LOG" 2>&1 &
disown
echo "  启动"

# Step 3: 等 HTTP 200
echo ""
echo "[3/3] 等 HTTP 200 ..."
for i in {1..30}; do
    sleep 2
    code=$(curl -s -o /dev/null -w "%{http_code}" -H "X-Hermes-Session-Token: $TOKEN_VALUE" http://127.0.0.1:9119/api/status 2>/dev/null)
    if [ "$code" = "200" ]; then
        echo "  OK attempt $i: HTTP 200 (耗时 $((i*2))s)"
        exit 0
    fi
    if [ $((i % 5)) -eq 0 ]; then
        echo "  attempt $i: HTTP $code (继续等)"
    fi
done

echo "  X  60s 内没起来"
tail -20 "$DASHBOARD_LOG"
exit 1
