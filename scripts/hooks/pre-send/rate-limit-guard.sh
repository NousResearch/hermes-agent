#!/bin/bash
# Rate-Limit Guard (Pre-Send Hook · 通用版 v2)
#
# 在通过任何消息平台推送前，检查失败次数
# 双窗口检查：30min 短期频率 + 24h 持续累积
# 任一超阈值 → 拒绝推送，避免刷平台计数
#
# 借鉴：affaan-m/everything-claude-code 的 gateguard-fact-force 模式
# 适用：所有 send_message 推送场景
# 默认配置：iLink（WeChat 通道），可改 PATTERN 适配其他平台

set -e

# === 可调参数 ===
LOG="${HERMES_RATE_GUARD_LOG:-/home/semperaug/.hermes/logs/rate-limit-guard.log}"
GATEWAY_LOG="${HERMES_GATEWAY_LOG:-/home/semperaug/.hermes/logs/gateway.log}"
WINDOW_MIN="${HERMES_RATE_WINDOW_MIN:-30}"
MAX_FAILURES="${HERMES_RATE_MAX_FAILURES:-3}"
WINDOW_24H_MAX="${HERMES_RATE_24H_MAX:-5}"
PATTERN="${HERMES_RATE_PATTERN:-iLink.*rate limited}"

mkdir -p "$(dirname "$LOG")"
[ -f "$GATEWAY_LOG" ] || GATEWAY_LOG="/home/semperaug/.hermes/logs/gateway.log"

# 统计最近 $WINDOW_MIN 分钟内失败次数（短期频率）
CUTOFF_SHORT="$(date -d "$WINDOW_MIN minutes ago" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')"
COUNT=$(grep -E "$PATTERN" "$GATEWAY_LOG" 2>/dev/null | \
    awk -v cutoff="$CUTOFF_SHORT" '{ if (($1 " " $2) >= cutoff) print }' | wc -l)

# 统计最近 24h 失败次数（持续累积）
CUTOFF_24H="$(date -d '24 hours ago' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')"
COUNT_24H=$(grep -E "$PATTERN" "$GATEWAY_LOG" 2>/dev/null | \
    awk -v cutoff="$CUTOFF_24H" '{ if (($1 " " $2) >= cutoff) print }' | wc -l)

TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

REFUSE_REASON=""
if [ "$COUNT" -ge "$MAX_FAILURES" ]; then
    REFUSE_REASON="短期 ${WINDOW_MIN}min 失败 $COUNT 次 ≥$MAX_FAILURES 阈值"
elif [ "$COUNT_24H" -ge "$WINDOW_24H_MAX" ]; then
    REFUSE_REASON="24h 累计失败 $COUNT_24H 次 ≥$WINDOW_24H_MAX 阈值（持续累积中）"
fi

if [ -n "$REFUSE_REASON" ]; then
    echo "[$TIMESTAMP] ❌ REFUSED: $REFUSE_REASON (short=$COUNT/${WINDOW_MIN}min, 24h=$COUNT_24H, pattern='$PATTERN')" >> "$LOG"
    cat <<EOF
❌ Rate-Limit Guard: 拒绝推送

$REFUSE_REASON

详情：
- 短期（${WINDOW_MIN}min 内）：$COUNT 次失败
- 长期（24h 内）：$COUNT_24H 次失败
- 阈值：短期 ≥$MAX_FAILURES / 24h ≥$WINDOW_24H_MAX

可能原因：
- 24h 激活窗口过期（需要用户主动和 bot 对话激活）
- 频率限制触发深度限流（1-2 小时才能恢复）
- 累计失败次数过多（平台计数器高位）

建议：
1. 让用户在推送通道里说任意一句话激活窗口
2. 等待 1-2 小时让平台计数清零
3. 本次推送改用 local 输出
4. 不要继续试探（会刷计数器让情况更糟）
EOF
    exit 1
fi

echo "[$TIMESTAMP] ✅ ALLOWED: short=$COUNT/${WINDOW_MIN}min, 24h=$COUNT_24H (thresholds: $MAX_FAILURES/$WINDOW_24H_MAX, pattern='$PATTERN')" >> "$LOG"
exit 0
