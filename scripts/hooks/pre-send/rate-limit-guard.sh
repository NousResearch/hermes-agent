#!/bin/bash
# Rate-Limit Guard (Pre-Send Hook · 通用版)
#
# 在通过任何消息平台推送前，检查最近失败次数
# 失败过多 → 拒绝推送，避免刷平台计数
#
# 双窗口检查：
#   - 短期窗口（30min）：高频失败检测
#   - 长期窗口（24h）：持续累积检测
# 任一超阈值 → 拒绝
#
# 借鉴：affaan-m/everything-claude-code 的 gateguard-fact-force 模式
# 适用：所有 send_message 推送场景
# 默认配置：iLink（WeChat 通道）
#
# 注意：hermes-agent AGENTS.md 规定非密钥配置必须走 config.yaml，**不能引入新 HERMES_* env vars**。
# 故本脚本阈值 hardcode（需要调阈值请直接改本脚本）。
# 跨平台适配请编辑下面的 PATTERN。

set -e

# === Hardcoded 阈值（要改请直接编辑）===
LOG="/home/semperaug/.hermes/logs/rate-limit-guard.log"
GATEWAY_LOG_FALLBACK="/home/semperaug/.hermes/logs/gateway.log"
WINDOW_MIN=30
MAX_FAILURES=3
WINDOW_24H_MAX=5
PATTERN="iLink.*rate limited"

# === 自动 fallback ===
mkdir -p "$(dirname "$LOG")"
[ ! -f "$GATEWAY_LOG_FALLBACK" ] && GATEWAY_LOG_FALLBACK=""

# 统计最近 $WINDOW_MIN 分钟内失败次数（短期频率）
CUTOFF_SHORT="$(date -d "$WINDOW_MIN minutes ago" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')"
COUNT=$(grep -hE "$PATTERN" $GATEWAY_LOG_FALLBACK 2>/dev/null | \
    awk -v cutoff="$CUTOFF_SHORT" '{ if (($1 " " $2) >= cutoff) print }' | wc -l)

# 统计最近 24h 失败次数（持续累积）
CUTOFF_24H="$(date -d '24 hours ago' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')"
COUNT_24H=$(grep -hE "$PATTERN" $GATEWAY_LOG_FALLBACK 2>/dev/null | \
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
- 匹配 pattern: $PATTERN

可能原因：
- 24h 激活窗口过期（需要用户主动和 bot 对话激活）
- 频率限制触发深度限流（1-2 小时才能恢复）
- 累计失败次数过多（平台计数器高位）

建议：
1. 让用户在推送通道里说任意一句话激活窗口
2. 等待 1-2 小时让平台计数清零
3. 本次推送改用 local 输出
4. 不要继续试探（会刷计数器让情况更糟）

适配其他平台：编辑本脚本的 PATTERN 变量（hermes-agent AGENTS.md 规定非密钥配置走 config.yaml，不引入新 HERMES_* env vars）。
EOF
    exit 1
fi

echo "[$TIMESTAMP] ✅ ALLOWED: short=$COUNT/${WINDOW_MIN}min, 24h=$COUNT_24H (thresholds: $MAX_FAILURES/$WINDOW_24H_MAX, pattern='$PATTERN')" >> "$LOG"
exit 0
