#!/bin/bash
# ============================================================================
# Pipeline Daily Report - 每日 Pipeline 健康報告
# 08:00 執行，發送到 Discord
# ============================================================================

set -e

CACHE_DIR="/home/ubuntu/.openclaw/cache/pipeline"
REPORT_FILE="$CACHE_DIR/daily-report.json"
LOG_FILE="/tmp/pipeline-daily-report.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

mkdir -p "$CACHE_DIR"

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

log "=== Pipeline Daily Report ==="

# ============================================================================
# 收集昨日數據
# ============================================================================

YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)

# 讀取昨日 log
collect_log_data() {
    local name="$1"
    local logs=$(ls -t /tmp/pipeline-${name}-*.log 2>/dev/null | head -3)
    
    local run_count=0
    local success_count=0
    local fail_count=0
    local last_result="unknown"
    
    for log_file in $logs; do
        # 檢查是否是昨天的
        log_date=$(stat -c %y "$log_file" 2>/dev/null | cut -d' ' -f1)
        if [ "$log_date" = "$YESTERDAY" ]; then
            run_count=$((run_count + 1))
            success_count=$((success_count + $(grep -c "成功\|sent\|完成" "$log_file" 2>/dev/null || echo "0")))
            fail_count=$((fail_count + $(grep -c "失敗\|ERROR\|❌" "$log_file" 2>/dev/null || echo "0")))
        fi
    done
    
    if [ $fail_count -gt 0 ]; then
        last_result="error"
    elif [ $success_count -gt 0 ]; then
        last_result="ok"
    else
        last_result="no_data"
    fi
    
    echo "{\"name\":\"$name\",\"runs\":$run_count,\"success\":$success_count,\"fails\":$fail_count,\"last_result\":\"$last_result\"}"
}

# ============================================================================
# 格式化報告
# ============================================================================

format_report() {
    python3 << 'PYEOF'
import json
from datetime import datetime, timedelta

YESTERDAY = datetime.now() - timedelta(days=1)
YESTERDAY_STR = YESTERDAY.strftime("%Y-%m-%d")
YESTERDAY_SHORT = YESTERDAY.strftime("%m/%d")

# Pipeline 資料
pipelines = [
    "stock-fetch", "stock-alert", "stock-format",
    "news-fetch", "news-classify", "news-format",
    "morning-fetch", "morning-format",
    "task-fetch", "task-analyze", "task-format",
    "memory-fetch", "memory-distill", "memory-update",
    "health", "backup"
]

results = []
for name in pipelines:
    import subprocess
    result = subprocess.run(
        f"grep -c '成功\\|sent\\|完成' /tmp/pipeline-{name}-*.log 2>/dev/null | tail -1 || echo 0",
        shell=True, capture_output=True, text=True
    )
    success = int(result.stdout.strip() or "0")
    
    result = subprocess.run(
        f"grep -c '失敗\\|ERROR\\|❌' /tmp/pipeline-{name}-*.log 2>/dev/null | tail -1 || echo 0",
        shell=True, capture_output=True, text=True
    )
    fail = int(result.stdout.strip() or "0")
    
    status = "ok" if fail == 0 else "error" if fail > 0 else "no_data"
    
    results.append({
        "name": name,
        "success": success,
        "fails": fail,
        "status": status
    })

# 統計
total_success = sum(r["success"] for r in results)
total_fails = sum(r["fails"] for r in results)
ok_count = sum(1 for r in results if r["status"] == "ok")
error_count = sum(1 for r in results if r["status"] == "error")

# 格式化訊息
msg = f"📊 **Pipeline 每日報告 {YESTERDAY_SHORT}**\n"
msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"

msg += f"📈 **執行統計**\n"
msg += f"  成功: {total_success} 次\n"
msg += f"  失敗: {total_fails} 次\n"
msg += f"  Pipeline 正常: {ok_count}/{len(results)}\n\n"

# Pipeline 狀態列表
msg += f"**Pipeline 狀態**\n"
for r in results:
    icon = "✅" if r["status"] == "ok" else "❌" if r["status"] == "error" else "⚠️"
    fails_str = f"({r['fails']} 失敗)" if r["fails"] > 0 else ""
    msg += f"  {icon} {r['name']:15} {fails_str}\n"

msg += "\n"

# 失敗的 Pipeline 詳細
failed_pipelines = [r for r in results if r["status"] == "error"]
if failed_pipelines:
    msg += f"🔴 **失敗細節**\n"
    for r in failed_pipelines:
        msg += f"  • **{r['name']}**: {r['fails']} 次失敗\n"
    msg += "\n"

# Cache 狀態
msg += f"**Cache 狀態**\n"
cache_files = [
    ("stock", "/home/ubuntu/.openclaw/cache/stock/04006C.json"),
    ("news", "/home/ubuntu/.openclaw/cache/news/classified.json"),
    ("weather", "/home/ubuntu/.openclaw/cache/weather.json"),
    ("tasks", "/home/ubuntu/.openclaw/cache/tasks.json"),
]
import os
for name, path in cache_files:
    if os.path.exists(path):
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
        age_str = f"{int(age.total_seconds() // 3600)}h ago" if age.total_seconds() > 3600 else f"{int(age.total_seconds() // 60)}m ago"
        msg += f"  ✅ {name}: {age_str}\n"
    else:
        msg += f"  ⚠️ {name}: 不存在\n"

msg += "\n"
msg += f"_🕐 報告時間: {datetime.now().strftime('%H:%M')}_"

print(msg)

# 保存報告
report = {
    "date": YESTERDAY_STR,
    "generated_at": datetime.now().isoformat(),
    "pipelines": results,
    "summary": {
        "total_success": total_success,
        "total_fails": total_fails,
        "ok_count": ok_count,
        "error_count": error_count
    }
}

with open("/home/ubuntu/.openclaw/cache/pipeline/daily-report.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

PYEOF
}

# ============================================================================
# 發送報告到 Discord
# ============================================================================

send_report() {
    local msg="$1"
    
    WEBHOOK_URL=$(python3 -c "
import json
with open('/home/ubuntu/.openclaw/config/system-config.json') as f:
    config = json.load(f)
print(config['webhooks']['stock_monitor']['url'])
" 2>/dev/null)
    
    if [ -z "$WEBHOOK_URL" ]; then
        log "無 Webhook URL"
        echo "無法發送: 無 Webhook URL"
        return 1
    fi
    
    echo "$msg" | jq -Rs '{content: .}' | curl -s -X POST \
        -H "Content-Type: application/json" \
        -d @- \
        "$WEBHOOK_URL"
    
    if [ $? -eq 0 ]; then
        log "✅ 報告已發送"
        echo "✅ Daily report sent"
    else
        log "❌ 發送失敗"
        echo "❌ Failed to send"
    fi
}

# ============================================================================
# 主流程
# ============================================================================

main() {
    # 收集數據
    log "收集昨日數據..."
    
    # 格式化報告
    log "格式化報告..."
    REPORT=$(format_report)
    
    # 顯示報告
    echo "$REPORT"
    
    # 發送到 Discord
    log "發送報告..."
    send_report "$REPORT"
    
    log "=== Daily Report 完成 ==="
}

main "$@"