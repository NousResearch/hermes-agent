#!/bin/bash
# ============================================================================
# Layer 3: 任務格式化輸出
# 每日 09:00 執行
# 職責：讀取分析結果，格式化，發送到 Discord #task
# ============================================================================

set -e

CACHE_DIR="/home/ubuntu/.openclaw/cache/tasks"
LOG_FILE="/tmp/pipeline-task-format.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

# 讀取 webhook URL
WEBHOOK_URL=$(python3 -c "
import json
with open('/home/ubuntu/.openclaw/config/system-config.json') as f:
    config = json.load(f)
# 發送到 #task 頻道 (用 stock_monitor 的 webhook 先)
print(config['webhooks']['stock_monitor']['url'])
" 2>/dev/null)

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

log "=== 任務格式化輸出 ==="

format_output() {
    python3 << 'PYEOF'
import json
from datetime import datetime

# 讀取分析結果
try:
    with open("/home/ubuntu/.openclaw/cache/tasks_analysis.json") as f:
        analysis = json.load(f)
except:
    # 如果沒有分析結果，讀取原始任務
    try:
        with open("/home/ubuntu/.openclaw/cache/tasks.json") as f:
            data = json.load(f)
            analysis = {
                "pending_count": data["stats"]["pending"],
                "done_count": data["stats"]["done"],
                "alerts": [],
                "tasks": data.get("tasks", [])
            }
    except:
        print("無法讀取任務資料")
        exit(1)

pending = analysis.get("pending_count", 0)
done = analysis.get("done_count", 0)
alerts = analysis.get("alerts", [])
tasks = analysis.get("tasks", [])

now = datetime.now()
today = now.strftime("%Y-%m-%d")

msg = f"📋 **任務報告 {today}**\n"
msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"

# 統計
msg += f"✅ 已完成: {done}\n"
msg += f"⏳ 待完成: {pending}\n\n"

# 警示
if alerts:
    msg += "🚨 **需要關注**\n"
    
    overdue = [a for a in alerts if a.get("type") == "overdue"]
    due_soon = [a for a in alerts if a.get("type") in ["due_today", "due_soon"]]
    urgent = [a for a in alerts if a.get("type") == "no_deadline"]
    
    if overdue:
        msg += f"\n🔴 **已超時 ({len(overdue)})**\n"
        for a in overdue[:3]:
            text = a.get("task", "")[:50]
            days = a.get("days", 0)
            msg += f"  • ~~{text}...~~ (超時 {days} 天)\n"
    
    if due_soon:
        msg += f"\n🟡 **即將到期 ({len(due_soon)})**\n"
        for a in due_soon[:3]:
            text = a.get("task", "")[:50]
            days = a.get("days", 0)
            if days == 0:
                msg += f"  • {text}... (今天！)\n"
            else:
                msg += f"  • {text}... ({days} 天後)\n"
    
    if urgent:
        msg += f"\n🟠 **緊急無期限 ({len(urgent)})**\n"
        for a in urgent[:3]:
            text = a.get("task", "")[:50]
            msg += f"  • {text}...\n"

# 待辦任務列表 (如果沒有警示)
if not alerts and tasks:
    pending_tasks = [t for t in tasks if not t.get("done")][:5]
    if pending_tasks:
        msg += "\n📌 **待辦任務**\n"
        for t in pending_tasks:
            text = t.get("text", "")[:50]
            section = t.get("section", "")
            msg += f"  • [{section}] {text}...\n"

msg += "\n---\n"
msg += f"_🕐 更新於 {now.strftime('%H:%M')}_"

print(msg)
PYEOF
}

MSG=$(format_output)

# 發送到 Discord
log "發送任務報告..."

response=$(echo "$MSG" | jq -Rs '{content: .}' | curl -s -X POST \
    -H "Content-Type: application/json" \
    -d @- \
    "$WEBHOOK_URL")

if [ $? -eq 0 ]; then
    log "✅ 發送成功"
    echo "✅ Task report sent"
else
    log "❌ 發送失敗"
    echo "❌ Failed"
fi

log "=== 完成 ==="