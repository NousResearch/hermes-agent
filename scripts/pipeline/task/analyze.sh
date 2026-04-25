#!/bin/bash
# ============================================================================
# Layer 2: 任務分析
# 每日 08:30 執行
# 職責：分析任務優先級、截止日期、超時任務
# ============================================================================

set -e

CACHE_DIR="/home/ubuntu/.openclaw/cache/tasks"
ALERTS_DIR="/home/ubuntu/.openclaw/alerts/task"
LOG_FILE="/tmp/pipeline-task-analyze.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

mkdir -p "$ALERTS_DIR"

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

log "=== 任務分析 ==="

# 讀取任務資料
if [ ! -f "$CACHE_DIR/tasks.json" ]; then
    log "無任務資料，跳過"
    exit 0
fi

# 分析
python3 << 'PYEOF'
import json
from datetime import datetime, timedelta
import re

with open("/home/ubuntu/.openclaw/cache/tasks.json") as f:
    data = json.load(f)

tasks = data.get("tasks", [])
pending = [t for t in tasks if not t.get("done")]

alerts = []

# 分析每一個待辦任務
for task in pending:
    text = task.get("text", "")
    section = task.get("section", "general")
    created = task.get("created", "")
    
    # 檢查是否有截止日期
    deadline_match = re.search(r'(\d{1,2})/(\d{1,2})|\[@(\d{4}-\d{2}-\d{2})\]', text)
    has_deadline = deadline_match is not None
    
    # 檢查是否為緊急 (@urgent, !, ASAP)
    is_urgent = any(k in text.lower() for k in ["@urgent", "!", "asap", "盡快", "緊急"])
    
    # 檢查是否為重要 (@important, **)
    is_important = "**" in text or any(k in text for k in ["@important", "重要", "優先"])
    
    # 超時檢查 (如果有截止日期)
    if deadline_match:
        # 嘗試解析截止日期
        try:
            if len(deadline_match.groups()) >= 3 and deadline_match.group(3):
                deadline = datetime.strptime(deadline_match.group(3), "%Y-%m-%d")
                days_until = (deadline - datetime.now()).days
                
                if days_until < 0:
                    alerts.append({
                        "type": "overdue",
                        "task": text,
                        "section": section,
                        "days": abs(days_until)
                    })
                elif days_until == 0:
                    alerts.append({
                        "type": "due_today",
                        "task": text,
                        "section": section
                    })
                elif days_until <= 2:
                    alerts.append({
                        "type": "due_soon",
                        "task": text,
                        "section": section,
                        "days": days_until
                    })
        except:
            pass
    
    # 緊急任務
    if is_urgent and not has_deadline:
        alerts.append({
            "type": "no_deadline",
            "task": text,
            "section": section,
            "priority": "high"
        })

# 統計
analysis = {
    "pending_count": len(pending),
    "done_count": data["stats"]["done"],
    "urgent_count": sum(1 for a in alerts if a.get("type") == "no_deadline"),
    "overdue_count": sum(1 for a in alerts if a.get("type") == "overdue"),
    "due_soon_count": sum(1 for a in alerts if a.get("type") in ["due_today", "due_soon"]),
    "alerts": alerts,
    "analyzed_at": datetime.now().isoformat()
}

# 寫入分析結果
with open("/home/ubuntu/.openclaw/cache/tasks_analysis.json", "w") as f:
    json.dump(analysis, f, ensure_ascii=False, indent=2)

# 寫入警示
if alerts:
    alert_file = f"/home/ubuntu/.openclaw/alerts/task/{datetime.now().strftime('%Y%m%d')}.json"
    with open(alert_file, "w") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

print(f"分析完成:")
print(f"  待完成: {len(pending)}")
print(f"  警示: {len(alerts)}")
for a in alerts[:3]:
    print(f"    - [{a['type']}] {a['task'][:50]}")

PYEOF

log "=== 分析完成 ==="