#!/bin/bash
# ============================================================================
# Layer 1: 任務收集
# 每日 08:00 執行
# 職責：讀取任務檔案，寫入 cache/tasks.json
# ============================================================================

set -e

CACHE_DIR="/home/ubuntu/.openclaw/cache/tasks"
LOG_FILE="/tmp/pipeline-task-fetch.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

mkdir -p "$CACHE_DIR"

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

log "=== 任務收集 ==="

# 讀取任務檔案
TODO_FILE="/home/ubuntu/.openclaw/memory/yao-todos.md"

if [ -f "$TODO_FILE" ]; then
    # 解析 Markdown 任務
    python3 << 'PYEOF'
import json
import re
from datetime import datetime

with open("/home/ubuntu/.openclaw/memory/yao-todos.md") as f:
    content = f.read()

tasks = []
lines = content.split("\n")
section = "general"

for line in lines:
    line = line.strip()
    
    # 檢查章節標題
    if line.startswith("## "):
        section = line[3:].strip()
        continue
    
    # 解析任務
    if "[ ]" in line:
        # 未完成
        text = re.sub(r'\[ \]\s*', '', line)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # 移除 Bold
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # 移除 Links
        tasks.append({
            "done": False,
            "text": text.strip(),
            "section": section,
            "created": datetime.now().isoformat()
        })
    elif "[x]" in line.lower():
        # 已完成
        text = re.sub(r'\[x\]\s*', '', line)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
        tasks.append({
            "done": True,
            "text": text.strip(),
            "section": section,
            "completed_at": datetime.now().isoformat()
        })

# 按 section 分組
by_section = {}
for t in tasks:
    sec = t.get("section", "general")
    if sec not in by_section:
        by_section[sec] = []
    by_section[sec].append(t)

result = {
    "tasks": tasks,
    "by_section": by_section,
    "stats": {
        "total": len(tasks),
        "done": sum(1 for t in tasks if t["done"]),
        "pending": sum(1 for t in tasks if not t["done"])
    },
    "fetched_at": datetime.now().isoformat()
}

with open("/home/ubuntu/.openclaw/cache/tasks.json", "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"收集了 {len(tasks)} 個任務")
print(f"待完成: {result['stats']['pending']}")
PYEOF
    
    log "✅ 任務收集成功"
else
    echo '{"tasks": [], "stats": {"total": 0, "done": 0, "pending": 0}}' > /home/ubuntu/.openclaw/cache/tasks.json
    log "⚠️ 任務檔案不存在"
fi

log "=== 完成 ==="