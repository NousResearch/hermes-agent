#!/bin/bash
# ============================================================================
# Layer 1: 早晨資料收集
# 06:00 執行
# 職責：一次收集所有需要的資料到 cache/
# 特性：純 Shell，無 AI，快速執行
# ============================================================================

set -e

CACHE_DIR="/home/ubuntu/.openclaw/cache"
LOG_FILE="/tmp/pipeline-morning-fetch.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

mkdir -p "$CACHE_DIR"

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

log "=== 開始早晨資料收集 ==="

# 1. 天氣
log "抓取天氣..."
curl -s --max-time 10 "wttr.in/Taipei?format=j1" > "$CACHE_DIR/weather.json" 2>/dev/null
if [ $? -eq 0 ]; then
    log "✅ 天氣成功"
else
    log "❌ 天氣失敗"
fi

# 2. 行事曆 (gog)
log "抓取行事曆..."
if command -v gog &> /dev/null; then
    gog calendar today > "$CACHE_DIR/calendar.json" 2>/dev/null
    log "✅ 行事曆成功"
else
    echo "[]" > "$CACHE_DIR/calendar.json"
    log "⚠️ gog 不可用"
fi

# 3. 任務
log "處理任務..."
if [ -f "/home/ubuntu/.openclaw/memory/yao-todos.md" ]; then
    python3 << 'PYEOF' > /dev/null
import json
import re

with open("/home/ubuntu/.openclaw/memory/yao-todos.md") as f:
    content = f.read()

# 簡單解析 markdown tasks
tasks = []
lines = content.split("\n")
for line in lines:
    if line.strip().startswith("- [ ]"):
        tasks.append({"done": False, "text": line.strip()[6:].strip()})
    elif line.strip().startswith("- [x]"):
        tasks.append({"done": True, "text": line.strip()[6:].strip()})

with open("/home/ubuntu/.openclaw/cache/tasks.json", "w") as f:
    json.dump({"tasks": tasks, "count": len(tasks), "pending": sum(1 for t in tasks if not t["done"])}, f)
PYEOF
    log "✅ 任務成功"
else
    echo '{"tasks": [], "count": 0, "pending": 0}' > "$CACHE_DIR/tasks.json"
    log "⚠️ 無任務檔案"
fi

# 4. 持股
log "複製持股資料..."
if [ -f "/home/ubuntu/.openclaw/memory/portfolio.json" ]; then
    cp /home/ubuntu/.openclaw/memory/portfolio.json "$CACHE_DIR/holdings.json"
    log "✅ 持股成功"
else
    echo '{}' > "$CACHE_DIR/holdings.json"
    log "⚠️ 無持股檔案"
fi

# 5. 股票快取 (觸發一次股票收集)
log "觸發股票收集..."
bash /home/ubuntu/.openclaw/scripts/pipeline/stock/fetch.sh >> "$LOG_FILE" 2>&1

log "=== 早晨資料收集完成 ==="

# 顯示狀態
echo "=== Morning Cache Status ==="
ls -la "$CACHE_DIR/"