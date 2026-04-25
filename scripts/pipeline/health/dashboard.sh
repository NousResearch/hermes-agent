#!/bin/bash
# ============================================================================
# Dashboard - 簡化版狀態顯示
# ============================================================================

CACHE_DIR="/home/ubuntu/.openclaw/cache/pipeline"

echo ""
echo "📊 Pipeline Status $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo ""

# 讀取 Dashboard Status
if [ -f "$CACHE_DIR/status.json" ]; then
    python3 << 'PYEOF'
import json
from datetime import datetime

try:
    with open("/home/ubuntu/.openclaw/cache/pipeline/status.json") as f:
        data = json.load(f)
    
    # Pipeline 狀態
    print("Pipeline:")
    for p in data.get("pipelines", []):
        name = p["name"]
        status = p["status"]
        last_run = p.get("last_run", "從未")
        
        if status == "ok":
            icon = "🟢"
        elif status == "error":
            icon = "🔴"
        else:
            icon = "🟡"
        
        print(f"  {icon} {name:18} | {last_run}")
    
    print()
    
    # Cache 狀態
    print("Cache:")
    for c in data.get("cache", []):
        name = c["name"]
        exists = c.get("exists", False)
        age = c.get("age_seconds", 0)
        
        if exists:
            if age < 3600:
                icon = "🟢"
                age_str = f"{age}s ago"
            else:
                icon = "🟡"
                age_str = f"{age//3600}h ago"
        else:
            icon = "🔴"
            age_str = "MISSING"
        
        print(f"  {icon} {name:12} | {age_str}")
    
    print()
    
    # 系統狀態
    sys = data.get("system", {})
    disk = sys.get("disk_usage", 0)
    mem = sys.get("memory_free_mb", 0)
    openclaw = sys.get("openclaw_running", False)
    
    print("System:")
    disk_icon = "🟢" if disk < 80 else "🟡"
    print(f"  {disk_icon} 磁碟: {disk}%")
    mem_icon = "🟢" if mem > 500 else "🟡"
    print(f"  {mem_icon} 記憶體: {mem}MB free")
    openclaw_icon = "🟢" if openclaw else "🔴"
    print(f"  {openclaw_icon} OpenClaw: {'Running' if openclaw else 'Stopped'}")

except Exception as e:
    print(f"讀取失敗: {e}")
PYEOF
else
    echo "⚠️ 無 Status 資料"
fi

echo ""