#!/bin/bash
# ============================================================================
# Layer 2: 記憶蒸餾
# 每日 23:00 執行
# 職責：從今日日誌提取重要事實，蒸餾到 MEMORY.md
# ============================================================================

set -e

MEMORY_DIR="/home/ubuntu/.openclaw/memory"
CACHE_DIR="/home/ubuntu/.openclaw/cache"
LOG_FILE="/tmp/pipeline-memory-distill.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)
TODAY=$(date +%Y-%m-%d)

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

log "=== 記憶蒸餾 ==="

DAILY_LOG="$MEMORY_DIR/$TODAY.md"

if [ ! -f "$DAILY_LOG" ]; then
    log "無今日日誌，跳過蒸餾"
    exit 0
fi

# 蒸餾邏輯
python3 << 'PYEOF'
import json
import re
from datetime import datetime

MEMORY_DIR = "/home/ubuntu/.openclaw/memory"
TODAY = datetime.now().strftime("%Y-%m-%d")
DAILY_LOG = f"{MEMORY_DIR}/{TODAY}.md"

# 讀取今日日誌
with open(DAILY_LOG) as f:
    content = f.read()

# 蒸餾信號
DISTILL_SIGNALS = [
    "完成", "修復", "新增", "建立", "發現",
    "決定", "設立", "✅", "🔧", "❌",
    "成功", "失敗", "錯誤", "問題", "優化"
]

distilled = []
lines = content.split("\n")

for line in lines:
    # 檢查是否包含蒸餾信號
    for signal in DISTILL_SIGNALS:
        if signal in line:
            # 清理格式
            cleaned = re.sub(r'[#*`]', '', line).strip()
            if len(cleaned) > 10 and len(cleaned) < 200:
                distilled.append(cleaned)
            break

# 去重
distilled = list(dict.fromkeys(distilled))

# 讀取現有 MEMORY.md
memory_file = f"{MEMORY_DIR}/MEMORY.md"
if os.path.exists(memory_file):
    with open(memory_file) as f:
        memory_content = f.read()
else:
    memory_content = "# MEMORY.md\n\n"

# 檢查是否已有今日蒸餾
if f"## Memory Distillation — {TODAY}" in memory_content:
    print(f"今日 ({TODAY}) 已蒸餾，跳過")
else:
    # 新增蒸餾區塊
    distill_block = f"\n## Memory Distillation — {TODAY}\n"
    distill_block += f"*Auto-extracted from daily logs. Review and keep significant items.*\n\n"
    
    if distilled:
        distill_block += "### Significant Events\n"
        for item in distilled[:10]:  # 最多 10 項
            distill_block += f"  - {item}\n"
    else:
        distill_block += "### Significant Events\n  - (No significant events today)\n"
    
    # 找到 ## Memory Distillation 或文檔末尾，插入
    import os
    
    with open(memory_file, "a") as f:
        f.write(distill_block)
    
    print(f"蒸餾完成: {len(distilled)} 項")

PYEOF

log "=== 蒸餾完成 ==="