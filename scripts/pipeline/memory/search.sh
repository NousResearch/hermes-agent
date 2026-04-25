#!/bin/bash
# ============================================================================
# Memory Pipeline - Layer 3: 搜尋過去記憶
# 按需觸發 (其他 Pipeline 可呼叫)
# 職責：搜尋相關歷史，提供上下文
# ============================================================================

set -e

MEMORY_DIR="/home/ubuntu/.openclaw/memory"
CACHE_DIR="/home/ubuntu/.openclaw/cache/memory"
LOG_FILE="/tmp/pipeline-memory-search.log"
TIMESTAMP=$(date +%Y%m%d-%H%M)

mkdir -p "$CACHE_DIR"

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

# ============================================================================
# 搜尋函數
# ============================================================================

# 搜尋關鍵字
search_keyword() {
    local keyword="$1"
    local days="${2:-30}"  # 預設30天
    
    log "搜尋關鍵字: '$keyword' (近 $days 天)"
    
    python3 << PYEOF
import os
import re
from datetime import datetime, timedelta

memory_dir = "$MEMORY_DIR"
keyword = "$keyword"
days = $days

# 計算日期範圍
end_date = datetime.now()
start_date = end_date - timedelta(days=days)

results = []

# 搜尋記憶檔案
search_files = [
    f"{memory_dir}/MEMORY.md",
]

# 加入每日日誌 (近30天)
for i in range(days):
    date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
    daily_log = f"{memory_dir}/{date}.md"
    if os.path.exists(daily_log):
        search_files.append(daily_log)

# 搜尋
for filepath in search_files:
    if not os.path.exists(filepath):
        continue
    
    with open(filepath) as f:
        content = f.read()
    
    # 簡單關鍵字匹配
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if keyword.lower() in line.lower():
            # 取得上下文 (前後各2行)
            context_start = max(0, i-2)
            context_end = min(len(lines), i+3)
            context = "\n".join(lines[context_start:context_end])
            
            results.append({
                "file": os.path.basename(filepath),
                "line": i+1,
                "match": line.strip()[:100],
                "context": context[:300]
            })

# 去重
seen = set()
unique_results = []
for r in results:
    key = r["match"][:50]
    if key not in seen:
        seen.add(key)
        unique_results.append(r)

print(f"找到 {len(unique_results)} 個結果")
for r in unique_results[:5]:
    print(f"\n[{r['file']}:{r['line']}]")
    print(f"  {r['match'][:80]}")

# 保存結果
import json
output = {
    "query": keyword,
    "results": unique_results[:10],
    "count": len(unique_results),
    "searched_at": datetime.now().isoformat()
}

with open("$CACHE_DIR/search-results.json", "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

PYEOF
}

# 搜尋相似案例
search_similar() {
    local topic="$1"
    local days="${2:-90}"
    
    log "搜尋相似案例: '$topic' (近 $days 天)"
    
    python3 << PYEOF
import os
import json
from datetime import datetime, timedelta

memory_dir = "$MEMORY_DIR"
topic = "$topic"
days = $days

# 關鍵字擴展
related_keywords = [
    topic,
    topic.replace(" ", ""),
    topic.upper(),
    topic.lower(),
]

# 收集過去的蒸餾記錄
similar_cases = []

# 讀取 MEMORY.md 中的 Memory Distillation 區塊
memory_file = f"{memory_dir}/MEMORY.md"
if os.path.exists(memory_file):
    with open(memory_file) as f:
        content = f.read()
    
    # 找到 Memory Distillation 區塊
    in_distill = False
    for line in content.split("\n"):
        if "Memory Distillation" in line:
            in_distill = True
            continue
        if in_distill and line.startswith("## "):
            break
        if in_distill and line.strip():
            # 檢查是否包含相關關鍵字
            for kw in related_keywords:
                if kw.lower() in line.lower():
                    similar_cases.append(line.strip()[:150])
                    break

# 搜尋每日日誌
end_date = datetime.now()
for i in range(days):
    date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
    daily_log = f"{memory_dir}/{date}.md"
    
    if os.path.exists(daily_log):
        with open(daily_log) as f:
            content = f.read()
        
        for line in content.split("\n"):
            for kw in related_keywords:
                if kw.lower() in line.lower() and len(line.strip()) > 20:
                    similar_cases.append(f"[{date}] {line.strip()[:100]}")
                    break

# 去重
similar_cases = list(dict.fromkeys(similar_cases))

print(f"找到 {len(similar_cases)} 個相似案例")
for case in similar_cases[:5]:
    print(f"  • {case[:80]}")

# 保存
output = {
    "query": topic,
    "similar_cases": similar_cases[:20],
    "count": len(similar_cases),
    "searched_at": datetime.now().isoformat()
}

with open("$CACHE_DIR/similar-cases.json", "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

PYEOF
}

# ============================================================================
# 主流程
# ============================================================================

main() {
    local action="${1:-status}"
    local query="${2:-}"
    
    log "=== Memory Search L3 ==="
    log "Action: $action, Query: $query"
    
    case "$action" in
        "keyword")
            if [ -z "$query" ]; then
                echo "用法: search.sh keyword <關鍵字> [天數]"
                exit 1
            fi
            search_keyword "$query" "${3:-30}"
            ;;
        "similar")
            if [ -z "$query" ]; then
                echo "用法: search.sh similar <主題> [天數]"
                exit 1
            fi
            search_similar "$query" "${3:-90}"
            ;;
        "status")
            echo "=== Memory Search 狀態 ==="
            echo "可用指令:"
            echo "  search.sh keyword <關鍵字> [天數]  - 搜尋關鍵字"
            echo "  search.sh similar <主題> [天數]    - 搜尋相似案例"
            echo ""
            echo "最近搜尋結果:"
            ls -la "$CACHE_DIR"/search-*.json 2>/dev/null || echo "無"
            ;;
        *)
            echo "未知指令: $action"
            exit 1
            ;;
    esac
    
    log "=== Memory Search 完成 ==="
}

main "$@"
