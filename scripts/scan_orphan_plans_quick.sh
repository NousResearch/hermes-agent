#!/bin/bash
# Plan 孤儿快速扫描 —— 每轮会话末尾调用
# 比周日版轻量：只查文件名不读内容，不扫队列断链
set -o pipefail

PLAN_DIR="$HOME/local/USER/plans"
QUEUE_FILE="$HOME/.hermes/queue/pending.txt"
DAYS=7

if [ ! -d "$PLAN_DIR" ]; then
    echo "orphans:0"
    exit 0
fi

count=0
first=""
while IFS= read -r -d '' plan; do
    rel="${plan#$PLAN_DIR/}"
    if ! grep -qF "$rel" "$QUEUE_FILE" 2>/dev/null; then
        count=$((count + 1))
        [ "$count" -eq 1 ] && first="$rel"
    fi
done < <(find "$PLAN_DIR" -name "*.md" -mtime -$DAYS -print0 2>/dev/null || true)

if [ $count -eq 0 ]; then
    echo "orphans:0"
else
    # 最多显示前 3 个
    shown=0
    echo -n "orphans:$count "
    while IFS= read -r -d '' plan; do
        rel="${plan#$PLAN_DIR/}"
        if ! grep -qF "$rel" "$QUEUE_FILE" 2>/dev/null; then
            shown=$((shown + 1))
            echo -n "$rel"
            [ $shown -lt $count ] && [ $shown -lt 3 ] && echo -n " | "
            [ $shown -ge 3 ] && break
        fi
    done < <(find "$PLAN_DIR" -name "*.md" -mtime -$DAYS -print0 2>/dev/null || true)
    echo
fi
