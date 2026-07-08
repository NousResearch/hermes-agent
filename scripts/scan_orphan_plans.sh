#!/bin/bash
# 孤儿 Plan 检测 —— 每周日运行
set -eo pipefail
# 检测：①plan文件无队列条目 ②队列条目指向不存在的plan

PLAN_DIR="$HOME/hermes-local/yangyang/plans"
QUEUE_FILE="$HOME/.hermes/queue/pending.txt"
DAYS=7

orphans=()
broken=()

# 扫描过去 N 天内创建的 plan 文件
while IFS= read -r -d '' plan; do
    rel_path="${plan#$PLAN_DIR/}"
    if ! grep -qF "$rel_path" "$QUEUE_FILE" 2>/dev/null; then
        orphans+=("$rel_path")
    fi
done < <(find "$PLAN_DIR" -name "*.md" -mtime -$DAYS -print0 2>/dev/null)

# 扫描队列中的 pending 条目指向的 plan 是否存在
while IFS= read -r line; do
    plan_ref=$(echo "$line" | grep -oP 'plans/\S+\.md' | head -1 || true)
    if [ -n "$plan_ref" ] && [ ! -f "$HOME/hermes-local/yangyang/$plan_ref" ]; then
        broken+=("$plan_ref")
    fi
done < <(grep 'pending' "$QUEUE_FILE" 2>/dev/null)

# 输出
if [ ${#orphans[@]} -eq 0 ] && [ ${#broken[@]} -eq 0 ]; then
    echo "[SILENT]"
    exit 0
fi

echo "🔍 孤儿 Plan 检测 ($(date '+%Y-%m-%d'))"
echo ""

if [ ${#orphans[@]} -gt 0 ]; then
    echo "🟡 无队列引用的 plan（$DAYS天内）："
    for p in "${orphans[@]}"; do
        echo "  - plans/$p"
    done
    echo ""
fi

if [ ${#broken[@]} -gt 0 ]; then
    echo "🔴 队列指向不存在的 plan："
    for p in "${broken[@]}"; do
        echo "  - $p"
    done
fi
