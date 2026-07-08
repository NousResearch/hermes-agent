#!/bin/bash
# check_progress_log.sh — 扫描进度日志，输出未闭合任务
# 用法: bash check_progress_log.sh [log_path] [offset_path]
# 输出: in-progress:N (N=0 时静默，N>0 时逐条列出)
# 退出码: 0=PASS

set -euo pipefail

LOG="${1:-$HOME/.hermes/state/progress.log}"
OFFSET_FILE="${2:-$HOME/.hermes/state/progress.offset}"
SUMMARY_FILE="$HOME/.hermes/state/running-tasks-summary.txt"

# 初始化
touch "$LOG" 2>/dev/null || true
last_offset=$(cat "$OFFSET_FILE" 2>/dev/null || echo 0)
total_lines=$(wc -l < "$LOG" 2>/dev/null || echo 0)

# 无新行 → 静默
if [ "$total_lines" -le "$last_offset" ]; then
    echo -n "" > "$SUMMARY_FILE"
    echo "in-progress:0"
    exit 0
fi

# 从上次扫描位置读取新行
new_lines=$(tail -n +$((last_offset + 1)) "$LOG" 2>/dev/null || echo "")

# 解析新旧行，用关联数组维护 UUID→状态
declare -A states
declare -A descriptions
declare -A timestamps
declare -A paths

# 先扫描全量日志（从开头到当前），每次扫描都重建状态——因为旧行可能被 Agent 手动追加"完成"
while IFS= read -r line; do
    # 跳过空行和注释
    [[ -z "$line" || "$line" == \#* ]] && continue
    
    # 格式: [timestamp] uuid:xxxxxxxx: description: 状态 → 路径
    # 用正则提取
    if [[ "$line" =~ ^\[(.*)\]\ uuid:([a-f0-9]{8}):\ (.*):\ (启动|完成|失败)\ \→\ (.*)$ ]]; then
        uuid="${BASH_REMATCH[2]}"
        ts="${BASH_REMATCH[1]}"
        desc="${BASH_REMATCH[3]}"
        status="${BASH_REMATCH[4]}"
        path="${BASH_REMATCH[5]}"
        
        states["$uuid"]="$status"
        descriptions["$uuid"]="$desc"
        timestamps["$uuid"]="$ts"
        paths["$uuid"]="$path"
    fi
done < "$LOG"

# 查找未闭合的：状态为"启动"且无后续"完成"或"失败"
unclosed=()
for uuid in "${!states[@]}"; do
    if [ "${states[$uuid]}" = "启动" ]; then
        unclosed+=("$uuid")
    fi
done

# 输出
count=${#unclosed[@]}

if [ "$count" -eq 0 ]; then
    echo -n "" > "$SUMMARY_FILE"
    echo "in-progress:0"
else
    # 写摘要文件供 prompt 注入
    {
        echo "## ⚡ 活跃后台任务 ($count)"
        echo ""
        for uuid in "${unclosed[@]}"; do
            echo "- **${descriptions[$uuid]}** — 启动于 ${timestamps[$uuid]}"
            echo "  - UUID: $uuid"
            echo "  - 路径: ${paths[$uuid]}"
            echo ""
        done
    } > "$SUMMARY_FILE"
    
    # stdout 输出
    echo "in-progress:$count"
    for uuid in "${unclosed[@]}"; do
        echo "  [$uuid] ${descriptions[$uuid]} (${timestamps[$uuid]}) → ${paths[$uuid]}"
    done
fi

# 更新 offset
echo "$total_lines" > "$OFFSET_FILE"
