#!/bin/bash
# done_marker_validate.sh — 标记 done 前验证格式 + 产出文件路径
# 用法: bash done_marker_validate.sh "<done条目>"
# 返回: 0=PASS, 1=FAIL, 2=WARN
#
# v2 (2026-06-17): 新格式含 session_id 字段
#   新格式: [N] done | ISO_TIMESTAMP | session_id | task desc → file:/path
#   旧格式: [N] done | ISO_TIMESTAMP | task desc → file:/path

set -euo pipefail

ENTRY="$1"
HOME_DIR="${HOME:-/home/ohtok}"

# ── 格式检测 ──
FIELD_COUNT=$(echo "$ENTRY" | awk -F'|' '{print NF}')

echo "=== done 条目验证 ==="
echo "  条目: $ENTRY"

if [ "$FIELD_COUNT" -ge 4 ]; then
    # 新格式：验证 session_id
    SESSION_ID=$(echo "$ENTRY" | cut -d'|' -f3 | sed 's/^ *//;s/ *$//')
    DESC=$(echo "$ENTRY" | cut -d'|' -f4- | sed 's/^ *//;s/ *$//')
    
    if ! echo "$SESSION_ID" | grep -qE '^[0-9]{8}_[0-9]{6}_[^[:space:]]{4,}$'; then
        echo "❌ FAIL: session_id 格式不正确: [$SESSION_ID]"
        echo "   期望格式: YYYYMMDD_HHMMSS_XXXX...（至少 4 位后缀，如 20260617_112314_f3b554）"
        echo "   用 echo \$HERMES_SESSION_ID 获取当前 session_id"
        exit 1
    fi
    echo "  ✅ session_id: $SESSION_ID"
else
    # 旧格式：无 session_id
    SESSION_ID=""
    DESC=$(echo "$ENTRY" | cut -d'|' -f3- | sed 's/^ *//;s/ *$//')
    echo "  ⚠️  旧格式（无 session_id），建议升级为新格式"
fi

# ── 提取 desc（去掉 → file: 后缀）──
CLEAN_DESC=$(echo "$DESC" | sed 's/[[:space:]]*→ file:[^[:space:]]*//g' | sed 's/^ *//;s/ *$//')
echo "  任务: $CLEAN_DESC"

# ── 提取 → file: 路径 ──
FILE_PATHS=$(echo "$ENTRY" | sed -n 's/.*→ file:\(.*\)$/\1/p')

if [ -z "$FILE_PATHS" ]; then
    # 无 → file: 标记
    if echo "$CLEAN_DESC" | grep -qE "(确认了|想清楚了|讨论|方向|决策|定了|算了|跳过)"; then
        echo "✅ PASS: 纯讨论/决策任务，不需要产出文件"
        exit 0
    elif [ ${#CLEAN_DESC} -lt 20 ]; then
        echo "✅ PASS: 描述很短，可能为讨论任务"
        exit 0
    else
        echo "⚠️  WARN: 非讨论任务缺少 → file: 路径"
        echo "   L2 语义验证将尝试从 write_file hook log 补发现。"
        echo "   如确实无文件产出，请确认。否则请追加 → file:{路径}"
        exit 2
    fi
fi

# ── 验证每个路径存在 ──
HAS_VALID=0
HAS_MISSING=0
for raw_path in $FILE_PATHS; do
    path="${raw_path/#\~/$HOME_DIR}"
    if [ -f "$path" ]; then
        echo "  ✅ 路径存在: $raw_path"
        HAS_VALID=1
    else
        echo "  ❌ 路径不存在: $raw_path"
        HAS_MISSING=1
    fi
done

if [ $HAS_MISSING -eq 1 ]; then
    if [ $HAS_VALID -eq 1 ]; then
        echo "⚠️  部分路径有效，建议修正或删除无效路径再标记 done"
        exit 2
    else
        echo "❌ FAIL: 所有 → file: 路径均不存在"
        exit 1
    fi
fi

echo "✅ PASS: 格式正确，所有产出文件路径验证通过"
exit 0
