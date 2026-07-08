#!/bin/bash
# dual_review.sh — 轻量双模型复查
set -eo pipefail
# 用法: bash dual_review.sh <产出文件路径>
# 用 Qwen 顺序复查，找事实错/逻辑跳/自相矛盾，追加到文件末尾

FILE="$1"
if [ ! -f "$FILE" ]; then
    echo "❌ 文件不存在: $FILE"
    exit 1
fi

CONTENT=$(cat "$FILE")
CHAR_COUNT=$(echo "$CONTENT" | wc -c)

# 只审 >500 字的文件
if [ "$CHAR_COUNT" -lt 500 ]; then
    exit 0
fi

# 取前 8000 字符给 Qwen（够覆盖核心内容）
SNIPPET=$(echo "$CONTENT" | head -c 8000)

PROMPT=$(cat << 'PROMPT_END'
你是轻量级 QA 审查员。读下面的产出文件，只找三类问题：
1. **事实错**：声称的数据/路径/命令/人名与实际矛盾
2. **逻辑跳**：A推论B但中间缺了一步
3. **自相矛盾**：文件内两处说法互斥

不要评价框架好不好、不要建议新方向、不要重写——只找硬伤。
格式：
### 🔍 双模型审查
- **事实错**：[如有，逐条列出；如无，写"无"]
- **逻辑跳**：[如有，逐条列出；如无，写"无"]
- **自相矛盾**：[如有，逐条列出；如无，写"无"]

以下是产出文件：
PROMPT_END
)

# 调用 Qwen
REVIEW=$(echo "$PROMPT

$SNIPPET" | python3 /home/USER/.hermes/scripts/qwen_call.py 2>/dev/null)

if [ -z "$REVIEW" ]; then
    exit 0  # silent fail — don't block delivery
fi

# 追加到文件末尾
echo "" >> "$FILE"
echo "---" >> "$FILE"
echo "$REVIEW" >> "$FILE"

echo "✅ 双模型审查已追加"
