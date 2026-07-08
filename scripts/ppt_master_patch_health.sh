#!/bin/bash
# ppt_master_patch_health.sh — 每周检查 PPT Master 补丁是否仍存在
# no_agent cron: PASS=静默, FAIL=推微信告警
set -euo pipefail

PATCHES=(
    "EXECUTOR-CN:skills/ppt-master/references/executor-base.md:PATCH EXECUTOR-CN"
    "STRATEGIST-CN:skills/ppt-master/references/strategist.md:PATCH STRATEGIST-CN"
    "VISUAL-STYLES-CN:skills/ppt-master/references/visual-styles/_index.md:PATCH VISUAL-STYLES-CN"
)

MISSING=""
REPO_DIR="$HOME/ppt-master"

for entry in "${PATCHES[@]}"; do
    name=$(echo "$entry" | cut -d: -f1)
    file=$(echo "$entry" | cut -d: -f2)
    marker=$(echo "$entry" | cut -d: -f3)
    
    if ! grep -q "$marker" "$REPO_DIR/$file" 2>/dev/null; then
        MISSING="$MISSING  🔴 $name: $file\n"
    fi
done

if [ -n "$MISSING" ]; then
    echo "⚠️ PPT Master 补丁丢失——升级后需重新 Apply"
    echo ""
    echo -e "$MISSING"
    echo ""
    echo "修复：查看 ~/hermes-local/yangyang/设计文档/PPT-Master定制补丁记录.md"
    exit 1
fi
exit 0
