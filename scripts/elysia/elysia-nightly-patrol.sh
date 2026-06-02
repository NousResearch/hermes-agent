#!/bin/bash
# =============================================================================
# elysia-nightly-patrol.sh — Elysia's Nightly Backup & Update Detector
# =============================================================================
#
# [WHAT] 每日凌晨自动灾备推送 + 侦测上游更新
# [WHY]  确保本地修改不会丢失，且第一时间发现官方更新（但不自动执行）
# [HOW]  推送到 fork 灾备 → fetch origin → 对比差异 → 写入审计日志
#
# Schedule: cronjob "妖精小姐夜间巡航" (job_id: 1dfe51567fa7)
#           每日凌晨 1:30 执行
#
# Safety:
#   - 仅推送到 fork，不动 origin
#   - 仅侦测更新，绝不自动执行 rebase
#   - 分支不匹配时静默退出
#
# Log: ~/.hermes/scripts/cron_audit.log
#
# Author: elysias-pink-realm (Elysia's private maintenance branch)
# Version: 1.0.0
# Created: 2026-06-03
# =============================================================================

set -euo pipefail

# === 配置常量 ===
REPO_DIR="$HOME/.hermes/hermes-agent"
LOG_FILE="$HOME/.hermes/scripts/cron_audit.log"
BRANCH="elysias-pink-realm"
REMOTE="fork"

cd "$REPO_DIR"

# === 分支检查 ===
current_branch=$(git branch --show-current)
if [ "$current_branch" != "$BRANCH" ]; then
    echo "[$(date)] ⚠️ 当前在 $current_branch 分支，跳过巡航" >> "$LOG_FILE"
    exit 0
fi

# === Step 1: 灾备推送到 fork ===
echo "[$(date)] 🌙 夜间巡航开始..." >> "$LOG_FILE"
if git push "$REMOTE" HEAD 2>&1; then
    echo "[$(date)] ✅ 灾备推送成功 → $REMOTE/$BRANCH" >> "$LOG_FILE"
else
    echo "[$(date)] ❌ 灾备推送失败，请检查网络或认证" >> "$LOG_FILE"
fi

# === Step 2: 侦测远端更新（仅侦测，不执行）===
git fetch origin 2>/dev/null || true

if ! git diff --quiet main..origin/main 2>/dev/null; then
    file_count=$(git diff --stat main..origin/main | tail -1 | grep -oE '[0-9]+ files? changed' | grep -oE '[0-9]+' || echo "未知")
    core_touched=$(git diff --name-only main..origin/main | grep -cE "(memory_tool|skill_manager_tool|skills_tool)\.py" || echo "0")
    echo "[$(date)] 🔔 远端有更新！文件数: $file_count | 核心路由触及: $core_touched | 等待妖精小姐白天审批~" >> "$LOG_FILE"
else
    echo "[$(date)] 😴 远端无更新，今晚可以安心睡觉~" >> "$LOG_FILE"
fi

echo "[$(date)] 🌙 夜间巡航结束" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"
