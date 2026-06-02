#!/bin/bash
# =============================================================================
# safe_update.sh — Elysia's Deterministic Safe Update Script
# =============================================================================
#
# [WHAT] 安全地将上游 (origin/main) 的更新合入 elysias-pink-realm 分支
# [WHY]  LLM 手写 git rebase 可能：忘记 abort、忽略冲突、盲目接受巨型更新
# [HOW]  三层防爆门控（文件数/核心文件/冲突检测）+ 失败自动回滚
#
# Usage:
#   ./safe_update.sh
#
# Exit codes:
#   0 — 更新成功合入
#   1 — 被门控拦截或 rebase 冲突（已自动回滚）
#
# Safety:
#   - 分支硬编码为 `elysias-pink-realm`，拒绝在其他分支执行
#   - 文件数 > 30 触发防爆拦截（巨型更新需人工审查）
#   - 核心路由文件被触及时拦截（需确认补丁兼容性）
#   - rebase 冲突时自动 abort 回滚，不留脏状态
#
# Output:
#   - 更新日志在 ---UPDATE_LOG_START--- 和 ---UPDATE_LOG_END--- 之间
#   - LLM 应解析此日志生成发版播报
#
# Author: elysias-pink-realm (Elysia's private maintenance branch)
# Version: 1.0.0
# Created: 2026-06-03
# =============================================================================

set -euo pipefail

# === 配置常量（硬编码，不可由 LLM 修改）===
REPO_DIR="$HOME/.hermes/hermes-agent"
BRANCH="elysias-pink-realm"
MAX_FILES=30
CORE_FILES_PATTERN="(memory_tool|skill_manager_tool|skills_tool)\.py"

# === 目录检查 ===
cd "$REPO_DIR" || {
    echo "[SafeUpdate] ERROR: 无法进入仓库目录 $REPO_DIR"
    exit 1
}

# === 分支安全检查 ===
current=$(git branch --show-current)
if [ "$current" != "$BRANCH" ]; then
    echo "[SafeUpdate] ERROR: 当前在 [$current] 分支，预期 [$BRANCH]，拒绝更新"
    exit 1
fi

# === Step 1: 获取上游情报 ===
echo "[SafeUpdate] 正在获取上游更新..."
git fetch origin main

# === Step 2: 防爆检查 — 文件数量门控 ===
CHANGE_COUNT=$(git diff --name-only main..origin/main | wc -l | tr -d ' ')
echo "[SafeUpdate] 远端更新文件数: $CHANGE_COUNT / 阈值: $MAX_FILES"

if [ "$CHANGE_COUNT" -gt "$MAX_FILES" ]; then
    echo "ERROR: [Gatekeeper] 官方更新文件超过 ${MAX_FILES} 个（实际: ${CHANGE_COUNT}），判定为高风险重构，拒绝合入。"
    echo "ERROR: 请爸爸人工审查后再决定是否合入。"
    exit 1
fi

# === Step 3: 防爆检查 — 核心路由文件保护 ===
CORE_TOUCHED=$(git diff --name-only main..origin/main | grep -cE "$CORE_FILES_PATTERN" || true)
if [ "$CORE_TOUCHED" -gt 0 ]; then
    echo "ERROR: [Gatekeeper] 官方更新触及了核心路由文件（${CORE_TOUCHED}个），拒绝自动合入。"
    echo "ERROR: 被触及的文件:"
    git diff --name-only main..origin/main | grep -E "$CORE_FILES_PATTERN"
    echo "ERROR: 请爸爸确认路由补丁是否需要手动迁移。"
    exit 1
fi

# === Step 4: 同步 main 指针 ===
git branch -f main origin/main
echo "[SafeUpdate] main 指针已同步到 origin/main"

# === Step 5: 安全变基（失败自动回滚）===
if ! git rebase main; then
    git rebase --abort 2>/dev/null || true
    echo "ERROR: Rebase 冲突，已自动回滚到变基前状态。"
    exit 1
fi

# === Step 6: 输出更新日志 ===
echo "[SafeUpdate] ✅ 远端更新已安全合入。"
echo "---UPDATE_LOG_START---"
git log main..HEAD --oneline
echo "---UPDATE_LOG_END---"
