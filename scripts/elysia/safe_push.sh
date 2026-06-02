#!/bin/bash
# =============================================================================
# safe_push.sh — Elysia's Deterministic Safe Push Script
# =============================================================================
#
# [WHAT] 将本地修改安全推送到灾备远端（fork），消除 LLM 手写 git 命令的幻觉风险
# [WHY]  LLM 生成 git push 时可能：推错远端(官方repo)、写错分支、遗漏安全检查
# [HOW]  所有参数硬编码 + 分支校验 + 错误处理，LLM 只需传入 commit message
#
# Usage:
#   ./safe_push.sh "<commit message>"
#   ./safe_push.sh "feat(router): 注入本地化强路由约束"
#
# Exit codes:
#   0 — 推送成功（或无变更跳过）
#   1 — 错误（目录不存在、分支不匹配等）
#
# Safety:
#   - 远端硬编码为 `fork`（非官方 origin），杜绝误推上游
#   - 分支硬编码为 `elysias-pink-realm`，拒绝在其他分支执行
#   - 无变更时优雅跳过，不会产生空提交
#
# Author: elysias-pink-realm (Elysia's private maintenance branch)
# Version: 1.0.0
# Created: 2026-06-03
# =============================================================================

set -euo pipefail

# === 配置常量（硬编码，不可由 LLM 修改）===
BRANCH="elysias-pink-realm"
REMOTE="fork"
REPO_DIR="$HOME/.hermes/hermes-agent"

# === 目录检查 ===
cd "$REPO_DIR" || {
    echo "[SafePush] ERROR: 无法进入仓库目录 $REPO_DIR"
    exit 1
}

# === 分支安全检查 ===
current=$(git branch --show-current)
if [ "$current" != "$BRANCH" ]; then
    echo "[SafePush] ERROR: 当前在 [$current] 分支，预期 [$BRANCH]，拒绝推送"
    echo "[SafePush] 请先执行: git checkout $BRANCH"
    exit 1
fi

# === 提交所有变更 ===
git add .

# 使用传入的 commit message，若未传入则使用默认值
COMMIT_MSG="${1:-chore(auto): 自动提交 $(date +%Y%m%d_%H%M%S)}"

# commit 返回非零表示无变更，此时跳过而非报错
if git commit -m "$COMMIT_MSG" 2>/dev/null; then
    echo "[SafePush] 已提交: $COMMIT_MSG"
else
    echo "[SafePush] 无新变更，跳过提交"
    exit 0
fi

# === 推送到灾备远端 ===
git push -u "$REMOTE" "$BRANCH"

echo "[SafePush] ✅ 成功推送到云端灾备: $REMOTE/$BRANCH"
