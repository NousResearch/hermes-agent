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
#   ./safe_push.sh "<commit message>" [file1 file2 ...]
#   ./safe_push.sh "feat(router): 注入本地化强路由约束"
#   ./safe_push.sh "feat(iac): 新增脚本" scripts/elysia/new_script.sh
#
# Exit codes:
#   0 — 推送成功（或无变更跳过）
#   1 — 错误（目录不存在、分支不匹配等）
#
# Safety:
#   - 远端硬编码为 `fork`（非官方 origin），杜绝误推上游
#   - 分支硬编码为 `elysias-pink-realm`，拒绝在其他分支执行
#   - 默认仅暂存已跟踪文件（git add -u），避免捡到垃圾
#   - 传入额外文件路径可显式添加新文件
#
# Author: elysias-pink-realm (Elysia's private maintenance branch)
# Version: 1.1.0
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

# === 解析参数 ===
# $1 = commit message（必填）
# $2... = 额外要 add 的文件路径（可选）
COMMIT_MSG="${1:?用法: safe_push.sh \"<commit message>\" [file1 file2 ...]}"
shift
EXTRA_FILES=("$@")

# === 暂存变更 ===
# 默认仅暂存已跟踪文件的修改，避免 git add . 捡到未跟踪的垃圾
git add -u
echo "[SafePush] 已暂存已跟踪文件的修改"

# 如果指定了额外文件路径，显式添加
if [ ${#EXTRA_FILES[@]} -gt 0 ]; then
    git add "${EXTRA_FILES[@]}"
    echo "[SafePush] 已暂存额外文件: ${EXTRA_FILES[*]}"
fi

# === 检查是否有变更 ===
if git diff --cached --quiet; then
    echo "[SafePush] 无新变更，跳过提交"
    exit 0
fi

# === 提交 ===
git commit -m "$COMMIT_MSG"
echo "[SafePush] 已提交: $COMMIT_MSG"

# === 推送到灾备远端 ===
git push -u "$REMOTE" "$BRANCH"

echo "[SafePush] ✅ 成功推送到云端灾备: $REMOTE/$BRANCH"
