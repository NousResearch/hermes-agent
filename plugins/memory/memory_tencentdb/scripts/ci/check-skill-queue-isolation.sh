#!/usr/bin/env bash
# scripts/ci/check-skill-queue-isolation.sh
#
# Skill 异步队列红线守护脚本
#
# 目的：在 CI 中阻止任何会污染"记忆模块"的 diff，并阻止 Skill 模块
#       直接 import node:fs（必须走 StorageAdapter）。
#
# 检查项：
#   1) 禁止改动以下记忆相关红线文件 / 目录：
#        - src/core/state/types.ts
#        - src/core/state/local-backend.ts
#        - src/services/pipeline-worker.ts
#        - src/integrations/redis/**          （未来记忆 Redis backend；注意不是 redis-skill）
#   2) 禁止 src/core/skill/** 新增 `from "node:fs"` / `from "fs"` / `from "fs/promises"` 引用。
#
# 用法：
#   - 本地（与 origin/main 对比）：bash scripts/ci/check-skill-queue-isolation.sh
#   - CI 中指定 base：BASE_REF=origin/main bash scripts/ci/check-skill-queue-isolation.sh
#   - 跳过（不推荐，仅紧急绕过）：SKIP_SKILL_QUEUE_ISOLATION=1 bash scripts/ci/check-skill-queue-isolation.sh
#
# 退出码：
#   0：通过
#   1：检测到红线被触碰
#   2：环境/依赖问题（git 不可用等）

set -euo pipefail

if [[ "${SKIP_SKILL_QUEUE_ISOLATION:-0}" == "1" ]]; then
  echo "[skill-queue-isolation] SKIP_SKILL_QUEUE_ISOLATION=1，跳过红线检查（不推荐）"
  exit 0
fi

if ! command -v git >/dev/null 2>&1; then
  echo "[skill-queue-isolation] ERROR: git 不可用" >&2
  exit 2
fi

BASE_REF="${BASE_REF:-origin/main}"
MODE="${MODE:-auto}"   # auto | working-tree | base-diff

# 若 BASE_REF 不存在（例如 shallow clone），回退到 HEAD~1
if ! git rev-parse --verify --quiet "$BASE_REF" >/dev/null; then
  if git rev-parse --verify --quiet "HEAD~1" >/dev/null; then
    BASE_REF="HEAD~1"
  else
    echo "[skill-queue-isolation] WARN: 无法定位 BASE_REF=$BASE_REF 也没有 HEAD~1，跳过检查"
    exit 0
  fi
fi

echo "[skill-queue-isolation] BASE_REF=$BASE_REF MODE=$MODE"

# 拿到 diff 文件清单（A/M/D/R 都算）
# - base-diff：与 BASE_REF 对比（CI 用）
# - working-tree：包含已暂存 + 未暂存的工作区改动（本地用）
# - auto：CI 环境（GITHUB_ACTIONS=true / CI=true）走 base-diff，否则 working-tree
case "$MODE" in
  base-diff)
    CHANGED=$(git diff --name-only --diff-filter=ACMRT "$BASE_REF"...HEAD || true)
    ;;
  working-tree)
    CHANGED=$(git status --porcelain | awk '$1 ~ /^[AM?RC]/ || $1 ~ /^.[AM]/ {print $NF}' | sort -u || true)
    ;;
  auto|*)
    if [[ "${GITHUB_ACTIONS:-}" == "true" || "${CI:-}" == "true" ]]; then
      CHANGED=$(git diff --name-only --diff-filter=ACMRT "$BASE_REF"...HEAD || true)
    else
      # 本地：合并 base..HEAD + 工作区改动，最大化覆盖
      CHANGED=$( { git diff --name-only --diff-filter=ACMRT "$BASE_REF"...HEAD 2>/dev/null || true; \
                   git status --porcelain | awk '$1 ~ /^[AM?RC]/ || $1 ~ /^.[AM]/ {print $NF}'; } | sort -u)
    fi
    ;;
esac

if [[ -z "$CHANGED" ]]; then
  echo "[skill-queue-isolation] 无文件变更，pass"
  exit 0
fi

VIOLATIONS=()

# ── 红线 1：禁止改动的具体文件 ──
FORBIDDEN_FILES=(
  "src/core/state/types.ts"
  "src/core/state/local-backend.ts"
  "src/services/pipeline-worker.ts"
)

for f in "${FORBIDDEN_FILES[@]}"; do
  if echo "$CHANGED" | grep -qx "$f"; then
    VIOLATIONS+=("禁止改动文件：$f")
  fi
done

# ── 红线 2：禁止改动 src/integrations/redis/ 目录（注意：redis-skill 是允许的）──
# 用 awk 精确匹配 `src/integrations/redis/` 前缀但排除 `src/integrations/redis-skill/`
while IFS= read -r f; do
  case "$f" in
    src/integrations/redis-skill/*) ;;  # 允许
    src/integrations/redis/*)
      VIOLATIONS+=("禁止改动记忆 Redis 目录：$f")
      ;;
  esac
done <<< "$CHANGED"

# ── 红线 3：src/core/skill/** 不得直接 import node:fs ──
# 仅对 diff 命中的 .ts 文件检查；测试文件 (*.test.ts / __tests__/) 豁免，
# 因为测试常需要直接搭 tmpdir 脚手架（不会走到运行时产品代码）。
SKILL_TS_CHANGED=$(echo "$CHANGED" | grep -E '^src/core/skill/.*\.ts$' | grep -vE '\.test\.ts$|/__tests__/' || true)

if [[ -n "$SKILL_TS_CHANGED" ]]; then
  while IFS= read -r f; do
    [[ -f "$f" ]] || continue
    # 抓 import 语句中的 fs / node:fs / fs/promises
    if grep -nE 'from[[:space:]]+["'"'"']?(node:fs|fs|fs/promises)["'"'"']?' "$f" >/dev/null; then
      MATCH=$(grep -nE 'from[[:space:]]+["'"'"']?(node:fs|fs|fs/promises)["'"'"']?' "$f" | head -3)
      VIOLATIONS+=("Skill 模块禁止直接 import fs：$f"$'\n'"$MATCH")
    fi
  done <<< "$SKILL_TS_CHANGED"
fi

# ── 汇总 ──
if [[ ${#VIOLATIONS[@]} -gt 0 ]]; then
  echo ""
  echo "==================================================================="
  echo "[skill-queue-isolation] FAIL：检测到 ${#VIOLATIONS[@]} 处红线违规"
  echo "==================================================================="
  for v in "${VIOLATIONS[@]}"; do
    echo "  - $v"
  done
  echo ""
  echo "如需绕过（仅紧急情况），可设置 SKIP_SKILL_QUEUE_ISOLATION=1，但 reviewer 必须 +2"
  echo "参考 ADR：docs/design/2026-06-16-skill-extract-queue.md / 2026-06-16-skill-storage-adapter.md"
  exit 1
fi

echo "[skill-queue-isolation] PASS"
exit 0
