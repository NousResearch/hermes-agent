#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Detect venv
if [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Check key — only test existence, never print or persist the value
if [ -z "${NINEROUTER_KEY:-}" ]; then
    echo "[cron] ERROR: NINEROUTER_KEY not set" >&2
    echo "[cron] Hint: export NINEROUTER_KEY=... or source ~/.zshenv" >&2
    exit 1
fi

RUN_DIR=$(mktemp -d /tmp/skill-sleep-run-XXXXXX)
echo "[cron] Run dir: $RUN_DIR"

# ── Stage 1: MINE ──────────────────────────────────────────────────────────
echo "[cron] === Stage 1: MINE ==="
if [ ! -f pipeline/mine.py ]; then
    echo "[cron] SKIP: pipeline/mine.py not found — aborting" >&2
    rm -rf "$RUN_DIR"
    exit 0
fi

# hermes CLI 缺失时 mine.py 会优雅返回空结果，不会崩溃
if ! python3 pipeline/mine.py --after 7d --output-dir "$RUN_DIR"; then
    echo "[cron] WARN: MINE stage failed (non-zero exit)" >&2
    rm -rf "$RUN_DIR"
    exit 1
fi

TASKS="$RUN_DIR/tasks.json"
if [ ! -f "$TASKS" ]; then
    echo "[cron] No tasks — nothing to optimize"
    rm -rf "$RUN_DIR"
    exit 0
fi

TOTAL_CARDS=$(TASKS="$TASKS" python3 -c 'import json,os; d=json.load(open(os.environ["TASKS"])); print(d.get("total_cards", len(d.get("tasks", []))))' 2>/dev/null || echo "0")
# 去掉可能的非数字字符，兜底 0
if ! echo "$TOTAL_CARDS" | grep -qE '^[0-9]+$'; then
    TOTAL_CARDS=0
fi
if [ "$TOTAL_CARDS" -eq 0 ]; then
    echo "[cron] No tasks (total_cards=0) — nothing to optimize"
    rm -rf "$RUN_DIR"
    exit 0
fi
echo "[cron] MINE complete — $TOTAL_CARDS task card(s)"

# ── Stage 2: PROPOSE ───────────────────────────────────────────────────────
echo "[cron] === Stage 2: PROPOSE ==="
if [ ! -f pipeline/propose.py ]; then
    echo "[cron] SKIP: pipeline/propose.py not found — aborting" >&2
    rm -rf "$RUN_DIR"
    exit 0
fi

if ! python3 pipeline/propose.py --tasks "$TASKS" --output-dir "$RUN_DIR"; then
    echo "[cron] WARN: PROPOSE stage failed" >&2
    rm -rf "$RUN_DIR"
    exit 1
fi

DIFF="$RUN_DIR/candidate.diff"
PROPOSAL="$RUN_DIR/proposal.json"
if [ ! -f "$DIFF" ] || [ ! -f "$PROPOSAL" ]; then
    echo "[cron] PROPOSE did not produce output — skipping"
    if [ ! -f "$DIFF" ]; then echo "[cron]   missing: $DIFF" >&2; fi
    if [ ! -f "$PROPOSAL" ]; then echo "[cron]   missing: $PROPOSAL" >&2; fi
    rm -rf "$RUN_DIR"
    exit 0
fi
DIFF_LINES=$(wc -l < "$DIFF" 2>/dev/null | tr -d ' ' || echo "0")
echo "[cron] PROPOSE complete — $DIFF_LINES lines"

# ── Stage 3: VALIDATE ──────────────────────────────────────────────────────
echo "[cron] === Stage 3: VALIDATE ==="
if [ ! -f pipeline/validate.py ]; then
    echo "[cron] SKIP: pipeline/validate.py not found — aborting" >&2
    rm -rf "$RUN_DIR"
    exit 0
fi

if ! python3 pipeline/validate.py --tasks "$TASKS" --diff "$DIFF" --proposal "$PROPOSAL" --output-dir "$RUN_DIR"; then
    echo "[cron] WARN: VALIDATE stage failed" >&2
    rm -rf "$RUN_DIR"
    exit 1
fi

VALIDATION="$RUN_DIR/validation.json"
if [ ! -f "$VALIDATION" ]; then
    echo "[cron] VALIDATE did not produce output — skipping"
    rm -rf "$RUN_DIR"
    exit 0
fi
echo "[cron] VALIDATE complete"

# ── Stage 4: REVIEW stage ──────────────────────────────────────────────────
echo "[cron] === Stage 4: REVIEW stage ==="
if [ ! -f pipeline/review.py ]; then
    echo "[cron] SKIP: pipeline/review.py not found — aborting" >&2
    rm -rf "$RUN_DIR"
    exit 0
fi

if ! python3 pipeline/review.py stage --validation "$VALIDATION" --diff "$DIFF" --proposal "$PROPOSAL" --output-dir "$SCRIPT_DIR"; then
    echo "[cron] WARN: REVIEW stage failed" >&2
    rm -rf "$RUN_DIR"
    exit 1
fi
echo "[cron] REVIEW stage complete"

# ── Summary ────────────────────────────────────────────────────────────────
TOTAL=$(TASKS="$TASKS" python3 -c 'import json,os; d=json.load(open(os.environ["TASKS"])); print(d.get("total_cards", len(d.get("tasks", []))))' 2>/dev/null || echo "0")
GATE=$(VALIDATION="$VALIDATION" python3 -c 'import json,os; d=json.load(open(os.environ["VALIDATION"])); print("passed" if d.get("overall_passed") else "failed")' 2>/dev/null || echo "unknown")
DIFF_LINES=$(wc -l < "$DIFF" 2>/dev/null | tr -d ' ' || echo "0")

# 最近一次 staging / rejected 目录（stage 脚本已按 PASS/FAIL 路由）
LATEST_STAGING=$(ls -td staging/* 2>/dev/null | head -n 1 || true)
LATEST_REJECTED=$(ls -td rejected/* 2>/dev/null | head -n 1 || true)

echo "[cron] === Summary ==="
echo "[cron] Tasks: $TOTAL | Diff: ${DIFF_LINES} lines | Gate: $GATE"
if [ "$GATE" = "passed" ] && [ -n "$LATEST_STAGING" ]; then
    echo "[cron] Staging: $LATEST_STAGING"
    echo "[cron] Next: python3 pipeline/review.py apply --staging-dir $LATEST_STAGING"
elif [ "$GATE" = "failed" ] && [ -n "$LATEST_REJECTED" ]; then
    echo "[cron] Rejected: $LATEST_REJECTED (see rejected/rejected.jsonl)"
else
    # 兜底：直接提示 RUN_DIR 中的产物
    echo "[cron] Artifacts: $RUN_DIR"
fi
echo "[cron] Log: $RUN_DIR"
echo "[cron] Done."
