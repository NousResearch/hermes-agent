#!/usr/bin/env bash
# KENSEI Content Engine v2.3 — Daily Pipeline Runner
# Stage 1: generates LLM text drafts for 5 brands into content_engine.db.
# Stage 2 (separate cron): digest generation + approval delivery.
#
# Output: summary line. [SILENT] when zero drafts generated (handled by engine).
#
# Fixed 2026-07-14:
#   - Telegram removed.
#   - Unsupported --max-per-brand and --html-out flags removed.
#   - Uses only supported content_engine.py stage1 flags.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "$(date -Iseconds) — Content Engine v2.3: generating LLM drafts..."
python3 content_engine.py stage1 \
    --brand matchdaymaestro plenishd sahil_twitter sahil_linkedin coachos
