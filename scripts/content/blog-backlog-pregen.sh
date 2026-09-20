#!/bin/bash
# blog-backlog-pregen.sh — backlog pre-generation (P13: dry-run + no background detach).
# Generates ONE ready-to-approve post per run (rotating ai/pm/builder), pulling
# from the backlog queues. Runs synchronously under the cron agent.
# BLOG_DAILY_DRY_RUN=1 short-circuits before any Python invocation.
# One post = ~3 Codex images, well under the usage cap.
# Posts accrue as approved:false drafts + approval cards.
#
# Quota gate (2026-08-16): when the LLM provider (ollama weekly usage limit)
# or Codex image cap is exhausted, the run records the reset time in
# blog_topics/quota_state.json and exits 0 (silent — no alert spam every 12h).
# Subsequent runs skip while the block is active, then automatically re-do the
# backlog once the reset time passes. Failed topics stay in the queue (never
# recorded), so the re-run picks them up naturally.
set -euo pipefail

HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
# Cron's minimal PATH does not include npm's user-global bin directory, where
# the ChatGPT-authenticated Codex CLI is installed. Keep this idempotent and
# prepend only when the directory exists.
if [[ -d "$HOME/.npm-global/bin" ]]; then
  export PATH="$HOME/.npm-global/bin:$PATH"
fi
ROOT="${BLOG_CONTENT_ROOT:-/home/kensei/repos/KenseiAgent/content_engine}"
LOG_DIR="$ROOT/output/logs"
QUOTA_STATE="$ROOT/blog_topics/quota_state.json"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/blog-backlog-pregen-$(date +%Y%m%d-%H%M%S).log"

if [[ "${BLOG_DAILY_DRY_RUN:-0}" == "1" ]]; then
  echo "dry-run: would launch backlog pregen (synchronous) -> $LOG"
  exit 0
fi

# Legacy noop kept for backward compatibility with existing env wiring.
if [[ "${BLOG_DAILY_NOOP:-}" == "1" ]]; then
  echo "noop: would launch backlog pregen -> $LOG"
  exit 0
fi

# ── Pre-run quota gate ─────────────────────────────────────────────────────
# If a quota block was recorded and its reset time has not yet passed, skip
# silently. The cron ticker stays armed; the first run after reset proceeds.
# NOTE: deliberately AFTER the dry-run/noop checks so P13 hermetic tests are
# never short-circuited by live provider state.
if [[ -f "$QUOTA_STATE" ]]; then
  reset_epoch=$(python3 -c "
import json, sys, time
from datetime import datetime
try:
    d = json.load(open('$QUOTA_STATE'))
    dt = datetime.fromisoformat(d['resets_at'].rstrip('.').replace('Z', '+00:00'))
    print(int(dt.timestamp()))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
  now_epoch=$(date +%s)
  if [[ -n "$reset_epoch" && "$reset_epoch" -gt "$now_epoch" ]]; then
    echo "[$(date -Is)] quota block active until $(date -d @"$reset_epoch" -Is 2>/dev/null || echo "$reset_epoch"); skipping pregen (silent)"
    exit 0
  fi
  # Reset time passed — clear the block and run normally.
  rm -f "$QUOTA_STATE"
  echo "[$(date -Is)] quota block expired; clearing quota_state and resuming pregen"
fi

rc=0
(
  cd "$ROOT"
  set -a
  . "${HERMES_HOME_DIR}/.env" 2>/dev/null || true
  set +a
  echo "[$(date -Is)] starting backlog pregen"
  # Pin the fleet venv interpreter explicitly. Bare `python3` resolves via PATH,
  # which differs between the gateway service context (venv-first) and manual/CLI
  # runs (system python3, no hermes_constants on the editable path) — the exact
  # split that produced ModuleNotFoundError on 2026-09-15 22:35.
  # `python -m blog.backlog_pregen` unconditionally inserts cwd ($ROOT, i.e.
  # content_engine/) at sys.path[0] *ahead of PYTHONPATH*, so content_engine's
  # own `tools/` package (no `threat_patterns` submodule) shadowed the repo-root
  # `tools/` package that agent.prompt_builder needs — the real cause behind the
  # 2026-09-16 02:56 and 2026-09-20 09:55 "No module named 'tools.threat_patterns'"
  # failures (PYTHONPATH ordering can't fix this; `-m`'s cwd insertion always wins).
  # runpy.run_module() does not do that implicit insertion, so pin the repo root
  # first via explicit sys.path surgery before ever running the module.
  PYTHONPATH=/home/kensei/repos/KenseiAgent:. /home/kensei/repos/KenseiAgent/.venv/bin/python -c "
import sys
sys.path.insert(0, '/home/kensei/repos/KenseiAgent')
import runpy
runpy.run_module('blog.backlog_pregen', run_name='__main__')
"
  rc=$?
  echo "[$(date -Is)] finished backlog pregen rc=$rc"
  exit "$rc"
) >>"$LOG" 2>&1 || rc=$?

# ── Post-run quota detection ────────────────────────────────────────────────
# If the run failed because of a provider usage cap, record the reset time and
# exit 0. A capped run is not a pipeline defect — the topics are still queued
# and will be picked up after the window refreshes. Alerting on it every 12h
# is noise, not signal.
if [[ "$rc" -ne 0 ]]; then
  # Guarded: under set -euo pipefail an unmatched grep exits 1 and would kill
  # the script BEFORE the fallback grep below could match "usage limit".
  reset_ts=$(grep -oE 'limit resets at [0-9TZ:.\-]+' "$LOG" 2>/dev/null | head -1 | sed 's/limit resets at //' || true)
  # Codex CLI format: "try again at Aug 24th, 2026 7:45 PM" — parse to ISO.
  if [[ -z "$reset_ts" ]]; then
    human=$(grep -oE 'try again at [A-Za-z]+ [0-9]{1,2}(st|nd|rd|th),? [0-9]{4} [0-9]{1,2}:[0-9]{2} (AM|PM)' "$LOG" 2>/dev/null | head -1 | sed 's/try again at //' || true)
    if [[ -n "$human" ]]; then
      clean=$(printf '%s' "$human" | sed -E 's/([0-9]{1,2})(st|nd|rd|th)/\1/; s/,//')
      parsed=$(date -d "$clean" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || true)
      [[ -n "$parsed" ]] && reset_ts="$parsed"
    fi
  fi
  if [[ -n "$reset_ts" ]] || grep -qE 'usage limit|usage cap|Codex usage cap|weekly usage' "$LOG" 2>/dev/null; then
    if [[ -z "$reset_ts" ]]; then
      # No explicit reset timestamp: default to 12h (next scheduled tick).
      reset_ts=$(date -u -d '+12 hours' +%Y-%m-%dT%H:%M:%SZ)
    fi
    echo "{\"blocked_at\": \"$(date -Is)\", \"resets_at\": \"$reset_ts\", \"reason\": \"provider usage cap\", \"log\": \"$LOG\"}" > "$QUOTA_STATE"
    echo "[$(date -Is)] quota cap detected; deferring until $reset_ts (silent exit 0)"
    exit 0
  fi
fi

# Synchronous — silent on success (no Discord delivery).
# NOTE: prior version always exited 0, masking the pipeline's rc=1 from the
# scheduler (last_status stayed 'ok' even when the run ended failed_images),
# so system-health could never report this job's failures. Propagate the real
# rc so a failed run surfaces as last_status=error and the alert chain works.
if [[ "$rc" -ne 0 ]]; then
  echo "[$(date -Is)] backlog pregen FAILED rc=$rc (log: $LOG)" >&2
fi
exit "$rc"
