#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
GUARD_SCRIPT="$ROOT/ops/openclaw/no_llm_watchdog_cron_authority_guard.sh"
DRY_RUN=0
EXPECTED_NAMES_CSV="${OPENCLAW_NO_LLM_AUTHORITY_EXPECTED_NAMES:-continuity:backup-checkpoint-90m,continuity:stale-progress-45m,web-capture-scheduler-governance-watchdog,obsidian:hourly-canary,obsidian:vault-tick-hourly}"
MODEL_OVERRIDE="${OPENCLAW_NO_LLM_AUTHORITY_MODEL:-openai-codex/gpt-5.3-codex-spark}"
THINKING_OVERRIDE="${OPENCLAW_NO_LLM_AUTHORITY_THINKING:-minimal}"
TIMEOUT_SECONDS="${OPENCLAW_NO_LLM_AUTHORITY_TIMEOUT_SECONDS:-300}"
CORE_ROADMAP_REFILL_TIMEOUT_SECONDS="${OPENCLAW_CORE_ROADMAP_REFILL_TIMEOUT_SECONDS:-900}"

usage() {
  cat <<'EOF'
Usage: harden_no_llm_watchdog_cron_authority.sh [options]

Rewire recurring watchdog/canary/checkpoint/scheduler-governance cron jobs so
authority stays deterministic (NO_LLM), while model wrappers are reduced to
non-authoritative command execution + NO_REPLY only.

Options:
  --expected-names <csv>   Override expected enabled authority job names
  --dry-run                Show planned edits only
  -h, --help
EOF
}

command_for_name() {
  local name="${1:-}"
  case "$name" in
    continuity:backup-checkpoint-90m|continuity:stale-progress-45m)
      printf '%s\n' "$ROOT/ops/openclaw/contract_no_nudge_continuity_watchdog.sh"
      ;;
    core-roadmap-executor-idle-watchdog)
      printf '%s\n' "$ROOT/ops/openclaw/contract_core_roadmap_floor_refill_watchdog.sh"
      ;;
    web-capture-scheduler-governance-watchdog|web-capture-scheduler-governance)
      printf '%s\n' "$ROOT/ops/openclaw/contract_web_capture_scheduler_governance_watchdog.sh"
      ;;
    obsidian:hourly-canary|obsidian-vault-tick-hourly-canary)
      printf '%s\n' "$ROOT/ops/openclaw/contract_obsidian_hourly_canary_watchdog.sh"
      ;;
    obsidian:vault-tick-hourly)
      printf '%s\n' "$ROOT/ops/openclaw/contract_obsidian_vault_tick_watchdog.sh"
      ;;
    context-watch\ 90%*)
      printf '%s\n' "$ROOT/ops/openclaw/contract_context_runtime_watchdog.sh"
      ;;
    *)
      printf '\n'
      ;;
  esac
}

timeout_for_name() {
  local name="${1:-}"
  case "$name" in
    core-roadmap-executor-idle-watchdog)
      printf '%s\n' "$CORE_ROADMAP_REFILL_TIMEOUT_SECONDS"
      ;;
    *)
      printf '%s\n' "$TIMEOUT_SECONDS"
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --expected-names)
      EXPECTED_NAMES_CSV="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$GUARD_SCRIPT" ]]; then
  echo "BLOCKER: missing guard script: $GUARD_SCRIPT"
  exit 1
fi

if ! [[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "BLOCKER: invalid OPENCLAW_NO_LLM_AUTHORITY_TIMEOUT_SECONDS=$TIMEOUT_SECONDS"
  exit 1
fi
if ! [[ "$CORE_ROADMAP_REFILL_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "BLOCKER: invalid OPENCLAW_CORE_ROADMAP_REFILL_TIMEOUT_SECONDS=$CORE_ROADMAP_REFILL_TIMEOUT_SECONDS"
  exit 1
fi

cron_json="$(openclaw cron list --json)"

EXPECTED_NAMES_CSV="$(python3 - "$cron_json" "$EXPECTED_NAMES_CSV" <<'PY'
import json
import sys

cron_raw = str(sys.argv[1] or "")
expected_csv = str(sys.argv[2] or "")
expected = []
for token in expected_csv.split(","):
    name = token.strip()
    if not name or name in expected:
        continue
    expected.append(name)

try:
    obj = json.loads(cron_raw)
except Exception:
    obj = {}

jobs = obj.get("jobs") if isinstance(obj, dict) else []
if not isinstance(jobs, list):
    jobs = []

auto_name = "core-roadmap-executor-idle-watchdog"
auto_enabled = any(
    isinstance(job, dict)
    and bool(job.get("enabled", False))
    and str(job.get("name") or "").strip() == auto_name
    for job in jobs
)
if auto_enabled and auto_name not in expected:
    expected.append(auto_name)

print(",".join(expected))
PY
)"

mapfile -t target_rows < <(python3 - "$cron_json" "$EXPECTED_NAMES_CSV" <<'PY'
import json
import sys

obj = json.loads(sys.argv[1])
expected_csv = str(sys.argv[2] or "")
expected = []
for token in expected_csv.split(","):
    name = token.strip()
    if not name or name in expected:
        continue
    expected.append(name)

jobs = obj.get("jobs") or []
for name in expected:
    matches = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        if not bool(job.get("enabled", False)):
            continue
        if str(job.get("name") or "").strip() != name:
            continue
        job_id = str(job.get("id") or "").strip()
        if not job_id:
            continue
        matches.append(job_id)

    if len(matches) == 1:
        print(f"{matches[0]}\t{name}")
PY
)

readarray -t expected_names < <(python3 - "$EXPECTED_NAMES_CSV" <<'PY'
import sys
rows = []
for token in str(sys.argv[1] or "").split(','):
    name = token.strip()
    if not name or name in rows:
        continue
    rows.append(name)
for row in rows:
    print(row)
PY
)

if [[ "${#target_rows[@]}" -ne "${#expected_names[@]}" ]]; then
  echo "BLOCKER: expected enabled authority jobs were not fully discovered; expected=${#expected_names[@]}; found=${#target_rows[@]}"
  exit 1
fi

updated=0
for row in "${target_rows[@]}"; do
  id="${row%%$'\t'*}"
  name="${row#*$'\t'}"

  if [[ -z "$id" || -z "$name" ]]; then
    continue
  fi

  command_path="$(command_for_name "$name")"
  if [[ -z "$command_path" ]]; then
    echo "BLOCKER: no deterministic command mapping for cron job name=$name"
    exit 1
  fi
  if [[ ! -x "$command_path" ]]; then
    echo "BLOCKER: missing executable deterministic contract command for $name: $command_path"
    exit 1
  fi

  timeout_seconds_for_job="$(timeout_for_name "$name")"

  read -r -d '' MESSAGE <<EOF || true
Run deterministic recurring control authority rail.

Execute exactly:
$command_path

This command is the authority path. It performs deterministic health/blocker evaluation and routes blocker events directly via event_router.
Do not forward, reinterpret, or summarize command output for authority decisions.
After execution, reply exactly: NO_REPLY.
EOF

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "PROGRESS: would harden id=$id name=$name command=$command_path timeout_seconds=$timeout_seconds_for_job"
    continue
  fi

  openclaw cron edit "$id" \
    --session isolated \
    --message "$MESSAGE" \
    --model "$MODEL_OVERRIDE" \
    --thinking "$THINKING_OVERRIDE" \
    --timeout-seconds "$timeout_seconds_for_job" \
    --light-context \
    --no-deliver >/tmp/harden_no_llm_watchdog_cron_authority_edit.out

  updated=$((updated + 1))
  echo "PROGRESS: hardened id=$id name=$name timeout_seconds=$timeout_seconds_for_job"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "READY: dry-run completed; targets=${#target_rows[@]}"
  exit 0
fi

if ! "$GUARD_SCRIPT" --strict --expected-names "$EXPECTED_NAMES_CSV" >/tmp/harden_no_llm_watchdog_cron_authority_guard.out 2>/tmp/harden_no_llm_watchdog_cron_authority_guard.err; then
  cat /tmp/harden_no_llm_watchdog_cron_authority_guard.out 2>/dev/null || true
  cat /tmp/harden_no_llm_watchdog_cron_authority_guard.err 2>/dev/null || true
  exit 1
fi

echo "READY: hardened recurring control jobs to deterministic no-llm authority mode; updated=${updated}"
exit 0
