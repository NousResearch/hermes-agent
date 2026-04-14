#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFERRED_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROOT="${OPENCLAW_ROOT:-${INFERRED_ROOT:-/home/yeqiuqiu/clawd-architect}}"
CANARY_SCRIPT="$ROOT/ops/obsidian/hourly_canary.sh"
LOCK_WRAP="$ROOT/ops/openclaw/cron_wrappers/openclaw_cron_lock_timeout.sh"
HASH_WRAP="$ROOT/ops/openclaw/cron_wrappers/openclaw_cron_state_hash.sh"
EVENT_ROUTER="$ROOT/ops/openclaw/event_router.sh"
BLOCKER_ROUTING_LIB="$ROOT/ops/openclaw/lib/blocker_routing.sh"

TELEMETRY_DIR="$ROOT/ops/telemetry/textfile"
STATE_DIR="$ROOT/state/cron_watchdog"

VAULT_PATH="${OPENCLAW_VAULT_PATH:-$ROOT/obsvault_yq_terminal}"
CANARY_STATE_FILE="${OPENCLAW_HOURLY_CANARY_STATE_FILE:-/tmp/obsidian_hourly_canary_state_canary.json}"
CANARY_REMINDER_SECS="${OPENCLAW_HOURLY_CANARY_REMINDER_SECS:-21600}"
# Keep default below cron agent timeout (600s) so wrapper emits structured BLOCKER
# instead of letting outer cron hard-timeout.
TIMEOUT_SEC="${OPENCLAW_HOURLY_CANARY_TIMEOUT_SEC:-480}"
GRACE_SEC="${OPENCLAW_HOURLY_CANARY_TIMEOUT_GRACE_SEC:-30}"
EVENT_COOLDOWN_SEC="${OPENCLAW_OBSIDIAN_CANARY_EVENT_COOLDOWN_SEC:-1800}"

mkdir -p "$TELEMETRY_DIR" "$STATE_DIR"

# shellcheck source=ops/openclaw/lib/blocker_routing.sh
source "$BLOCKER_ROUTING_LIB"

OPENCLAW_BLOCKER_EVENT_ROUTER="$EVENT_ROUTER"
OPENCLAW_BLOCKER_SOURCE="watchdog.obsidian_hourly_canary"
OPENCLAW_BLOCKER_COOLDOWN_SEC="$EVENT_COOLDOWN_SEC"
OPENCLAW_BLOCKER_DEFAULT_EVIDENCE_REF="$CANARY_STATE_FILE"

if [[ ! -x "$LOCK_WRAP" || ! -x "$HASH_WRAP" ]]; then
  openclaw_watchdog_route_blocker "wrapper_dependency_missing" "task=obsidian_hourly_canary; reason=wrapper_dependency_missing"
  exit 0
fi

if [[ ! -x "$CANARY_SCRIPT" ]]; then
  openclaw_watchdog_route_blocker "canary_script_missing" "task=obsidian_hourly_canary; reason=missing_canary_script"
  exit 0
fi

vault_sig="$(python3 - "$VAULT_PATH" <<'PY'
import hashlib, pathlib, sys
vault = pathlib.Path(sys.argv[1])
if not vault.exists() or not vault.is_dir():
    print('vault:missing')
    raise SystemExit(0)
rows = []
for p in vault.rglob('*.md'):
    try:
        st = p.stat()
    except OSError:
        continue
    rows.append(f"{p.relative_to(vault)}|{int(st.st_mtime)}|{st.st_size}")
rows.sort()
joined = "\n".join(rows).encode('utf-8', errors='ignore')
print(f"vault_hash={hashlib.sha256(joined).hexdigest()}|files={len(rows)}")
PY
)"

"$HASH_WRAP" \
  --state-file "$STATE_DIR/obsidian_hourly_canary_input_hash.json" \
  --input "$vault_sig" >/dev/null

set +e
output="$($LOCK_WRAP \
  --lock-name "obsidian_hourly_canary" \
  --lock-dir "$STATE_DIR/locks" \
  --timeout-sec "$TIMEOUT_SEC" \
  --grace-sec "$GRACE_SEC" \
  --busy-exit-code 75 \
  --emit-blocker \
  --soft-timeout \
  -- env \
      OPENCLAW_VAULT_PATH="$VAULT_PATH" \
      OPENCLAW_HOURLY_CANARY_STATE_FILE="$CANARY_STATE_FILE" \
      OPENCLAW_HOURLY_CANARY_REMINDER_SECS="$CANARY_REMINDER_SECS" \
      bash "$CANARY_SCRIPT" 2>&1)"
rc=$?
set -e

if [[ -n "$output" && "${OPENCLAW_WATCHDOG_VERBOSE_STDOUT:-0}" == "1" ]]; then
  filtered_output="$(printf '%s\n' "$output" | sed '/^BLOCKER:/d;/^BLOCKER_JSON:/d')"
  if [[ -n "$filtered_output" ]]; then
    # Keep raw child output off stdout so cron/chat first-line relays cannot leak internals.
    printf '%s\n' "$filtered_output" >&2
  fi
fi

if [[ "$rc" -eq 75 ]]; then
  # lock held by another run; silent skip
  exit 0
fi

first_line="$(printf '%s\n' "$output" | awk 'NF{print; exit}')"

if [[ "$rc" -ne 0 ]]; then
  if [[ "$first_line" != BLOCKER:* && "$first_line" != BLOCKER_JSON:* ]]; then
    openclaw_watchdog_route_blocker "wrapper_exit" "task=obsidian_hourly_canary; reason=wrapper_exit_${rc}" "$CANARY_STATE_FILE"
  else
    openclaw_watchdog_route_blocker "task_blocker" "$(openclaw_blocker_summary_from_line "$first_line")" "$CANARY_STATE_FILE"
  fi
  exit 0
fi

if [[ "$first_line" == BLOCKER:* || "$first_line" == BLOCKER_JSON:* ]]; then
  openclaw_watchdog_route_blocker "task_blocker" "$(openclaw_blocker_summary_from_line "$first_line")" "$CANARY_STATE_FILE"
  exit 0
fi

now_epoch="$(date +%s)"
printf 'openclaw_hourly_canary_last_success_epoch %s\n' "$now_epoch" > "$TELEMETRY_DIR/openclaw_hourly_canary_last_success_epoch.prom"
exit 0
