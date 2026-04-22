#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_FILE="${HERMES_ENV_FILE:-$HOME/.hermes/.env}"
X_ACCESS_STATE_FILE="${HERMES_X_ACCESS_STATE_FILE:-$HOME/.hermes/state/x-access.json}"

required_files=(
  "$ROOT_DIR/starter-kits/agentic-cron-orchestration-kit/launch/launch-thread.md"
  "$ROOT_DIR/starter-kits/agentic-cron-orchestration-kit/launch/publish-runbook.md"
  "$ROOT_DIR/starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md"
  "$ROOT_DIR/starter-kits/agent-launch-closeout-kit/launch-execution-log.md"
)

auth_vars=(
  X_API_KEY
  X_API_SECRET
  X_BEARER_TOKEN
  X_ACCESS_TOKEN
  X_ACCESS_TOKEN_SECRET
)

missing_files=()
for file in "${required_files[@]}"; do
  [[ -f "$file" ]] || missing_files+=("${file#$ROOT_DIR/}")
done

missing_auth=()
for var_name in "${auth_vars[@]}"; do
  [[ -n "${!var_name:-}" ]] || missing_auth+=("$var_name")
done

env_keys_present=0
if [[ -f "$ENV_FILE" ]]; then
  for var_name in "${auth_vars[@]}"; do
    if grep -Eq "^(export[[:space:]]+)?${var_name}=" "$ENV_FILE"; then
      env_keys_present=$((env_keys_present + 1))
    fi
  done
fi

browser_ready=0
browser_handle=""
browser_status="missing"
if [[ -f "$X_ACCESS_STATE_FILE" ]]; then
  browser_status="$(python3 - <<'PY' "$X_ACCESS_STATE_FILE"
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    print(data.get("status", "unknown"))
except Exception:
    print("unknown")
PY
)"
  if grep -Eq '"status"[[:space:]]*:[[:space:]]*"ready"' "$X_ACCESS_STATE_FILE" \
    && grep -Eq '"mode"[[:space:]]*:[[:space:]]*"browser-session"' "$X_ACCESS_STATE_FILE"; then
    browser_ready=1
    browser_handle="$(python3 - <<'PY' "$X_ACCESS_STATE_FILE"
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    print(data.get("handle", ""))
except Exception:
    print("")
PY
)"
  fi
fi

if (( ${#missing_files[@]} > 0 )); then
  printf 'Publish preflight blocked: missing required files\n'
  printf ' - %s\n' "${missing_files[@]}"
  exit 2
fi

if (( ${#missing_auth[@]} > 0 )) && (( browser_ready == 0 )); then
  printf 'Publish preflight blocked: missing X API auth env vars and no browser-session publish path is marked ready\n'
  printf ' - %s\n' "${missing_auth[@]}"
  if [[ -f "$ENV_FILE" ]]; then
    printf 'Env file checked: %s\n' "$ENV_FILE"
    printf 'X auth keys present in env file: %d/%d\n' "$env_keys_present" "${#auth_vars[@]}"
  else
    printf 'Env file checked: %s (missing)\n' "$ENV_FILE"
  fi
  printf 'Browser-session state checked: %s (status: %s)\n' "$X_ACCESS_STATE_FILE" "$browser_status"
  exit 3
fi

printf 'Publish preflight OK\n'
printf 'Required files present: %d\n' "${#required_files[@]}"
if (( ${#missing_auth[@]} == 0 )); then
  printf 'Publish path: X API credentials available (%d vars)\n' "${#auth_vars[@]}"
elif (( browser_ready == 1 )); then
  printf 'Publish path: browser-session marker ready'
  if [[ -n "$browser_handle" ]]; then
    printf ' (%s)' "$browser_handle"
  fi
  printf '\n'
  printf 'Live browser sign-in still must be verified in the actual publish session before claiming publish is unblocked\n'
  printf 'X API env vars still missing: %d/%d\n' "${#missing_auth[@]}" "${#auth_vars[@]}"
fi
