#!/usr/bin/env bash
set -euo pipefail

load_bundle_env() {
  local bundle_root="${MEMD_BUNDLE_ROOT:-.memd}"
  local backend_env_file="$bundle_root/backend.env"
  local env_file="$bundle_root/env"
  if [ -f "$backend_env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$backend_env_file"
    set +a
  fi
  if [ -f "$env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
}

load_bundle_env

MEMD_BASE_URL="${MEMD_BASE_URL:-http://100.104.154.24:8787}"
MEMD_PROJECT="${MEMD_PROJECT:?MEMD_PROJECT is required}"
MEMD_NAMESPACE="${MEMD_NAMESPACE:-}"
MEMD_AGENT="${MEMD_AGENT:?MEMD_AGENT is required}"
MEMD_ROUTE="${MEMD_ROUTE:-auto}"
MEMD_INTENT="${MEMD_INTENT:-current_task}"
MEMD_LIMIT="${MEMD_LIMIT:-8}"
MEMD_REHYDRATION_LIMIT="${MEMD_REHYDRATION_LIMIT:-4}"

args=(
  --base-url "$MEMD_BASE_URL"
  wake
  --output "${MEMD_BUNDLE_ROOT:-.memd}"
  --project "$MEMD_PROJECT"
  --agent "$MEMD_AGENT"
  --route "$MEMD_ROUTE"
  --intent "$MEMD_INTENT"
  --limit "$MEMD_LIMIT"
  --rehydration-limit "$MEMD_REHYDRATION_LIMIT"
  --write
)

if [ -n "$MEMD_NAMESPACE" ]; then
  args+=(--namespace "$MEMD_NAMESPACE")
fi
if [ -n "${MEMD_WORKSPACE:-}" ]; then
  args+=(--workspace "$MEMD_WORKSPACE")
fi
if [ -n "${MEMD_VISIBILITY:-}" ]; then
  args+=(--visibility "$MEMD_VISIBILITY")
fi

if memd "${args[@]}"; then
  exit 0
fi

bundle_root="${MEMD_BUNDLE_ROOT:-.memd}"
if [ -f "$bundle_root/wake.md" ]; then
  cat "$bundle_root/wake.md"
  exit 0
fi
