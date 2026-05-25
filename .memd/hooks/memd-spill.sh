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

exec memd --base-url "$MEMD_BASE_URL" hook spill "$@"
