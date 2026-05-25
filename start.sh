#!/bin/bash
set -euo pipefail

# Railway service variables can override PATH at runtime.
# Re-add standard system dirs so mkdir/cp/touch/rm/python are always available.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

export HOME="${HOME:-/data}"
export HERMES_HOME="${HERMES_HOME:-/data/.hermes}"

mkdir -p \
  "${HERMES_HOME}/cron" \
  "${HERMES_HOME}/sessions" \
  "${HERMES_HOME}/logs" \
  "${HERMES_HOME}/memories" \
  "${HERMES_HOME}/skills" \
  "${HERMES_HOME}/pairing" \
  "${HERMES_HOME}/hooks" \
  "${HERMES_HOME}/image_cache" \
  "${HERMES_HOME}/audio_cache" \
  "${HERMES_HOME}/workspace"

if [ ! -f "${HERMES_HOME}/config.yaml" ] && [ -f "/app/cli-config.yaml.example" ]; then
  cp /app/cli-config.yaml.example "${HERMES_HOME}/config.yaml"
fi

if [ ! -f "${HERMES_HOME}/.env" ]; then
  touch "${HERMES_HOME}/.env"
fi

exec python /app/server.py
