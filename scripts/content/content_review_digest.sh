#!/usr/bin/env bash
# Daily content review digest: generate cheap on-brand images for recent
# text+image drafts and deliver the batch to Discord, grouped by brand.
# Runs no_agent (the script IS the job). Delivery happens inside the Python
# (discord_digest), so the cron deliver target is 'local'.
set -euo pipefail

# P13: disabled-staging guard — exit early when cron is disabled
if [ "${DRY_RUN:-0}" = "1" ]; then echo "[DRY_RUN] $(basename "$0")"; exit 0; fi

HERMES_HOME="${HERMES_HOME:-/home/kensei/.hermes}"
REPO="${REPO:-/home/kensei/repos/KenseiAgent}"

CE="${REPO}/content_engine"
cd "$CE"

# The gateway env does not carry FAL_KEY / DISCORD_BOT_TOKEN — source them.
set -a
# shellcheck disable=SC1091
source "${HERMES_HOME}/.env"
set +a

export PYTHONPATH="${CE}:${PYTHONPATH:-}"

# Wrap in timeout to prevent FAL/Discord hangs from blocking the cron slot
# 300s matches the cron-level timeout; FAL image generation can take 90-300s
# when the API is slow (not locked — locked returns 403 instantly)
timeout 600 "${REPO}/.venv/bin/python" "${CE}/content_engine.py" review-digest --since-minutes 75 --max 3
