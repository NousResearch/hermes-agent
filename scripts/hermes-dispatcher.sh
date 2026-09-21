#!/usr/bin/env bash
# Profile-aware Hermes CLI dispatcher.
# Routes `hermes` commands to the correct Python venv based on active profile.
# NEVER gets overwritten by pip installs because it lives here as a standalone script.

set -euo pipefail

# Map profile → venv bin/hermes
declare -A PROFILE_MAP
PROFILE_MAP[default]="/home/kensei/repos/KenseiAgent/.venv/bin/hermes"
PROFILE_MAP[kensei]="/home/kensei/repos/KenseiAgent/.venv/bin/hermes"
PROFILE_MAP[moss]="/home/kensei/repos/hermes-agent-vanilla/.venv/bin/hermes"
PROFILE_MAP[upstream]="/home/kensei/repos/hermes-agent-vanilla/.venv/bin/hermes"

# Detect active profile
PROFILE="${HERMES_PROFILE:-}"
if [[ -z "$PROFILE" && -f /home/kensei/.hermes/profile ]]; then
    PROFILE=$(cat /home/kensei/.hermes/profile)
fi
if [[ -z "$PROFILE" ]]; then
    PROFILE="default"
fi

# Resolve target
TARGET="${PROFILE_MAP[$PROFILE]:-}"
if [[ -z "$TARGET" ]]; then
    echo "error: unknown profile '$PROFILE'" >&2
    echo "known profiles: ${!PROFILE_MAP[*]}" >&2
    exit 1
fi
if [[ ! -x "$TARGET" ]]; then
    echo "error: venv not found for profile '$PROFILE': $TARGET" >&2
    exit 1
fi

# Exec the correct hermes
exec "$TARGET" "$@"
