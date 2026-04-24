#!/usr/bin/env bash
set -euo pipefail

PRIMARY_HOST="${HERMES_PRIMARY_HOST:-soichiyo@100.76.4.115}"
PRIMARY_AGENT_DIR="${HERMES_PRIMARY_AGENT_DIR:-/Users/soichiyo/.hermes/hermes-agent}"

if [[ $# -eq 0 ]]; then
  set -- chat
fi

quoted_args=()
for arg in "$@"; do
  quoted_args+=("$(printf '%q' "$arg")")
done

remote_cmd="cd $(printf '%q' "$PRIMARY_AGENT_DIR") && source venv/bin/activate && hermes ${quoted_args[*]}"

exec ssh -t "$PRIMARY_HOST" "$remote_cmd"
