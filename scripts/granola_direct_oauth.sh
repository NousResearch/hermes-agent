#!/usr/bin/env bash
set -euo pipefail

cd /home/rj/.hermes/hermes-agent

export HERMES_HOME="${HERMES_HOME:-/home/rj/.hermes}"

if [[ -f /home/rj/.config/onecli/hermes-spark-proxy.env ]]; then
  set -a
  # shellcheck disable=SC1091
  . /home/rj/.config/onecli/hermes-spark-proxy.env
  set +a
fi

exec npx -y mcp-remote https://mcp.granola.ai/mcp
