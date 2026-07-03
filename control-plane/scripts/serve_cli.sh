#!/usr/bin/env bash
# Interactive Hermes through the control-plane (CLI ingress). Unprefixed input is
# treated as the given employee. Secrets are re-read per turn, so a token entered
# mid-session (via the link the agent hands you) takes effect on the next message.
set -euo pipefail
cd "$(dirname "$0")/.."
export ORCHARD_LLM_API_KEY="$(cat "$HOME/.claude/.env_deepseek")"
export PATH="$HOME/.local/bin:/Users/stanislav/hermes-agent/.venv/bin:$PATH"
exec .venv/bin/orchard --config scripts/demo.config.yaml serve --ingress cli --as "${1:-alice}"
