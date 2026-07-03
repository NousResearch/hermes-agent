#!/usr/bin/env bash
# End-to-end local demo: provision two isolated employees and chat as both.
# Each gets their own HERMES_HOME + own Hermes agent, woken on demand.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=scripts/demo.config.yaml
export ORCHARD_LLM_API_KEY="$(cat "$HOME/.claude/.env_deepseek")"
# Make sure node/ripgrep/etc. that Hermes may need are on PATH for the workers.
export PATH="$HOME/.local/bin:/Users/stanislav/hermes-agent/.venv/bin:$PATH"

echo "== provisioning two employees =="
.venv/bin/orchard --config "$CFG" provision alice --mm-user alice --name "Alice Ivanova" || true
.venv/bin/orchard --config "$CFG" provision bob   --mm-user bob   --name "Bob Petrov"   || true

echo "== registry =="
.venv/bin/orchard --config "$CFG" list

echo "== isolation check: each home is 0700 and separate =="
ls -ld data/employees/alice data/employees/bob
echo "alice .env perms:"; stat -f "%Sp %N" data/employees/alice/.env

echo "== chatting as alice then bob (real DeepSeek calls) =="
printf '%s\n' \
  "alice: In one short sentence, what is a VPN?" \
  "bob: Reply with exactly the word: PONG" \
  | .venv/bin/orchard --config "$CFG" serve --ingress cli
