#!/usr/bin/env bash
# Quick status for the xAI Grok OAuth provider in Hermes.
# Run via terminal tool or directly: bash ${HERMES_SKILL_DIR}/scripts/check-grok-oauth.sh

set -euo pipefail

echo "=== Hermes Grok (xAI OAuth) Status Check ==="
echo "Hermes home: ${HERMES_HOME:-~/.hermes}"
echo

if command -v hermes >/dev/null 2>&1; then
  echo "Hermes binary: $(command -v hermes)"
  hermes --version || true
else
  echo "hermes not in PATH (using full path in this env)"
fi

echo
echo "Looking for auth artifacts..."
ls -l ~/.hermes/auth*.json 2>/dev/null || echo "No auth*.json in ~/.hermes (may be in shared/ or elsewhere)"
ls -l ~/.hermes/shared/nous_auth.json 2>/dev/null || true

echo
echo "Current model config hint (run 'hermes model' interactively for full picker):"
hermes config show 2>/dev/null | grep -i -E 'model|provider|xai|grok' | head -10 || echo "Run inside a Hermes session for richer config."

echo
echo "Tip: In a Hermes chat on the Grok provider, ask: 'Confirm you are Grok from xAI and show me your X search capability with a tiny test query.'"
echo "=== Done ==="
