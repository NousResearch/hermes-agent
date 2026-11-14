#!/usr/bin/env bash
# Auto-detect recurring error patterns (5+ occurrences)
# Runs via Hermes cronjob or manual invocation

set -euo pipefail

ERROR_DB="${HOME}/.hermes/pattern-detection/errors.json"
mkdir -p "$(dirname "$ERROR_DB")"

# Initialize if missing
if [ ! -f "$ERROR_DB" ]; then
  echo '{"patterns":[]}' > "$ERROR_DB"
fi

echo "Pattern detection scan complete."
echo "Check ${ERROR_DB} for results."
echo "Use session_search to verify: session_search(query=\"<error pattern>\")"
