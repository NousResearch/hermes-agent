#!/usr/bin/env bash
# Run a known prompt against the gemini35 test profile and emit a JSON
# baseline of state.db's usage + cost row. Intended for manual capture
# before the PR-1 refactor; the captured file becomes the regression
# target for tests/integration/test_prompt_cache_baseline.py.
#
# Usage:
#   scripts/baseline_prompt_cache.sh                # write to stdout
#   scripts/baseline_prompt_cache.sh > tests/fixtures/prompt_cache_baseline.json
#
# Prerequisites:
#   - ~/.hermes/profiles/gemini35/ exists with GOOGLE_API_KEY in .env
#   - .venv set up at the repo root
#
# Exit code:
#   0 on success; non-zero on any failure (missing profile, missing key,
#   API error, sqlite error).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE="${HERMES_BASELINE_PROFILE:-gemini35}"
PROFILE_PATH="${HOME}/.hermes/profiles/${PROFILE}"

if [[ ! -d "$PROFILE_PATH" ]]; then
  echo "ERROR: profile not found: $PROFILE_PATH" >&2
  echo "Set up via 'hermes profile create $PROFILE' first." >&2
  exit 2
fi

if [[ ! -f "$PROFILE_PATH/.env" ]]; then
  echo "ERROR: no .env in $PROFILE_PATH — API key required." >&2
  exit 3
fi

if [[ ! -d "$REPO_ROOT/.venv" ]]; then
  echo "ERROR: .venv not found at $REPO_ROOT/.venv" >&2
  echo "Run: uv venv .venv --python 3.11 && uv pip install -e \".[all,dev]\"" >&2
  exit 4
fi

# Run a one-shot prompt — kept short to minimize variability in output_tokens.
PROMPT="Reply with exactly: 'baseline-prompt-cache-test'"

# Use a fixed timestamp so the session row is identifiable.
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

# shellcheck source=/dev/null
source "$REPO_ROOT/.venv/bin/activate"
HERMES_HOME="$PROFILE_PATH" hermes chat -q "$PROMPT" > /dev/null 2>&1 || {
  echo "ERROR: hermes chat invocation failed" >&2
  exit 5
}

# Pull the most recent session row (the one we just created).
ROW="$(sqlite3 "$PROFILE_PATH/state.db" \
  "SELECT model, input_tokens, output_tokens, cache_read_tokens,
          estimated_cost_usd, billing_provider, cost_status
   FROM sessions
   WHERE input_tokens > 0
   ORDER BY started_at DESC LIMIT 1")"

if [[ -z "$ROW" ]]; then
  echo "ERROR: no session row found with input_tokens > 0 — call may have failed" >&2
  exit 6
fi

IFS='|' read -r MODEL INPUT_TOKENS OUTPUT_TOKENS CACHE_READ COST PROVIDER STATUS <<< "$ROW"

cat <<EOF
{
  "prompt": "${PROMPT}",
  "profile": "${PROFILE}",
  "captured_at_utc": "${TIMESTAMP}",
  "observed": {
    "model": "${MODEL}",
    "input_tokens": ${INPUT_TOKENS},
    "output_tokens": ${OUTPUT_TOKENS},
    "cache_read_tokens": ${CACHE_READ},
    "estimated_cost_usd": ${COST:-null},
    "billing_provider": "${PROVIDER}",
    "cost_status": "${STATUS}"
  },
  "invariants_to_assert": [
    "input_tokens > 0",
    "output_tokens > 0",
    "cost_status == 'estimated'",
    "estimated_cost_usd > 0"
  ]
}
EOF
