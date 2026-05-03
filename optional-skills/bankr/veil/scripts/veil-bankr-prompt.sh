#!/usr/bin/env bash
# Submit a prompt to Bankr and return the response.
# Delegates to the Bankr CLI when available, falls back to curl.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

need_bankr

PROMPT="$*"
if [[ -z "$PROMPT" ]]; then
  echo "Usage: $0 <prompt>" >&2
  exit 1
fi

# CLI path (preferred) — need_bankr already verified one of these is available
if command -v bankr >/dev/null 2>&1; then
  exec bankr agent "$PROMPT"
fi

# Curl fallback — config file was validated by need_bankr
API_KEY=$(jq -r '.apiKey // empty' "$BANKR_CONFIG")
API_URL=$(jq -r '.apiUrl // "https://api.bankr.bot"' "$BANKR_CONFIG")

if [[ -z "$API_KEY" ]]; then
  echo "apiKey missing in $BANKR_CONFIG" >&2
  exit 1
fi

# Submit
SUBMIT=$(curl -sf -X POST "$API_URL/agent/prompt" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "$(jq -nc --arg prompt "$PROMPT" '{prompt: $prompt}')")

JOB_ID=$(echo "$SUBMIT" | jq -r '.jobId // empty')
if [[ -z "$JOB_ID" ]]; then
  echo "$SUBMIT" | jq . >&2
  echo "Failed to get jobId from Bankr" >&2
  exit 1
fi

# Poll
ATTEMPT=0
MAX_ATTEMPTS=150
LAST_STATUS=""
while [[ $ATTEMPT -lt $MAX_ATTEMPTS ]]; do
  sleep 2
  STATUS=$(curl -sf -X GET "$API_URL/agent/job/$JOB_ID" -H "X-API-Key: $API_KEY") || {
    echo "Failed to poll Bankr job status (attempt $((ATTEMPT+1))/$MAX_ATTEMPTS)" >&2
    ATTEMPT=$((ATTEMPT+1))
    continue
  }
  LAST_STATUS="$STATUS"

  # Validate response is JSON
  if ! echo "$STATUS" | jq empty 2>/dev/null; then
    echo "Bankr returned non-JSON response: $STATUS" >&2
    ATTEMPT=$((ATTEMPT+1))
    continue
  fi

  STATE=$(echo "$STATUS" | jq -r '.status')
  case "$STATE" in
    completed)
      echo "$STATUS" | jq .
      exit 0
      ;;
    failed|cancelled)
      ERROR_DETAIL=$(echo "$STATUS" | jq -r '.error // .message // .reason // empty')
      echo "Bankr job $STATE: $JOB_ID" >&2
      [[ -n "$ERROR_DETAIL" ]] && echo "Detail: $ERROR_DETAIL" >&2
      echo "$STATUS" | jq . >&2
      exit 1
      ;;
    pending|processing)
      :
      ;;
    *)
      echo "Unexpected Bankr job status '$STATE' for job $JOB_ID" >&2
      echo "$STATUS" | jq . >&2
      ;;
  esac
  ATTEMPT=$((ATTEMPT+1))
done

echo "Timed out waiting for Bankr job: $JOB_ID (after $MAX_ATTEMPTS attempts)" >&2
if [[ -n "$LAST_STATUS" ]]; then
  LAST_STATE=$(echo "$LAST_STATUS" | jq -r '.status // "unknown"')
  echo "Last known status: $LAST_STATE" >&2
  echo "$LAST_STATUS" | jq . >&2
fi
exit 1
