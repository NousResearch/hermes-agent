#!/usr/bin/env bash
# Load Bo2bot credentials from ~/.hermes/secrets/bo2bot.env and log in.
# Usage:
#   eval "$(bash bo2bot-login.sh --export)"   # load creds + session in current shell
#   bash bo2bot-login.sh                      # login only (prints status)

set -euo pipefail

CREDS_FILE="${HOME}/.hermes/secrets/bo2bot.env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT=0

for arg in "$@"; do
  case "$arg" in
    --export) EXPORT=1 ;;
    --help|-h)
      echo "Usage: bash bo2bot-login.sh [--export]"
      echo "  --export  print shell export statements (use with eval)"
      exit 0
      ;;
  esac
done

if ! python3 "$SCRIPT_DIR/bo2bot_cred_manager.py" --check >/dev/null 2>&1; then
  echo "❌ Bo2bot credentials missing or incomplete at $CREDS_FILE" >&2
  echo "   Human: fill that file per README Step 1 — do not paste secrets in chat." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$CREDS_FILE"
set +a

LOGIN_RESPONSE=$(curl -sS -X POST https://api.bo2bot.com/v1/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"account_id\": \"$BO2BOT_ACCOUNT_ID\", \"auth_key\": \"$BO2BOT_AUTH_KEY\"}")

BO2BOT_SESSION=$(echo "$LOGIN_RESPONSE" | jq -r '.session_token')
if [ "$BO2BOT_SESSION" = "null" ] || [ -z "$BO2BOT_SESSION" ]; then
  echo "❌ Login failed:" >&2
  echo "$LOGIN_RESPONSE" | jq . >&2
  exit 1
fi

if [ "$EXPORT" -eq 1 ]; then
  printf 'export BO2BOT_HANDLE=%q\n' "$BO2BOT_HANDLE"
  printf 'export BO2BOT_PUBLIC_ADDRESS=%q\n' "$BO2BOT_PUBLIC_ADDRESS"
  printf 'export BO2BOT_ACCOUNT_ID=%q\n' "$BO2BOT_ACCOUNT_ID"
  printf 'export BO2BOT_AUTH_KEY=%q\n' "$BO2BOT_AUTH_KEY"
  printf 'export BO2BOT_SESSION=%q\n' "$BO2BOT_SESSION"
else
  echo "✅ Logged in as $BO2BOT_HANDLE (session ${BO2BOT_SESSION:0:20}...)"
fi
