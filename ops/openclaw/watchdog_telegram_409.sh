#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$SCRIPT_DIR/watchdog_telegram_409_inner.sh"

if [ ! -x "$INNER" ]; then
  echo "watchdog_telegram_409 wrapper error: inner script missing or not executable: $INNER" >&2
  exit 1
fi

# Retry policy for transient fetch/network failures.
# - No retry on success (0) or real conflict (2).
# - Retry only on other failures (commonly transient fetch errors).
# - Backoff: 2s then 5s.
attempt=1
for backoff in 0 2 5; do
  if [ "$backoff" -gt 0 ]; then
    sleep "$backoff"
  fi

  set +e
  output="$($INNER "$@" 2>&1)"
  code=$?
  set -e

  # Replay underlying output so existing parsers/logs still see it.
  if [ -n "$output" ]; then
    printf '%s\n' "$output"
  fi

  case "$code" in
    0|2)
      exit "$code"
      ;;
    *)
      if [ "$attempt" -lt 3 ]; then
        echo "watchdog_telegram_409 wrapper: transient failure (exit=$code), retrying... attempt=$((attempt+1))/3" >&2
      fi
      ;;
  esac

  attempt=$((attempt+1))
done

echo "watchdog_telegram_409 wrapper: fetch_error_persistent after 3 attempts" >&2
exit 1
