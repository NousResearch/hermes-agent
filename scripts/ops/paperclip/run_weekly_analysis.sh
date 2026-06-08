#!/usr/bin/env bash
# Paperclip Weekly Analysis — pure bash + Telegram Bot API.
# Replaces Hermes cron job aa5b511f4200, which fails because the Anthropic
# OAuth content filter rejects the Hermes system prompt. trajectory.py emits
# the formatted text directly (and handles the empty-data case itself), so
# we just relay its stdout.
#
# Schedule via user crontab: `0 9 * * 1` (weekly, Monday 09:00)
set -uo pipefail

REPO=/home/leos/paperclip-agent-daemon
HERMES_PY=/home/leos/hermes-agent/venv/bin/python3
HERMES_ENV=/home/leos/.hermes/.env
LOG_DIR=/home/leos/.hermes/cron/output
TG_CHAT_ID=7596078461

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/paperclip_weekly_analysis_$(date -u +%Y%m%dT%H%M%SZ).log"
exec >"$LOG" 2>&1

echo "[$(date -Iseconds)] start: paperclip-weekly-analysis"

set -a
# shellcheck disable=SC1090
source <(grep -E '^TELEGRAM_BOT_TOKEN=' "$HERMES_ENV")
set +a

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
    echo "FAIL: missing TELEGRAM_BOT_TOKEN"
    exit 2
fi

cd "$REPO"
REPORT=$("$HERMES_PY" trajectory.py 2>&1)
RC=$?

if [[ $RC -ne 0 ]]; then
    REPORT=$'❌ Paperclip Weekly Analysis selhal (exit '"$RC"$').\n\n'"$REPORT"
fi

if [[ ${#REPORT} -gt 4000 ]]; then
    REPORT="${REPORT:0:3990}…(truncated)"
fi

HTTP=$(curl --silent --show-error --output /tmp/tg_resp.$$ --write-out '%{http_code}' \
    --max-time 30 \
    --data-urlencode "chat_id=$TG_CHAT_ID" \
    --data-urlencode "text=$REPORT" \
    --data-urlencode "disable_web_page_preview=true" \
    "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage")
TG_RC=$?
TG_BODY=$(cat /tmp/tg_resp.$$ 2>/dev/null)
rm -f /tmp/tg_resp.$$

echo "[$(date -Iseconds)] telegram http=$HTTP curl_rc=$TG_RC"
if [[ "$HTTP" != "200" ]]; then
    echo "telegram body: $TG_BODY"
    exit 3
fi

echo "[$(date -Iseconds)] done"
