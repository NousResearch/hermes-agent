#!/usr/bin/env bash
# Paperclip Daily Report — pure bash + Telegram Bot API.
# Replaces Hermes cron job 4c43dff82cdc, which fails because the Anthropic
# OAuth content filter rejects the Hermes system prompt. This script needs no
# LLM — daily_report.py emits the formatted text directly; we just relay it.
#
# Schedule via user crontab: `0 8 * * *`
# Logs: /home/leos/.hermes/cron/output/paperclip_daily_report_<UTC>.log
set -uo pipefail

REPO=/home/leos/paperclip-agent-daemon
HERMES_PY=/home/leos/hermes-agent/venv/bin/python3
HERMES_ENV=/home/leos/.hermes/.env
LOG_DIR=/home/leos/.hermes/cron/output
TG_CHAT_ID=7596078461

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/paperclip_daily_report_$(date -u +%Y%m%dT%H%M%SZ).log"
exec >"$LOG" 2>&1

echo "[$(date -Iseconds)] start: paperclip-daily-report"

# Load secrets without leaking them to stdout/log
set -a
# shellcheck disable=SC1090
source <(grep -E '^(PAPERCLIP_DAEMON_SECRET|PAPERCLIP_BOARD_EMAIL|PAPERCLIP_BOARD_PASSWORD)=' "$REPO/.env")
# shellcheck disable=SC1090
source <(grep -E '^(TELEGRAM_BOT_TOKEN)=' "$HERMES_ENV")
set +a

if [[ -z "${PAPERCLIP_DAEMON_SECRET:-}" || -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
    echo "FAIL: missing PAPERCLIP_DAEMON_SECRET or TELEGRAM_BOT_TOKEN"
    exit 2
fi

cd "$REPO"
REPORT=$("$HERMES_PY" daily_report.py 2>&1)
RC=$?

if [[ $RC -ne 0 ]]; then
    REPORT=$'❌ Paperclip Daily Report selhal (exit '"$RC"$').\n\n'"$REPORT"
fi

# Telegram caps text at 4096 chars per message; truncate gracefully.
if [[ ${#REPORT} -gt 4000 ]]; then
    REPORT="${REPORT:0:3990}…(truncated)"
fi

# Send via Bot API
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
