#!/usr/bin/env bash
# resurrect.sh — called by Android WorkManager outside Termux's process tree.
# Restores Termux, Gateway, and Services to full operation after any OS-level kill.
set -u
HOME_DIR="$HOME"
LOG="$HOME_DIR/.hermes/logs/resurrect.log"
mkdir -p "$(dirname "$LOG")"
echo "$(date) resurrection triggered" >> "$LOG"

# 1. Acquire wake lock immediately
command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock 2>/dev/null || true

GATEWAY_WAS_DEAD=0

# 2. Check if gateway is alive
if ! pgrep -f "venv/bin/hermes gateway" > /dev/null 2>&1; then
    echo "$(date) gateway dead — restarting" >> "$LOG"
    GATEWAY_WAS_DEAD=1
    cd "$HOME_DIR"
    nohup "$HOME_DIR/.hermes/hermes-agent/venv/bin/hermes" gateway >> "$HOME_DIR/.hermes/logs/gateway_stdout.log" 2>&1 &
    sleep 5
fi

# 3. Restart runit service directory if not running
if ! pgrep -x runsvdir > /dev/null 2>&1 && [ -d "/data/data/com.termux/files/usr/var/service" ]; then
    echo "$(date) runsvdir dead — restarting" >> "$LOG"
    cd /data/data/com.termux/files/usr/var/service
    SVDIR=$PWD setsid runsvdir "$PWD" > /dev/null 2>&1 &
    sleep 3
fi

# 4. Re-apply RAM priorities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/termux_priority.sh" > /dev/null 2>&1 || true

# 5. Notify operator via Telegram if gateway was dead and revived
if [ "$GATEWAY_WAS_DEAD" = "1" ]; then
    BOT_TOKEN=$(grep "^TELEGRAM_BOT_TOKEN=." "$HOME_DIR/.hermes/.env" 2>/dev/null | head -1 | cut -d= -f2 | tr -d '"' | tr -d "'")
    CHAT_ID=$(grep -E "^(TELEGRAM_CHAT_ID|TELEGRAM_ADMIN_ID|TELEGRAM_ALLOWED_USERS)=" "$HOME_DIR/.hermes/.env" 2>/dev/null | head -1 | cut -d= -f2 | tr -d '"' | tr -d "'" | cut -d, -f1)
    if [ -n "$BOT_TOKEN" ] && [ -n "$CHAT_ID" ]; then
        curl -s -m 10 "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
            -d chat_id="$CHAT_ID" \
            -d text="🔄 Hermes Agent resurrected: gateway was dead, restarted successfully at $(date +%H:%M). All systems restored." > /dev/null 2>&1 || true
        echo "$(date) telegram notification sent" >> "$LOG"
    fi
fi

echo "$(date) resurrection complete" >> "$LOG"
