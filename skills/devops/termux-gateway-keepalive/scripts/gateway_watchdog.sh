#!/usr/bin/env bash
# gateway_watchdog.sh — supervised, detached, wake-locked Hermes gateway keeper.
#
# Prevents Android from terminating the Hermes gateway when Termux is backgrounded.
#
# What it does:
#   1. Acquires a termux-wake-lock (keeps the device CPU awake).
#   2. Runs the gateway detached (setsid), independent of the interactive shell.
#   3. If the gateway process dies, restarts it (max 10 attempts in 10 min, then
#      backs off to a 60s loop to prevent fork-bombing).
#   4. If this script is killed, a cron monitor (gateway_monitor.sh) restarts it.
#
# Usage:
#   bash gateway_watchdog.sh start   # launch (detached, returns immediately)
#   bash gateway_watchdog.sh stop    # stop gateway + watchdog + release lock
#   bash gateway_watchdog.sh status  # check status
#
set -u
HOME_DIR="$HOME"
GW_PY="$HOME_DIR/.hermes/hermes-agent/venv/bin/hermes"
LOCK_FILE="$HOME_DIR/.hermes/gateway_watchdog.lock"
PID_FILE="$HOME_DIR/.hermes/gateway_watchdog.pid"
RUN_FLAG="$HOME_DIR/.hermes/gateway_watchdog.run"

start_gateway() {
    mkdir -p "$HOME_DIR/.hermes/logs"
    setsid "$GW_PY" gateway >> "$HOME_DIR/.hermes/logs/gateway_stdout.log" 2>&1 &
    echo $! > "$HOME_DIR/.hermes/gateway.pid"
}

stop_all() {
    rm -f "$RUN_FLAG"
    if [ -f "$HOME_DIR/.hermes/gateway.pid" ]; then
        kill "$(cat "$HOME_DIR/.hermes/gateway.pid")" 2>/dev/null || true
    fi
    pkill -f "hermes-agent/venv/bin/hermes gateway" 2>/dev/null || true
    command -v termux-wake-unlock >/dev/null 2>&1 && termux-wake-unlock 2>/dev/null || true
    rm -f "$LOCK_FILE" "$PID_FILE" "$HOME_DIR/.hermes/gateway.pid"
    echo "stopped gateway + watchdog, released wake-lock"
}

status() {
    if pgrep -f "hermes-agent/venv/bin/hermes gateway" >/dev/null 2>&1; then
        echo "gateway: ALIVE (pid $(pgrep -f 'hermes-agent/venv/bin/hermes gateway' | head -1))"
    else
        echo "gateway: DEAD"
    fi
    if [ -f "$RUN_FLAG" ]; then
        echo "watchdog: RUNNING (pid $(cat "$PID_FILE" 2>/dev/null))"
    else
        echo "watchdog: not running"
    fi
}

case "${1:-start}" in
    start)
        if [ -f "$RUN_FLAG" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
            echo "watchdog already running (pid $(cat "$PID_FILE"))"
            exit 0
        fi
        mkdir -p "$HOME_DIR/.hermes/logs"
        touch "$RUN_FLAG"
        echo $$ > "$PID_FILE"
        command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock 2>/dev/null || true
        echo "watchdog started (pid $$), wake-lock held"
        attempts=0
        window_start=$(date +%s)
        while [ -f "$RUN_FLAG" ]; do
            if ! pgrep -f "hermes-agent/venv/bin/hermes gateway" >/dev/null 2>&1; then
                now=$(date +%s)
                if [ $((now - window_start)) -gt 600 ]; then
                    attempts=0
                    window_start=$now
                fi
                attempts=$((attempts+1))
                if [ $attempts -gt 10 ]; then
                    sleep 60
                    attempts=0
                    window_start=$(date +%s)
                    continue
                fi
                echo "$(date '+%Y-%m-%d %H:%M:%S') watchdog: gateway down, restart #$attempts" >> "$HOME_DIR/.hermes/logs/gateway_watchdog.log"
                start_gateway
                bash "$HOME_DIR/skills/devops/termux-gateway-keepalive/scripts/presence_notify.sh" "✅ Hermes Agent: gateway recovered automatically (restart #$attempts)." 2>/dev/null || true
            fi
            sleep 15
        done
        ;;
    stop) stop_all ;;
    status) status ;;
    *) echo "usage: $0 {start|stop|status}"; exit 1 ;;
esac
