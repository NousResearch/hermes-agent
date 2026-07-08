#!/bin/bash
# systemd oneshot — 由 hermes-restart.timer 每 10s 触发
# 检查 ~/.hermes/state/restart-flag，存在时强制重启 gateway
FLAG="/home/ohtok/.hermes/state/restart-flag"
LOG="/tmp/hermes_restart.log"
touch "$LOG" 2>/dev/null || LOG="/dev/stderr"

if [ -f "$FLAG" ]; then
    rm -f "$FLAG"
    state=$(systemctl is-active hermes-gateway 2>/dev/null)
    echo "[$(date)] flag found, state=$state" >> "$LOG"
    
    if [ "$state" = "active" ]; then
        echo "[$(date)] restarting" >> "$LOG"
        systemctl restart hermes-gateway
    elif [ "$state" = "deactivating" ]; then
        # Stuck deactivating — force-kill the process
        pid=$(systemctl show hermes-gateway -p MainPID --value 2>/dev/null)
        echo "[$(date)] deactivating stuck, killing pid=$pid" >> "$LOG"
        kill -9 $pid 2>/dev/null || true
        sleep 3
        # Verify process is dead
        if kill -0 $pid 2>/dev/null; then
            echo "[$(date)] WARNING: pid=$pid still alive after kill -9" >> "$LOG"
        fi
        systemctl restart hermes-gateway
    else
        # inactive/failed — just start
        echo "[$(date)] state=$state, starting" >> "$LOG"
        systemctl start hermes-gateway
    fi
    
    # Verify restart succeeded
    sleep 5
    new_state=$(systemctl is-active hermes-gateway 2>/dev/null)
    echo "[$(date)] post-restart state=$new_state" >> "$LOG"
    if [ "$new_state" != "active" ]; then
        echo "[$(date)] RESTART FAILED: gateway in state=$new_state" >> "$LOG"
        # Event-driven alert: touch marker file, 30min cron picks it up
        ALERT_MARKER="/tmp/hermes_restart_alert"
        echo "[$(date)] gateway restart failed — state=$new_state" > "$ALERT_MARKER"
        chmod 644 "$ALERT_MARKER"
    fi
fi