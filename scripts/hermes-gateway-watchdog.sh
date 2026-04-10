 #!/bin/bash
  # Feishu Gateway Watchdog v4
  # Replaces ss-based TCP check with pure log-based health detection
  # Fixes: https://github.com/NousResearch/hermes-agent/issues/7213

  GATEWAY_LOG="/root/.hermes/logs/gateway.log"

  # 1. Check gateway process is alive
  if ! systemctl is-active --quiet hermes-gateway; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] Gateway not running, starting..."
      systemctl start hermes-gateway
      exit 0
  fi

  # 2. Check if WS disconnected >5min without reconnect (log-based)
  LAST_DISCONNECT=$(tail -200 "$GATEWAY_LOG" | grep "Feishu.*Disconnected" | tail -1)
  LAST_CONNECT=$(tail -200 "$GATEWAY_LOG" | grep "Feishu.*Connected in websocket" | tail -1)

  if [ -n "$LAST_DISCONNECT" ]; then
      DISC_TIME=$(echo "$LAST_DISCONNECT" | awk '{print $1, $2}' | cut -d',' -f1)
      CONN_TIME=""
      if [ -n "$LAST_CONNECT" ]; then
          CONN_TIME=$(echo "$LAST_CONNECT" | awk '{print $1, $2}' | cut -d',' -f1)
      fi

      DISC_TS=$(date -d "$DISC_TIME" +%s 2>/dev/null || echo 0)
      CONN_TS=0
      if [ -n "$CONN_TIME" ]; then
          CONN_TS=$(date -d "$CONN_TIME" +%s 2>/dev/null || echo 0)
      fi

      NOW=$(date +%s)
      if [ "$DISC_TS" -gt "$CONN_TS" ] && [ $((NOW - DISC_TS)) -gt 300 ]; then
          echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] WS disconnected $((NOW - DISC_TS))s without reconnect,
  restarting..."
          systemctl restart hermes-gateway
          exit 0
      fi
  fi

  # 3. Startup failure: process alive >60s but never connected
  PID=$(systemctl show hermes-gateway -p MainPID --value 2>/dev/null)
  if [ -n "$PID" ] && [ "$PID" -gt 0 ]; then
      UPTIME_SEC=$(ps -o etimes= -p "$PID" 2>/dev/null | tr -d ' ')
      if [ -n "$UPTIME_SEC" ] && [ "$UPTIME_SEC" -gt 60 ]; then
          LAST_CONNECT_LOG=$(tail -50 "$GATEWAY_LOG" | grep "Connected in websocket" | tail -1)
          if [ -z "$LAST_CONNECT_LOG" ]; then
              echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] No WS connect after ${UPTIME_SEC}s, restarting..."
              systemctl restart hermes-gateway
              exit 0
          fi
      fi
  fi
