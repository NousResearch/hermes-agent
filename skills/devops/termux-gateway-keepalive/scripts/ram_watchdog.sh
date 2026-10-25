#!/usr/bin/env bash
# ram_watchdog.sh — protects Android device from low-memory (OOM) shutdowns.
#
# Prevents the OOM chain: Chrome/subprocesses grow -> physical RAM exhausted ->
# swap fills -> Android Low Memory Killer terminates the gateway -> Termux dies.
#
# Defense: monitor every 20s with escalating response:
#   WARN  (swap >600MB or chrome RSS >450MB): log only
#   KILL  (swap >900MB or chrome RSS >650MB): stop browser service, kill stragglers
#   PANIC (swap >1200MB): kill all chrome & Xvfb, protect gateway at all costs
#
set -u
HOME_DIR="$HOME"
LOG="$HOME_DIR/.hermes/logs/ram_watchdog.log"
STATE="$HOME_DIR/.hermes/ram_watchdog.state"
COOLDOWN_FILE="$HOME_DIR/.hermes/ram_watchdog_cooldown"
mkdir -p "$(dirname "$LOG")"

log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }

chrome_rss() {
  local T=0
  for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$c" in *chromium*)
      rss=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null)
      T=$((T+${rss:-0}))
    ;; esac
  done 2>/dev/null
  echo $((T/1024))
}

swap_used() {
  awk '/SwapTotal/{s=$2} /SwapFree/{f=$2} END{printf "%d", (s-f)/1024}' /proc/meminfo 2>/dev/null || echo 0
}

kill_browser() {
  # DOWN the runit service so it stays dead (break the kill-restart treadmill)
  if [ -d "/data/data/com.termux/files/usr/var/service" ]; then
    cd /data/data/com.termux/files/usr/var/service && SVDIR=$PWD sv down chromium-headless 2>/dev/null || true
  fi
  sleep 1
  for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in *chromium/chrome*|*chromium/*) kill -9 $p 2>/dev/null || true;; esac
  done 2>/dev/null
}

alert() {
  if [ "${level:-}" = "PANIC" ] || ! pgrep -f "venv/bin/hermes gateway" >/dev/null 2>&1; then
    termux-toast "$1" 2>/dev/null || true
    termux-notification --title "RAM Watchdog" --content "$1" --priority high 2>/dev/null || true
  fi
}

log "watchdog started (pid $$)"

while true; do
  SWAP=$(swap_used)
  CRSS=$(chrome_rss)
  CHROME_ALIVE=$(for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    if echo "$c" | grep -q "chromium"; then
      echo y
      break
    fi
  done)

  LEVEL="ok"
  if [ "${SWAP:-0}" -gt 1200 ]; then LEVEL="PANIC"
  elif [ "${SWAP:-0}" -gt 900 ] || [ "${CRSS:-0}" -gt 650 ]; then LEVEL="KILL"
  elif [ "${SWAP:-0}" -gt 600 ] || [ "${CRSS:-0}" -gt 450 ]; then LEVEL="WARN"
  fi

  NOW=$(date +%s)
  LAST=$(cat "$COOLDOWN_FILE" 2>/dev/null || echo 0)
  CAN_ACT=$(( NOW - LAST > 180 ))

  case "$LEVEL" in
    WARN)
      if [ "$CAN_ACT" -eq 1 ] && [ -n "$CHROME_ALIVE" ]; then
        log "WARN swap=${SWAP}MB chromeRSS=${CRSS}MB — watching closely"
        echo "$NOW" > "$COOLDOWN_FILE"
      fi
      ;;
    KILL)
      KILLS_TODAY=$(grep -c "KILL " "$LOG" 2>/dev/null || echo 0)
      if [ "$KILLS_TODAY" -gt 8 ]; then
        log "ESCALATE: ${KILLS_TODAY} kills today — browser stays OFF (treadmill prevention)"
        kill_browser
        echo "$NOW" > "$COOLDOWN_FILE"
      elif [ "$CAN_ACT" -eq 1 ] && [ -n "$CHROME_ALIVE" ]; then
        log "KILL swap=${SWAP}MB chromeRSS=${CRSS}MB — kill #${KILLS_TODAY} today, browser DOWN"
        level="KILL" alert "⚠️ RAM: closing browser (swap ${SWAP}MB)"
        kill_browser
        echo "$NOW" > "$COOLDOWN_FILE"
      fi
      ;;
    PANIC)
      log "PANIC swap=${SWAP}MB — killing ALL chrome, protecting gateway"
      level="PANIC" alert "🚨 RAM PANIC: killing browser to save phone"
      kill_browser
      pkill -x Xvfb 2>/dev/null || true
      echo "$NOW" > "$COOLDOWN_FILE"
      ;;
  esac

  echo "{\"ts\":\"$(date +%s)\",\"swap\":${SWAP:-0},\"chrome_rss\":${CRSS:-0},\"level\":\"$LEVEL\"}" > "$STATE"
  sleep 20
done
