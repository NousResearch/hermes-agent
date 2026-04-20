#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${HERMES_AUTOSTART_REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
LOG_DIR="${HERMES_AUTOSTART_LOG_DIR:-${HOME}/.hermes/logs}"
MODE="${HERMES_AUTOSTART_MODE:-gateway}"
STARTUP_DELAY="${HERMES_AUTOSTART_STARTUP_DELAY:-15}"
RESTART_DELAY="${HERMES_AUTOSTART_RESTART_DELAY:-5}"
MAX_RESTART_DELAY="${HERMES_AUTOSTART_MAX_RESTART_DELAY:-60}"
WATCHDOG_ENABLED="${HERMES_AUTOSTART_WATCHDOG:-1}"
DASHBOARD_ENABLED="${HERMES_AUTOSTART_WEBUI:-0}"

mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/autostart.log" 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

is_gateway_running() {
  pgrep -f 'hermes_cli\.main.*gateway run|hermes gateway run' >/dev/null 2>&1
}

start_gateway_background() {
  local logfile="$LOG_DIR/gateway.log"
  if [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
    "$PROJECT_DIR/venv/bin/python" -m hermes_cli.main gateway run --replace >>"$logfile" 2>&1 &
    printf '%s' "$!"
    return 0
  fi

  if command -v hermes >/dev/null 2>&1; then
    hermes gateway run --replace >>"$logfile" 2>&1 &
    printf '%s' "$!"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 -m hermes_cli.main gateway run --replace >>"$logfile" 2>&1 &
    printf '%s' "$!"
    return 0
  fi

  return 1
}

start_dashboard_background() {
  local logfile="$LOG_DIR/dashboard.log"
  if [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
    "$PROJECT_DIR/venv/bin/python" -m hermes_cli.main dashboard --no-open >>"$logfile" 2>&1 &
    printf '%s' "$!"
    return 0
  fi

  if command -v hermes >/dev/null 2>&1; then
    hermes dashboard --no-open >>"$logfile" 2>&1 &
    printf '%s' "$!"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 -m hermes_cli.main dashboard --no-open >>"$logfile" 2>&1 &
    printf '%s' "$!"
    return 0
  fi

  return 1
}

watch_gateway() {
  local delay="$RESTART_DELAY"
  local pid

  if is_gateway_running; then
    log "Gateway is already running; leaving it alone."
    return 0
  fi

  while true; do
    if ! pid="$(start_gateway_background)"; then
      log "No gateway launch command was found."
      return 1
    fi

    log "Gateway PID: ${pid}"

    if wait "$pid"; then
      log "Gateway exited cleanly."
    else
      log "Gateway exited with status $?"
    fi

    if [[ "$WATCHDOG_ENABLED" != "1" ]]; then
      log "Watchdog disabled; not restarting gateway."
      return 0
    fi

    if is_gateway_running; then
      log "Another gateway instance is already running; stopping watchdog."
      return 0
    fi

    log "Restarting gateway in ${delay}s..."
    sleep "$delay"
    if (( delay < MAX_RESTART_DELAY )); then
      delay=$(( delay * 2 ))
      if (( delay > MAX_RESTART_DELAY )); then
        delay="$MAX_RESTART_DELAY"
      fi
    fi
  done
}

watch_dashboard() {
  local delay="${HERMES_AUTOSTART_DASHBOARD_RESTART_DELAY:-10}"
  local pid

  if [[ "$DASHBOARD_ENABLED" != "1" ]]; then
    log "Web UI mode disabled; dashboard watcher will not start."
    return 0
  fi

  while true; do
    if ! pid="$(start_dashboard_background)"; then
      log "Dashboard launch command not found; skipping web UI mode."
      return 0
    fi

    log "Dashboard PID: ${pid}"

    if wait "$pid"; then
      log "Dashboard exited cleanly."
      return 0
    fi

    log "Dashboard exited with status $?; restarting in ${delay}s..."
    sleep "$delay"
  done
}

main() {
  local normalized_mode
  normalized_mode="$(printf '%s' "$MODE" | tr '[:upper:]' '[:lower:]')"

  case "$normalized_mode" in
    gateway|headless)
      DASHBOARD_ENABLED=0
      ;;
    webui|dashboard|both)
      DASHBOARD_ENABLED=1
      ;;
    *)
      log "Unknown HERMES_AUTOSTART_MODE='$MODE' (expected gateway, headless, dashboard, webui, or both)."
      return 1
      ;;
  esac

  log "Starting Hermes autostart watcher (mode=${normalized_mode})"
  log "Project directory: ${PROJECT_DIR}"
  log "Log directory: ${LOG_DIR}"
  sleep "$STARTUP_DELAY"

  if [[ "$DASHBOARD_ENABLED" != "1" ]]; then
    watch_gateway
    return 0
  fi

  watch_gateway &
  local gateway_watcher_pid=$!
  log "Gateway watcher PID: ${gateway_watcher_pid}"

  watch_dashboard &
  local dashboard_watcher_pid=$!
  log "Dashboard watcher PID: ${dashboard_watcher_pid}"

  wait "$gateway_watcher_pid"
  wait "$dashboard_watcher_pid" || true
}

trap 'log "Autostart watcher interrupted; exiting."' INT TERM
main "$@"
