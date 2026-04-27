#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PID_FILE="${LOG_DIR}/gateway-guard.pid"
RUN_LOG="${LOG_DIR}/gateway-guard.log"

mkdir -p "${LOG_DIR}"

usage() {
  cat <<'EOF'
Usage: scripts/gateway_guard.sh <start|stop|restart|status|logs>

start   Start gateway in background with auto-restart loop
stop    Stop background guard process
restart Restart guard process
status  Show whether guard is running
logs    Follow logs
EOF
}

is_running() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

start_guard() {
  if is_running; then
    echo "gateway guard is already running (pid: $(cat "${PID_FILE}"))"
    exit 0
  fi

  (
    cd "${ROOT_DIR}"
    while true; do
      echo "===== $(date '+%F %T') gateway start =====" >> "${RUN_LOG}"
      if command -v caffeinate >/dev/null 2>&1; then
        caffeinate -dimsu venv/bin/python -m hermes_cli.main gateway run --replace -v >> "${RUN_LOG}" 2>&1
      else
        venv/bin/python -m hermes_cli.main gateway run --replace -v >> "${RUN_LOG}" 2>&1
      fi
      code=$?
      echo "===== $(date '+%F %T') gateway exited code=${code}, restart in 5s =====" >> "${RUN_LOG}"
      sleep 5
    done
  ) &

  echo $! > "${PID_FILE}"
  echo "gateway guard started (pid: $(cat "${PID_FILE}"))"
  echo "log file: ${RUN_LOG}"
}

stop_guard() {
  if ! is_running; then
    echo "gateway guard is not running"
    rm -f "${PID_FILE}"
    exit 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  kill "${pid}" 2>/dev/null || true
  sleep 1
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
  echo "gateway guard stopped"
}

status_guard() {
  if is_running; then
    echo "gateway guard is running (pid: $(cat "${PID_FILE}"))"
  else
    echo "gateway guard is not running"
  fi
}

logs_guard() {
  touch "${RUN_LOG}"
  tail -f "${RUN_LOG}"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  case "${1}" in
    start) start_guard ;;
    stop) stop_guard ;;
    restart) stop_guard; start_guard ;;
    status) status_guard ;;
    logs) logs_guard ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
