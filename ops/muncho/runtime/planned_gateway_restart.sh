#!/usr/bin/env bash
set -euo pipefail

mark_planned_gateway_stop() {
  local active_link="$1"
  local service="$2"
  local owner="$3"
  local hermes_home="$4"
  local pid python_bin

  pid="$(systemctl show "$service" --property MainPID --value)"
  if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
    echo "BLOCKED_INVALID_GATEWAY_PID" >&2
    return 2
  fi
  if [ "$pid" = "0" ]; then
    echo "PLANNED_STOP_MARKER_NOT_NEEDED"
    return 0
  fi

  python_bin="$active_link/.venv/bin/python"
  if [ ! -x "$python_bin" ]; then
    echo "BLOCKED_ACTIVE_GATEWAY_PYTHON_MISSING" >&2
    return 3
  fi

  if ! sudo -n -u "$owner" env \
    HERMES_HOME="$hermes_home" \
    PYTHONPATH="$active_link" \
    "$python_bin" -c \
    'import sys; from gateway.status import write_planned_stop_marker; raise SystemExit(0 if write_planned_stop_marker(int(sys.argv[1])) else 1)' \
    "$pid"
  then
    echo "BLOCKED_PLANNED_STOP_MARKER_WRITE_FAILED" >&2
    return 4
  fi

  echo "PLANNED_STOP_MARKER_PASS"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [ "$#" -ne 4 ]; then
    echo "Usage: planned_gateway_restart.sh <active-link> <service> <owner> <hermes-home>" >&2
    exit 2
  fi
  mark_planned_gateway_stop "$1" "$2" "$3" "$4"
fi
