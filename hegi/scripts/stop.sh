#!/usr/bin/env bash
set -euo pipefail

hermes_home="${HERMES_HOME:-$HOME/.hermes}"
pidfile="$hermes_home/hegi/daemon.pid"
if [[ ! -f "$pidfile" ]]; then
  echo "HEGI daemon is not running"
  exit 0
fi
pid="$(<"$pidfile")"
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
  echo "Invalid HEGI pidfile: $pidfile" >&2
  exit 2
fi
if kill -0 "$pid" 2>/dev/null; then
  kill -TERM "$pid"
  echo "Stop requested for HEGI daemon: pid=$pid"
else
  echo "Stale HEGI pidfile found; daemon is not running"
fi
