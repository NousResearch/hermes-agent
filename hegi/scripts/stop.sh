#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin=""
for candidate in "$repo_root/.venv/bin/python" "$repo_root/venv/bin/python" "$(command -v python3 || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then python_bin="$candidate"; break; fi
done
[[ -n "$python_bin" ]] || { echo "Python runtime not found" >&2; exit 1; }
runtime_home="${HERMES_HOME:-$("$python_bin" -m hegi.bootstrap locate-home)}"
pidfile="$runtime_home/hegi/daemon.pid"
readyfile="$runtime_home/hegi/daemon.ready"

if command -v systemctl >/dev/null 2>&1 \
  && systemctl --user is-active --quiet hegi.service 2>/dev/null; then
  systemctl --user stop hegi.service
fi

if [[ ! -f "$pidfile" ]]; then
  rm -f "$readyfile"
  echo "HEGI daemon is not running"
  exit 0
fi
pid="$(<"$pidfile")"
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
  echo "Invalid HEGI pidfile: $pidfile" >&2
  exit 2
fi
if kill -0 "$pid" 2>/dev/null; then
  command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
  if [[ "$command_line" != *"hegi daemon"* ]]; then
    rm -f "$pidfile" "$readyfile"
    echo "Removed stale HEGI pidfile; pid=$pid belongs to another process"
    exit 0
  fi
  kill -TERM "$pid"
  for _ in {1..100}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "HEGI daemon did not stop within 10 seconds: pid=$pid" >&2
    exit 1
  fi
  echo "HEGI daemon stopped: pid=$pid"
else
  echo "Removed stale HEGI pidfile: pid=$pid"
fi
rm -f "$pidfile" "$readyfile"
