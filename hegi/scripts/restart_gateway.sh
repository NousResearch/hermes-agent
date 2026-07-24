#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin=""
for candidate in "$repo_root/.venv/bin/python" "$repo_root/venv/bin/python" "$(command -v python3 || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then python_bin="$candidate"; break; fi
done
[[ -n "$python_bin" ]] || { echo "Python runtime not found" >&2; exit 1; }
runtime_home="${HERMES_HOME:-$("$python_bin" -m hegi.bootstrap locate-home)}"
pidfile="$runtime_home/gateway.pid"
if [[ ! -f "$pidfile" ]]; then
  echo "Hermes gateway is not running; HEGI plugin will load on its next start."
  exit 0
fi
raw_pidfile="$(<"$pidfile")"
if [[ "$raw_pidfile" == \{* ]]; then
  old_pid="$(printf '%s' "$raw_pidfile" | sed -n 's/.*"pid"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')"
else
  old_pid="$raw_pidfile"
fi
if [[ ! "$old_pid" =~ ^[0-9]+$ ]] || ! kill -0 "$old_pid" 2>/dev/null; then
  echo "Hermes gateway pidfile is stale; plugin will load on its next start."
  exit 0
fi
command_line="$(tr '\0' ' ' <"/proc/$old_pid/cmdline" 2>/dev/null || true)"
if [[ "$command_line" != *"hermes"* || "$command_line" != *"gateway"* ]]; then
  echo "Refusing to stop unrelated process from gateway pidfile: pid=$old_pid" >&2
  exit 1
fi

hermes_bin="$(command -v hermes || true)"
if [[ -z "$hermes_bin" && -x "$HOME/.local/bin/hermes" ]]; then
  hermes_bin="$HOME/.local/bin/hermes"
fi
[[ -n "$hermes_bin" ]] || { echo "hermes CLI not found" >&2; exit 1; }
profile_args=()
if [[ "$runtime_home" == "$HOME/.hermes/profiles/"* ]]; then
  profile_args=(-p "$(basename "$runtime_home")")
fi

kill -TERM "$old_pid"
for _ in {1..300}; do
  kill -0 "$old_pid" 2>/dev/null || break
  sleep 0.1
done
if kill -0 "$old_pid" 2>/dev/null; then
  echo "Hermes gateway did not stop within 30 seconds: pid=$old_pid" >&2
  exit 1
fi

log="$runtime_home/logs/gateway-start.log"
mkdir -p "$(dirname "$log")"
nohup "$hermes_bin" "${profile_args[@]}" gateway run --force >>"$log" 2>&1 &
launcher_pid=$!
for _ in {1..300}; do
  if [[ -f "$pidfile" ]]; then
    raw_pidfile="$(<"$pidfile")"
    if [[ "$raw_pidfile" == \{* ]]; then
      new_pid="$(printf '%s' "$raw_pidfile" | sed -n 's/.*"pid"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')"
    else
      new_pid="$raw_pidfile"
    fi
    if [[ "$new_pid" =~ ^[0-9]+$ && "$new_pid" != "$old_pid" ]] \
      && kill -0 "$new_pid" 2>/dev/null; then
      echo "Hermes gateway restarted for HEGI plugin: pid=$new_pid"
      exit 0
    fi
  fi
  kill -0 "$launcher_pid" 2>/dev/null || break
  sleep 0.1
done
echo "Hermes gateway failed to restart; inspect $log" >&2
tail -n 80 "$log" >&2 || true
exit 1
