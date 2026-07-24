#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin=""
for candidate in "$repo_root/.venv/bin/python" "$repo_root/venv/bin/python" "$(command -v python3 || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    python_bin="$candidate"
    break
  fi
done
[[ -n "$python_bin" ]] || { echo "Python runtime not found" >&2; exit 1; }

send=false
if [[ "${1:-}" == "--send" ]]; then
  send=true
  shift
fi
[[ $# -eq 0 ]] || { echo "Usage: $0 [--send]" >&2; exit 2; }

runtime_home="${HERMES_HOME:-$("$python_bin" -m hegi.bootstrap locate-home)}"
export HERMES_HOME="$runtime_home"
state_dir="$runtime_home/hegi"
pidfile="$state_dir/daemon.pid"
readyfile="$state_dir/daemon.ready"
mkdir -p "$state_dir"

cd "$repo_root"
"$python_bin" -m hegi doctor >/dev/null

if [[ -f "$pidfile" ]]; then
  pid="$(<"$pidfile")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null \
    && tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -q 'hegi daemon'; then
    echo "HEGI daemon already running: pid=$pid"
    exit 0
  fi
  rm -f "$pidfile" "$readyfile"
fi

use_systemd=false
if [[ "$send" == true ]] && command -v systemctl >/dev/null 2>&1 \
  && systemctl --user cat hegi.service >/dev/null 2>&1; then
  use_systemd=true
  systemctl --user start hegi.service
else
  daemon_args=(daemon)
  [[ "$send" == true ]] && daemon_args+=(--send)
  nohup "$python_bin" -m hegi "${daemon_args[@]}" >>"$state_dir/daemon.log" 2>&1 &
fi

for _ in {1..100}; do
  if [[ -f "$readyfile" && -f "$pidfile" ]]; then
    pid="$(<"$pidfile")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "HEGI daemon ready: pid=$pid send=$send systemd=$use_systemd"
      exit 0
    fi
  fi
  if [[ "$use_systemd" == true ]] && ! systemctl --user is-active --quiet hegi.service; then
    break
  fi
  sleep 0.1
done

echo "HEGI daemon failed readiness check; inspect $state_dir/daemon.log" >&2
if [[ "$use_systemd" == true ]]; then
  systemctl --user status hegi.service --no-pager >&2 || true
else
  tail -n 80 "$state_dir/daemon.log" >&2 || true
fi
exit 1
