#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin=""
for candidate in "$repo_root/.venv/bin/python" "$repo_root/venv/bin/python" "$(command -v python3 || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then python_bin="$candidate"; break; fi
done
[[ -n "$python_bin" ]] || { echo "Python runtime not found" >&2; exit 1; }

apply=false
no_systemd=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) apply=true ;;
    --no-systemd) no_systemd=true ;;
    *) echo "Usage: $0 [--apply] [--no-systemd]" >&2; exit 2 ;;
  esac
  shift
done

hermes_root="$HOME/.hermes"
runtime_home="${HERMES_HOME:-$("$python_bin" -m hegi.bootstrap locate-home --hermes-root "$hermes_root")}"
v1_state="$hermes_root/hegi-watch"
echo "v1 watcher: $HOME/bin/hegi-memory-watch.py"
echo "v1 state: $v1_state"
echo "v2 runtime: $runtime_home"
echo "v2 config: $runtime_home/hegi/config.yaml"
if [[ "$apply" != true ]]; then
  echo "Diagnostic only. Re-run with --apply to stop v1, back up state, install v2, and start its daemon."
  exit 0
fi

pids=()
if [[ -f "$v1_state/loop.pid" ]]; then
  candidate="$(<"$v1_state/loop.pid")"
  [[ "$candidate" =~ ^[0-9]+$ ]] && pids+=("$candidate")
fi
while IFS= read -r candidate; do
  [[ "$candidate" =~ ^[0-9]+$ ]] && pids+=("$candidate")
done < <(pgrep -f "$HOME/bin/hegi-memory-watch-loop|$HOME/bin/hegi-memory-watch.py" || true)

for pid in "${pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$command_line" == *"hegi-memory-watch"* ]]; then
      kill -TERM "$pid"
      echo "Requested v1 watcher stop: pid=$pid"
    fi
  fi
done
for _ in {1..50}; do
  remaining=false
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && remaining=true
  done
  [[ "$remaining" == false ]] && break
  sleep 0.1
done
for pid in "${pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$command_line" == *"hegi-memory-watch"* ]]; then
      kill -KILL "$pid"
      echo "Forced stale v1 watcher stop after grace period: pid=$pid"
    fi
  fi
done

if [[ -d "$v1_state" ]]; then
  backup="$hermes_root/backups/hegi-watch-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$(dirname "$backup")"
  cp -a "$v1_state" "$backup"
  echo "Preserved v1 state: $backup"
fi

install_args=(--hermes-root "$hermes_root" --runtime-home "$runtime_home")
[[ "$no_systemd" == true ]] && install_args+=(--no-systemd)
"$repo_root/hegi/scripts/install.sh" "${install_args[@]}"
HERMES_HOME="$runtime_home" "$repo_root/hegi/scripts/stop.sh"
HERMES_HOME="$runtime_home" "$repo_root/hegi/scripts/start.sh" --send
echo "HEGI v2 migration complete; v1 stopped, configuration backed up, v2 enabled and daemon started."
