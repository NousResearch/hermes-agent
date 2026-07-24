#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hermes_home="${HERMES_HOME:-$HOME/.hermes}"
v1_state="$HOME/.hermes/hegi-watch"
apply=false
if [[ "${1:-}" == "--apply" ]]; then
  apply=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--apply]" >&2
  exit 2
fi

echo "v1 watcher: $HOME/bin/hegi-memory-watch.py"
echo "v1 state: $v1_state"
echo "v2 config: $hermes_home/hegi/config.yaml"
if [[ "$apply" != true ]]; then
  echo "Diagnostic only. Re-run with --apply to stop v1 and install v2 config."
  exit 0
fi

if [[ -f "$v1_state/loop.pid" ]]; then
  pid="$(<"$v1_state/loop.pid")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid"
    echo "Requested v1 watcher stop: pid=$pid"
  fi
fi
if [[ -d "$v1_state" ]]; then
  backup="$HOME/.hermes/backups/hegi-watch-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$(dirname "$backup")"
  cp -a "$v1_state" "$backup"
  echo "Preserved v1 state: $backup"
fi
HERMES_HOME="$hermes_home" "$repo_root/hegi/scripts/install.sh"
