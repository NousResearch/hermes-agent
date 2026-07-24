#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hermes_home="${HERMES_HOME:-$HOME/.hermes}"
state_dir="$hermes_home/hegi"
mkdir -p "$state_dir"

cd "$repo_root"
nohup python -m hegi daemon "$@" >>"$state_dir/daemon.log" 2>&1 &
launcher_pid=$!
sleep 1
if ! kill -0 "$launcher_pid" 2>/dev/null; then
  echo "HEGI daemon failed to start; inspect $state_dir/daemon.log" >&2
  exit 1
fi
echo "HEGI daemon started: pid=$launcher_pid"
