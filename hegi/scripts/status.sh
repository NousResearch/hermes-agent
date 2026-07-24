#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin=""
for candidate in "$repo_root/.venv/bin/python" "$repo_root/venv/bin/python" "$(command -v python3 || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then python_bin="$candidate"; break; fi
done
[[ -n "$python_bin" ]] || { echo "Python runtime not found" >&2; exit 1; }
runtime_home="${HERMES_HOME:-$("$python_bin" -m hegi.bootstrap locate-home)}"
cd "$repo_root"
HERMES_HOME="$runtime_home" "$python_bin" -m hegi status "$@"
