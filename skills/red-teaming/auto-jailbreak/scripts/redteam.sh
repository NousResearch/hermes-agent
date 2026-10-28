#!/usr/bin/env bash
# Auto-Jailbreak — run wrapper. Sources your config and runs the engine in the
# isolated env. Reads a JSON objective on stdin, prints the JSON result.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${AUTOJB_HOME:-$HOME/.auto-jailbreak}"
VENV="${AUTOJB_VENV:-$DATA/venv}"
CONFIG="${AUTOJB_CONFIG:-$DATA/config.env}"
[ -x "$VENV/bin/python" ] || { echo "Not installed. Run: bash \"$HERE/install.sh\"" >&2; exit 1; }
if [ -f "$CONFIG" ]; then set -a; . "$CONFIG"; set +a; fi
exec "$VENV/bin/python" "$HERE/attaque.py"
