#!/usr/bin/env bash
# Hermes prompt-optimizer installer (macOS / Linux / WSL)
# Usage: bash install.sh
# Deploys plugin.js to <hermes home>/desktop-plugins/prompt-optimizer/ and the
# desktop app hot-loads it within seconds (Ctrl/Cmd+K -> "Reload desktop
# plugins" if the button does not appear). Idempotent — safe to re-run.
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_FILE="$SOURCE_DIR/plugin.js"

if [ ! -f "$SOURCE_FILE" ]; then
  echo "[ERROR] plugin.js not found next to this script." >&2
  exit 1
fi

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
DEST_DIR="$HERMES_HOME/desktop-plugins/prompt-optimizer"
mkdir -p "$DEST_DIR"
cp -f "$SOURCE_FILE" "$DEST_DIR/plugin.js"

SIZE="$(wc -c < "$DEST_DIR/plugin.js")"
echo "[OK] deployed: $DEST_DIR/plugin.js ($SIZE bytes)"
echo "     The desktop app will load the plugin within seconds."
echo "     If the button does not appear, run 'Reload desktop plugins' from Ctrl/Cmd+K."
