#!/usr/bin/env bash
# Boot a bare SuperCollider audio server (scsynth) listening for OSC on UDP.
#
# This is the SuperCollider analog of "open the receiver patch first": once
# scsynth is up, the Python client speaks OSC to it directly -- no sclang, no
# IDE. Leave it running, then drive it with sc_examples.py / sc_client.py.
#
#   ./templates/boot_scsynth.sh            # boot on UDP :57110 (default)
#   PORT=57110 ./templates/boot_scsynth.sh # explicit port
#
# Stop it with Ctrl-C, or from another shell: python3 sc_client.py --quit
set -euo pipefail

PORT="${PORT:-57110}"

# 1. Locate the scsynth binary and its UGen plugins.
if command -v scsynth >/dev/null 2>&1; then
    SCSYNTH="$(command -v scsynth)"
    PLUGINS=""   # a PATH install usually finds its own plugins
else
    APP="/Applications/SuperCollider.app/Contents/Resources"
    SCSYNTH="$APP/scsynth"
    PLUGINS="$APP/plugins"
fi

if [ ! -x "$SCSYNTH" ]; then
    echo "scsynth not found. Install SuperCollider:" >&2
    echo "  macOS:  brew install --cask supercollider" >&2
    echo "  Linux:  sudo apt install supercollider-server" >&2
    exit 1
fi

echo "Booting scsynth on UDP :$PORT  ($SCSYNTH)"
echo "Stop with Ctrl-C, or: python3 sc_client.py --quit"

# -u <port>  : listen for OSC on this UDP port
# -U <path>  : where to load UGen plugins from (app-bundle build needs this)
if [ -n "$PLUGINS" ]; then
    exec "$SCSYNTH" -u "$PORT" -U "$PLUGINS"
else
    exec "$SCSYNTH" -u "$PORT"
fi
