#!/bin/sh
set -eu

# This Flatpak is the Electron client only. It connects to the native Hermes
# backend that `hermes desktop` starts on localhost; it never carries a Python
# CLI, gateway, plugins, credentials, or a second Hermes state directory.
export HERMES_DESKTOP_FLATPAK=1
export HERMES_HOME="${HOME}/.hermes"
# Ensure the native backend can find system tools
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"
cd "$HOME"
exec /app/bin/zypak-wrapper.sh /app/share/hermes-desktop/Hermes "$@"
