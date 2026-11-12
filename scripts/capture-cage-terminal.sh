#!/usr/bin/env bash
set -euo pipefail

out=${1:?usage: $0 OUTPUT.png [command...]}
shift
mkdir -p "$(dirname "$out")"

# This runs inside `cage --`, after Ghostty has already opened the isolated
# Wayland display. Capture before this script exits so Cage keeps the surface
# alive long enough for grim to see it.
if (($#)); then
  "$@"
fi
if ! command -v grim >/dev/null 2>&1; then
  echo "grim is required inside the Cage client session" >&2
  exit 127
fi
grim "$out"
