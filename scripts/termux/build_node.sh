#!/usr/bin/env bash
# Stage the pinned bionic node (pm package `node`, linux-arm64-bionic lock
# row -- the termux-main nodejs .deb) into <payload>/node/.
# Thin wrapper around the shared termux_pkg_build.sh; pm owns the pins.
#
# Usage: build_node.sh <payload-dir>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/termux_pkg_build.sh" node node "$@"
