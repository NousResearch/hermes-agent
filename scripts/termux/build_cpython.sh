#!/usr/bin/env bash
# Stage the pinned bionic CPython (pm package `python`, linux-arm64-bionic
# lock row -- a TUR .deb) into <payload>/python/.
# Thin wrapper around the shared termux_pkg_build.sh; pm owns the pins.
#
# Usage: build_cpython.sh <payload-dir>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/termux_pkg_build.sh" python python "$@"
