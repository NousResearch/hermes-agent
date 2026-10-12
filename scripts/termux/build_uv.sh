#!/usr/bin/env bash
# Stage the pinned bionic uv (pm package `uv`, linux-arm64-bionic lock row
# -- the termux-main uv .deb) into <payload>/uv/. Same pm-consumer shape
# as build_cpython/build_node: pm owns the pin, the download, the hardened
# extraction, and the file-evidence verify. No PATH install -- hermes's
# resolvers compose the payload env at runtime (desktop payload model).
#
# Usage: build_uv.sh <payload-dir>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/termux_pkg_build.sh" uv uv "$@"
