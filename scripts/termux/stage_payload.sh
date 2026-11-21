#!/usr/bin/env bash
# Stage pinned bionic pm packages, then the runtime libraries they link, into a
# Termux payload. Each package lands in <payload-dir>/<package>/.
#
# Usage: stage_payload.sh <payload-dir> <package>...
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PAYLOAD="$1"
shift
for package in "$@"; do
  bash "$HERE/termux_pkg_build.sh" "$package" "$package" "$PAYLOAD"
done
python3 "$HERE/stage_runtime_libs.py" "$PAYLOAD"
