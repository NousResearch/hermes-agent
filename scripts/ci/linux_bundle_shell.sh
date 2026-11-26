#!/usr/bin/env bash
# Actions accepts matrix expressions in job env, not in a step's shell field.
set -euo pipefail
case "${LINUX_BUNDLE_TARGET:?missing Linux bundle target}" in
  linux-x64) exec bash scripts/ci/linux_x64_userland.sh "$1" "$2" ;;
  linux-arm64) exec bash --noprofile --norc -eo pipefail "$2" ;;
  *) printf 'Unsupported Linux bundle target: %s\n' "$LINUX_BUNDLE_TARGET" >&2; exit 2 ;;
esac
