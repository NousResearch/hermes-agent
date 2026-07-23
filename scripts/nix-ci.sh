#!/usr/bin/env bash

set -euo pipefail

case "${1:-check}" in
  check)
    nix flake check --print-build-logs
    ;;
  build)
    nix build .#default .#tui .#web .#desktop \
      --no-link --print-build-logs
    ;;
  *)
    echo "Usage: $0 [check|build]" >&2
    exit 2
    ;;
esac
