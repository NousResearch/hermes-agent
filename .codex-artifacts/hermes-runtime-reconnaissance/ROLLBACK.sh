#!/usr/bin/env bash
set -euo pipefail

target="${1:-docs/research/hermes-runtime-reconnaissance.md}"
if [[ ! -f "$target" ]]; then
  printf 'ROLLBACK_RESULT=already_absent:%s\n' "$target"
  exit 0
fi
rm -- "$target"
printf 'ROLLBACK_RESULT=removed:%s\n' "$target"