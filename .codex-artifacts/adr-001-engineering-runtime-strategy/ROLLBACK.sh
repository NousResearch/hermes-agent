#!/usr/bin/env bash
set -euo pipefail

root="${1:-.}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_file="$script_dir/DIFF_FILE.patch"

patch --batch --silent -R -p1 -d "$root" < "$patch_file"
printf 'ROLLBACK_RESULT=restored:%s/docs/decisions/ADR-001-engineering-runtime-strategy.md\n' "$root"
