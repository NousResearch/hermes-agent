#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

need_bin jq
# Extract queue balance from unified veil balance output
veil_cli balance --quiet "$@" | jq '.queue'
