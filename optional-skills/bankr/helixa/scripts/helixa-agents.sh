#!/usr/bin/env bash
set -euo pipefail

limit="${1:-20}"
offset="${2:-0}"

"$(dirname "$0")/helixa-get.sh" "/api/v2/agents" "limit=$limit&offset=$offset"
