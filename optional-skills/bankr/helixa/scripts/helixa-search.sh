#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: helixa-search.sh <query>" >&2
  echo "Example: helixa-search.sh clawdbot" >&2
  exit 1
fi

query=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$1'))" 2>/dev/null || echo "$1")
"$(dirname "$0")/helixa-get.sh" "/api/v2/agents" "search=$query"
