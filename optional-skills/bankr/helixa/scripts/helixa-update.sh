#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: helixa-update.sh <id> <json_body> <siwa_auth>" >&2
  echo "Example: helixa-update.sh 1 '{\"traits\":[...]}' 'Bearer addr:ts:sig'" >&2
  echo "" >&2
  echo "Requires SIWA auth + x402 payment (\$1 USDC)." >&2
  exit 1
fi

"$(dirname "$0")/helixa-post.sh" "/api/v2/agent/$1/update" "$2" "$3"
