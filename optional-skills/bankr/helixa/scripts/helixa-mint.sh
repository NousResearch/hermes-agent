#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: helixa-mint.sh <json_body> <siwa_auth>" >&2
  echo "Example: helixa-mint.sh '{\"name\":\"MyAgent\",\"framework\":\"openclaw\"}' 'Bearer addr:ts:sig'" >&2
  echo "" >&2
  echo "Requires SIWA auth + x402 payment (\$1 USDC)." >&2
  echo "For automatic x402 handling, use the Node.js x402 SDK instead." >&2
  exit 1
fi

"$(dirname "$0")/helixa-post.sh" "/api/v2/mint" "$1" "$2"
