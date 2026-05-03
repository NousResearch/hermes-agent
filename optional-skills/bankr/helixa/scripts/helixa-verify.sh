#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: helixa-verify.sh <id> <json_body> <siwa_auth>" >&2
  echo "Example: helixa-verify.sh 1 '{\"handle\":\"@myagent\"}' 'Bearer addr:ts:sig'" >&2
  echo "" >&2
  echo "Requires SIWA auth." >&2
  exit 1
fi

"$(dirname "$0")/helixa-post.sh" "/api/v2/agent/$1/verify" "$2" "$3"
