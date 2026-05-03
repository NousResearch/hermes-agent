#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: helixa-post.sh <path> <json_body> [auth_header]" >&2
  echo "Example: helixa-post.sh /api/v2/mint '{\"name\":\"MyAgent\"}' 'Bearer addr:ts:sig'" >&2
  exit 1
fi

path="$1"
body="$2"
auth="${3-}"

if [[ "$path" != /* ]]; then
  echo "helixa-post.sh: path must start with /" >&2
  exit 1
fi

base="${HELIXA_BASE_URL:-https://api.helixa.xyz}"
url="$base$path"

tmp_body=$(mktemp)
trap 'rm -f "$tmp_body"' EXIT

auth_args=()
if [ -n "$auth" ]; then
  auth_args=(-H "Authorization: $auth")
fi

http_code=$(curl -sS --connect-timeout 10 --max-time 30 -X POST \
  -H "User-Agent: helixa-skill/1.0" \
  -H "Content-Type: application/json" \
  "${auth_args[@]}" \
  -d "$body" \
  -w '%{http_code}' \
  -o "$tmp_body" \
  "$url") || {
  echo "helixa-post.sh: curl transport error (exit $?)" >&2
  exit 1
}

if [[ "$http_code" =~ ^2 ]]; then
  cat "$tmp_body"
  exit 0
fi

echo "helixa-post.sh: HTTP $http_code error" >&2
cat "$tmp_body" >&2
exit 1
