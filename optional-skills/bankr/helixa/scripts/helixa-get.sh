#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: helixa-get.sh <path> [query]" >&2
  echo "Example: helixa-get.sh /api/v2/stats" >&2
  exit 1
fi

path="$1"
query="${2-}"

if [[ "$path" != /* ]]; then
  echo "helixa-get.sh: path must start with /" >&2
  exit 1
fi

base="${HELIXA_BASE_URL:-https://api.helixa.xyz}"
url="$base$path"
if [ -n "$query" ]; then
  url="$url?$query"
fi

tmp_body=$(mktemp)
trap 'rm -f "$tmp_body"' EXIT

max_attempts=3
base_delay=2

for (( attempt=1; attempt<=max_attempts; attempt++ )); do
  http_code=$(curl -sS --connect-timeout 10 --max-time 30 \
    -H "User-Agent: helixa-skill/1.0" \
    -w '%{http_code}' \
    -o "$tmp_body" \
    "$url") || {
    echo "helixa-get.sh: curl transport error (exit $?)" >&2
    exit 1
  }

  if [[ "$http_code" =~ ^2 ]]; then
    cat "$tmp_body"
    exit 0
  fi

  if [ "$http_code" = "429" ] && [ "$attempt" -lt "$max_attempts" ]; then
    delay=$(( base_delay * (1 << (attempt - 1)) ))
    echo "helixa-get.sh: 429 rate limited, retrying in ${delay}s (attempt $attempt/$max_attempts)" >&2
    sleep "$delay"
    continue
  fi

  if [[ "$http_code" =~ ^5 ]] && [ "$attempt" -lt "$max_attempts" ]; then
    delay=$(( base_delay * (1 << (attempt - 1)) ))
    echo "helixa-get.sh: HTTP $http_code server error, retrying in ${delay}s (attempt $attempt/$max_attempts)" >&2
    sleep "$delay"
    continue
  fi

  echo "helixa-get.sh: HTTP $http_code error" >&2
  cat "$tmp_body" >&2
  exit 1
done
