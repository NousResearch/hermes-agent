#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: helixa-stake.sh <id>" >&2
  echo "Example: helixa-stake.sh 1" >&2
  exit 1
fi

"$(dirname "$0")/helixa-get.sh" "/api/v2/stake/$1"
