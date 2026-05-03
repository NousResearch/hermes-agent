#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: helixa-name.sh <name>" >&2
  echo "Example: helixa-name.sh MyAgent" >&2
  exit 1
fi

"$(dirname "$0")/helixa-get.sh" "/api/v2/name/$1"
