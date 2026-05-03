#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: helixa-cred.sh <id>" >&2
  echo "Example: helixa-cred.sh 1" >&2
  exit 1
fi

"$(dirname "$0")/helixa-get.sh" "/api/v2/agent/$1/cred-breakdown"
