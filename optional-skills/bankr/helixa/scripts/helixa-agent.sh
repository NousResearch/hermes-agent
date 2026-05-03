#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: helixa-agent.sh <id>" >&2
  echo "Example: helixa-agent.sh 1" >&2
  exit 1
fi

"$(dirname "$0")/helixa-get.sh" "/api/v2/agent/$1"
