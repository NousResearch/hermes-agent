#!/usr/bin/env bash
# Check an agent's Cred Score
# Usage: ./check-cred.sh <agent_id>

AGENT_ID="${1:?Usage: check-cred.sh <agent_id>}"
curl -s "https://api.helixa.xyz/api/v2/cred/${AGENT_ID}" | python3 -m json.tool 2>/dev/null || cat
