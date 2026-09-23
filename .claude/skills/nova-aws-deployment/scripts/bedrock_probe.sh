#!/usr/bin/env bash
# Minimal read-only-ish Bedrock invocation probe (costs a few tokens). Same request from every
# vantage point so results are comparable.
#   bedrock_probe.sh <model-or-profile-id> <region>
# Run locally, and on the runtime via SSM (it uses whatever credentials the environment provides —
# on EC2 that is the instance role; never pass keys to it).
set -uo pipefail
MODEL="${1:?model or inference profile id}"; REGION="${2:?region}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ID=$(aws sts get-caller-identity --query Arn --output text 2>&1) || { echo "credentials: FAIL ($ID)"; exit 1; }
echo "caller: $ID"
OUT=$(aws bedrock-runtime converse --region "$REGION" --model-id "$MODEL" \
  --messages '[{"role":"user","content":[{"text":"Reply with the single word: ok"}]}]' \
  --inference-config '{"maxTokens":10}' --output json 2>&1)
if [ $? -eq 0 ]; then
  echo "bedrock: PASS"; echo "$OUT" | python3 -c 'import json,sys;d=json.load(sys.stdin);print("stopReason:",d.get("stopReason"),"usage:",d.get("usage"))'
  exit 0
fi
echo "bedrock: FAIL"; echo "$OUT" | head -5
echo "$OUT" | python3 "$HERE/classify_bedrock_error.py"
exit 2
