#!/bin/bash
# Checkmk Alert Handler — forwards alerts to Hermes enrichment pipeline
# Location: /omd/sites/cmk/local/bin/checkmk-webhook.sh
#
# Checkmk invokes this script with the alert payload as stdin
# (application/x-www-form-urlencoded). We convert to JSON and POST
# to the Hermes Flask receiver at http://192.168.1.8:5001/alerts

HERMES_URL="http://192.168.1.8:5001/alerts"

# Read stdin (form-encoded key=value pairs)
PAYLOAD=$(cat)

# Convert form-encoded to JSON via Python, then POST
python3 -c "
import sys, urllib.parse, json

form = urllib.parse.parse_qsl(sys.stdin.read())
data = {}
for k, v in form:
    if k == 'context':
        try:
            ctx = json.loads(v)
            data['context'] = ctx
            # Flatten context keys to root for easier Hermes parsing
            for ck, cv in ctx.items():
                if ck not in data:
                    data[ck] = cv
        except Exception:
            data[k] = v
    else:
        data[k] = v

payload = json.dumps(data)
print(payload)
" <<< "$PAYLOAD" \
  | curl -s -X POST "${HERMES_URL}" \
      -H "Content-Type: application/json" \
      -d @-
