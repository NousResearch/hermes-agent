#!/usr/bin/env bash
# presence_notify.sh — budgeted ambient alerts (max N/day to avoid noise).
# Usage: presence_notify.sh "message"
set -u
HOME_DIR="$HOME"
BUDGET_FILE="$HOME_DIR/.hermes/presence_budget.json"
MAX_PER_DAY=3
MSG="${1:-Hermes Agent notification}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

today=$(date +%Y-%m-%d)
count=$(python3 -c "
import json,os
p='$BUDGET_FILE'
try:
    d=json.load(open(p))
    print(d.get('count',0) if d.get('day')=='$today' else 0)
except Exception:
    print(0)
")

if [ "$count" -ge "$MAX_PER_DAY" ]; then
    mkdir -p "$HOME_DIR/.hermes/logs"
    echo "$(date '+%F %T') presence: BUDGET EXHAUSTED ($count/$MAX_PER_DAY) — skipped: $MSG" >> "$HOME_DIR/.hermes/logs/presence.log"
    exit 0
fi

if python3 "$SCRIPT_DIR/hermes_presence.py" "$MSG"; then
    python3 -c "
import json,os
os.makedirs(os.path.dirname('$BUDGET_FILE'), exist_ok=True)
json.dump({'day':'$today','count':$count+1}, open('$BUDGET_FILE','w'))
"
    mkdir -p "$HOME_DIR/.hermes/logs"
    echo "$(date '+%F %T') presence: sent ($((count+1))/$MAX_PER_DAY): $MSG" >> "$HOME_DIR/.hermes/logs/presence.log"
fi
