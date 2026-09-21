#!/usr/bin/env bash
# Orchestrator memory tracking script
# Captures RSS of the orchestrator Node.js process
# Output: machine-readable JSON for cron delivery or aggregation

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Find orchestrator PID (look for the specific command)
ORCHESTRATOR_PID=$(pgrep -f "node --experimental-require-module ./dist/apps/orchestrator/src/main.js" | head -1)

if [ -z "$ORCHESTRATOR_PID" ]; then
  echo '{"ts":"'"$TIMESTAMP"'","error":"orchestrator_not_found"}'
  exit 1
fi

# Get RSS in KB from ps, convert to bytes
ORCH_RSS_KB=$(ps -o rss= -p "$ORCHESTRATOR_PID" 2>/dev/null | tr -d ' ' || echo 0)
ORCH_RSS_BYTES=$(( ORCH_RSS_KB * 1024 ))

# System-wide memory
MEM_AVAIL=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
SWAP_USED=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
SWAP_FREE=$(grep SwapFree /proc/meminfo | awk '{print $2}')
SWAP_USED_BYTES=$(( (SWAP_USED - SWAP_FREE) * 1024 ))

cat <<JSON
{
  "ts": "$TIMESTAMP",
  "orchestrator_pid": $ORCHESTRATOR_PID,
  "orchestrator_rss_bytes": $ORCH_RSS_BYTES,
  "system_mem_avail_kb": $MEM_AVAIL,
  "system_swap_used_bytes": $SWAP_USED_BYTES
}
JSON