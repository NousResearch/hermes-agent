#!/usr/bin/env bash
# Gateway memory tracking script
# Captures cgroup memory stats to establish growth curve
# Output: machine-readable JSON for cron delivery or aggregation

CGROUP="/sys/fs/cgroup/system.slice/hermes-gateway.service"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

if [ ! -d "$CGROUP" ]; then
  echo '{"ts":"'"$TIMESTAMP"'","error":"cgroup_not_found"}'
  exit 1
fi

MEM_CURRENT=$(cat "$CGROUP/memory.current" 2>/dev/null || echo 0)
MEM_PEAK=$(cat "$CGROUP/memory.peak" 2>/dev/null || echo 0)
MEM_SWAP_CURRENT=$(cat "$CGROUP/memory.swap.current" 2>/dev/null || echo 0)

# Parse memory.stat key fields
ANON=$(grep -E '^anon ' "$CGROUP/memory.stat" | awk '{print $2}' 2>/dev/null || echo 0)
FILE=$(grep -E '^file ' "$CGROUP/memory.stat" | awk '{print $2}' 2>/dev/null || echo 0)
KERNEL=$(grep -E '^kernel ' "$CGROUP/memory.stat" | awk '{print $2}' 2>/dev/null || echo 0)

# System-wide memory
MEM_AVAIL=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
SWAP_USED=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
SWAP_FREE=$(grep SwapFree /proc/meminfo | awk '{print $2}')
SWAP_USED_BYTES=$(( (SWAP_USED - SWAP_FREE) * 1024 ))

# Gateway PID RSS
GATEWAY_PID=$(systemctl show hermes-gateway.service -p MainPID --value 2>/dev/null || echo 0)
GATEWAY_RSS=0
if [ "$GATEWAY_PID" -gt 1 ] 2>/dev/null; then
  GATEWAY_RSS=$(ps -o rss= -p "$GATEWAY_PID" 2>/dev/null | tr -d ' ' || echo 0)
  GATEWAY_RSS=$(( GATEWAY_RSS * 1024 ))
fi

cat <<JSON
{
  "ts": "$TIMESTAMP",
  "cgroup_current_bytes": $MEM_CURRENT,
  "cgroup_peak_bytes": $MEM_PEAK,
  "cgroup_anon_bytes": $ANON,
  "cgroup_file_cache_bytes": $FILE,
  "cgroup_kernel_bytes": $KERNEL,
  "cgroup_swap_bytes": $MEM_SWAP_CURRENT,
  "system_mem_avail_kb": $MEM_AVAIL,
  "system_swap_used_bytes": $SWAP_USED_BYTES,
  "gateway_rss_bytes": $GATEWAY_RSS
}
JSON
