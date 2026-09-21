#!/usr/bin/env bash
# gateway-set-memory-limits.sh
# Sets MemoryMax and MemoryHigh on hermes-gateway.service
# Dry-run: echo the proposed changes without applying
# Apply: pass --apply flag

DRY_RUN=true
if [ "$1" = "--apply" ]; then
  DRY_RUN=false
fi

UNIT="hermes-gateway.service"
MEMORY_MAX="5G"
MEMORY_HIGH="4G"

echo "=== Proposed changes for $UNIT ==="
echo "MemoryMax: $MEMORY_MAX (hard limit)"
echo "MemoryHigh: $MEMORY_HIGH (soft throttle)"
echo ""
echo "Current:"
systemctl show "$UNIT" -p MemoryMax,MemoryHigh | sed 's/^/  /'
echo ""
PEAK=$(cat /sys/fs/cgroup/system.slice/hermes-gateway.service/memory.peak 2>/dev/null || echo 0)
CURR=$(cat /sys/fs/cgroup/system.slice/hermes-gateway.service/memory.current 2>/dev/null || echo 0)
echo "Peak cgroup memory last 12h: $(echo "scale=1; $PEAK/1073741824" | bc 2>/dev/null || echo '?') GB"
echo "Current cgroup: $(echo "scale=1; $CURR/1073741824" | bc 2>/dev/null || echo '?') GB"
echo "System available: $(grep MemAvailable /proc/meminfo | awk '{printf "%.1f GB", $2/1048576}')"
echo ""

if [ "$DRY_RUN" = true ]; then
  echo "[DRY RUN] No changes applied."
  echo "Manual:"
  echo "  mkdir -p /etc/systemd/system/$UNIT.d/"
  echo "  cat <<'EOF' | sudo tee /etc/systemd/system/$UNIT.d/99-memory-limits.conf"
  echo "[Service]"
  echo "MemoryMax=$MEMORY_MAX"
  echo "MemoryHigh=$MEMORY_HIGH"
  echo "EOF"
  echo "  sudo systemctl daemon-reload && sudo systemctl restart $UNIT"
else
  sudo mkdir -p /etc/systemd/system/${UNIT}.d/
  cat > /tmp/99-memory-limits.conf <<CONF
[Service]
MemoryMax=$MEMORY_MAX
MemoryHigh=$MEMORY_HIGH
CONF
  sudo mv /tmp/99-memory-limits.conf /etc/systemd/system/${UNIT}.d/99-memory-limits.conf
  sudo systemctl daemon-reload
  echo "Limits written. Restart to apply: sudo systemctl restart $UNIT"
fi
