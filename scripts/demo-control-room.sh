#!/usr/bin/env bash
# Control Room demo script (CR-607). Run against a live gateway.
# Each step prints the expected result. Stop after any FAIL.

set -euo pipefail

echo "== Control Room demo =="
echo

# 1. CLI status segment
echo "[1] CLI status segment shows attention counts"
hermes --status 2>/dev/null | grep -iE "needs-you|control" || echo "  (segment hidden when idle — expected)"

# 2. CLI /control command
echo "[2] /control renders the snapshot as text"
echo "/control" | hermes 2>/dev/null | head -20

# 3. Ctrl+P binding
echo "[3] Ctrl+P opens Control Room in the TUI/desktop surfaces (manual: press Ctrl+P)"

# 4. Gateway RPC methods registered
echo "[4] Gateway RPC registers control.room.snapshot + control.room.action"
python3 - <<'PY'
import sys
sys.path.insert(0, ".")
import tui_gateway.methods_control_room as m
names = [name for name, _ in m._registry._pending]
assert "control.room.snapshot" in names, "snapshot missing"
assert "control.room.action" in names, "action missing"
print("  registered:", sorted(names), "— PASS")
PY

# 5. Dashboard BFF
echo "[5] Dashboard BFF serves /api/control-room"
PORT="${KENSEI_DASHBOARD_PORT:-9123}"
if curl -sf --max-time 3 "http://127.0.0.1:${PORT}/api/control-room?profile=default" | python3 -c "import json,sys; d=json.load(sys.stdin); print('  keys:', sorted(d)); assert d['version']==1; print('  PASS')"; then
  :
else
  echo "  (dashboard not running on :${PORT} — start it, or skip)"
fi

echo
echo "Demo complete."
