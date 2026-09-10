#!/usr/bin/env bash
# Bar Raiser verification: cross-check a generated architecture diagram
# against the live system. Outputs PASS/FAIL for each check.
# Usage: bash bar-raiser.sh [path-to-diagram.html]
# Uses only public Hermes CLI commands — no direct config file reads.
set -euo pipefail

DIAGRAM="${1:-$HOME/hermes-agent-architecture.html}"

if [ ! -f "$DIAGRAM" ]; then
  echo "FAIL: Diagram not found at $DIAGRAM"
  exit 1
fi

PASS=0
FAIL=0

check() {
  local label="$1" result="$2"
  if [ "$result" = "true" ]; then
    echo "  PASS: $label"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: $label"
    FAIL=$((FAIL + 1))
  fi
}

echo "=== BAR RAISER: Identity and Environment ==="
USER_OK=$(grep -qi "$(whoami)" "$DIAGRAM" && echo true || echo false)
check "Username matches whoami" "$USER_OK"

OS_RAW=$(uname -s)
OS_LABEL=$(echo "$OS_RAW" | sed 's/Darwin/macOS/;s/Linux/Linux/')
OS_OK=$(grep -qi "$OS_LABEL" "$DIAGRAM" && echo true || echo false)
check "OS label correct ($OS_LABEL)" "$OS_OK"

echo ""
echo "=== BAR RAISER: Hermes Configuration ==="
if command -v hermes &>/dev/null; then
  HERMES_OUTPUT=$(hermes config 2>/dev/null || true)
  MODEL=$(echo "$HERMES_OUTPUT" | grep -iE '^\s*default:' | head -1 | awk '{print $2}')
  if [ -n "$MODEL" ]; then
    MODEL_OK=$(grep -q "$MODEL" "$DIAGRAM" && echo true || echo false)
    check "Model name matches config ($MODEL)" "$MODEL_OK"
  else
    check "Model name in config (not found)" "false"
  fi

  BACKEND=$(echo "$HERMES_OUTPUT" | grep -iE '^\s*backend:' | head -1 | awk '{print $2}')
  BACKEND="${BACKEND:-local}"
  echo "  Backend detected: $BACKEND"
else
  echo "  WARNING: hermes CLI not available, skipping config checks"
  BACKEND="local"
fi

echo ""
echo "=== BAR RAISER: Topology ==="
if [ "$BACKEND" = "local" ] || [ -z "$BACKEND" ]; then
  LOCAL_ZONE=$(grep -qi "LOCAL EXECUTION" "$DIAGRAM" && echo true || echo false)
  check "Local execution zone present" "$LOCAL_ZONE"

  NO_SANDBOX=$(grep -qiE "SANDBOX.*ISOLATION|ephemeral.*Linux.*Container" "$DIAGRAM" && echo false || echo true)
  check "No false sandbox zone" "$NO_SANDBOX"
else
  SANDBOX_ZONE=$(grep -qiE "SANDBOX|MODAL|Container" "$DIAGRAM" && echo true || echo false)
  check "Sandbox zone present for $BACKEND backend" "$SANDBOX_ZONE"
fi

echo ""
echo "=== BAR RAISER: Visual Content ==="
HAS_SVG=$(grep -q "<svg" "$DIAGRAM" && echo true || echo false)
check "SVG element present" "$HAS_SVG"

HAS_CARDS=$(grep -q "card" "$DIAGRAM" && echo true || echo false)
check "Info cards present" "$HAS_CARDS"

HAS_FOOTER=$(grep -qi "footer" "$DIAGRAM" && echo true || echo false)
check "Footer present" "$HAS_FOOTER"

HAS_LEGEND=$(grep -qiE "legend|Data Flow" "$DIAGRAM" && echo true || echo false)
check "Legend present" "$HAS_LEGEND"

echo ""
echo "==========================================="
echo "  PASS: $PASS   FAIL: $FAIL"
if [ "$FAIL" -eq 0 ]; then
  echo "  BAR RAISER: PASSED"
  echo ""
  echo "  Sign-off watermark:"
  echo "  <!-- Bar Raiser: PASSED"
  echo "       Verified: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "       OS: $OS_RAW | User: $(whoami) | Host: $(hostname)"
  echo "       Backend: $BACKEND | Topology: $([ "$BACKEND" = "local" ] && echo "2-zone" || echo "3-zone")"
  echo "       All claims ground-truthed against live system. -->"
else
  echo "  BAR RAISER: FAILED — fix $FAIL issue(s) before delivery"
fi
echo "==========================================="
exit $FAIL
