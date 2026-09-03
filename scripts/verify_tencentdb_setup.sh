#!/usr/bin/env bash
# Quick TencentDB setup verification

set -euo pipefail

echo "=== TencentDB Memory Provider Setup Verification ==="
echo ""

# 1. Check SDK
echo "1. Checking tencentdb_agent_memory SDK..."
if ~/.hermes/hermes-agent/venv/bin/python -c "import tencentdb_agent_memory; print('   OK:', tencentdb_agent_memory.__version__)" 2>/dev/null; then
    echo "   ✓ SDK installed"
else
    echo "   ✗ SDK NOT installed"
    echo "   Run: ~/.hermes/hermes-agent/venv/bin/python -m pip install ./tencentdb_agent_memory_sdk_python-0.1.0-py3-none-any.whl"
    exit 1
fi

# 2. Check provider symlink
echo ""
echo "2. Checking provider symlink..."
PROVIDER_DIR="/Users/louisling/.hermes/hermes-agent/plugins/memory/memory_tencentdb_v2"
if [[ -L "$PROVIDER_DIR" ]]; then
    TARGET=$(readlink "$PROVIDER_DIR")
    echo "   ✓ Symlink exists: $TARGET"
elif [[ -d "$PROVIDER_DIR" ]]; then
    echo "   ✓ Directory exists (copied)"
else
    echo "   ✗ Provider not found at $PROVIDER_DIR"
    exit 1
fi

# 3. Check config
echo ""
echo "3. Checking Hermes config..."
CONFIG_FILE="$HOME/.hermes/config.yaml"
if grep -q "memory_tencentdb_v2" "$CONFIG_FILE" 2>/dev/null; then
    echo "   ✓ Provider enabled in config.yaml"
else
    echo "   ✗ Provider NOT enabled in config.yaml"
    echo "   Add to ~/.hermes/config.yaml:"
    echo "   memory:"
    echo "     provider: memory_tencentdb_v2"
    exit 1
fi

# 4. Check env vars
echo ""
echo "4. Checking environment variables..."
if [[ -n "${TDAI_MEMORY_ENDPOINT:-}" ]]; then
    echo "   ✓ TDAI_MEMORY_ENDPOINT=$TDAI_MEMORY_ENDPOINT"
else
    echo "   ⚠ TDAI_MEMORY_ENDPOINT not set (using default http://127.0.0.1:8420)"
fi

if [[ -n "${TDAI_MEMORY_API_KEY:-}" ]]; then
    echo "   ✓ TDAI_MEMORY_API_KEY=***"
else
    echo "   ⚠ TDAI_MEMORY_API_KEY not set (using default 'local')"
fi

if [[ -n "${TDAI_MEMORY_SERVICE_ID:-}" ]]; then
    echo "   ✓ TDAI_MEMORY_SERVICE_ID=$TDAI_MEMORY_SERVICE_ID"
else
    echo "   ⚠ TDAI_MEMORY_SERVICE_ID not set (using default 'default')"
fi

# 5. Check Gateway connectivity
echo ""
echo "5. Checking Gateway connectivity..."
ENDPOINT="${TDAI_MEMORY_ENDPOINT:-http://127.0.0.1:8420}"
if curl -s -f --max-time 3 "$ENDPOINT/health" >/dev/null 2>&1; then
    echo "   ✓ Gateway reachable at $ENDPOINT"
elif curl -s -f --max-time 3 "$ENDPOINT/v2/health" >/dev/null 2>&1; then
    echo "   ✓ Gateway reachable at $ENDPOINT (v2)"
else
    echo "   ⚠ Gateway NOT reachable at $ENDPOINT"
    echo "   Start local gateway: docker run -d --name tdai-gateway -p 8420:8420 ..."
fi

# 6. Check skill exists
echo ""
echo "6. Checking Ruflo skill..."
SKILL_DIR="$HOME/.hermes/skills/ruflo-workflows"
if [[ -d "$SKILL_DIR" ]]; then
    echo "   ✓ Skill installed at $SKILL_DIR"
else
    echo "   ✗ Skill NOT found"
    echo "   Run: hermes skills sync  (after adding to repo skills/)"
fi

echo ""
echo "=== Setup verification complete ==="
echo ""
echo "Next steps:"
echo "  1. If any ✗ above, fix and re-run this script"
echo "  2. Push skill to TencentDB: python3 scripts/push_ruflo_to_tencentdb.py"
echo "  3. Test in Hermes: hermes chat → 'Use tdai_memory_search to search for ruflo'"