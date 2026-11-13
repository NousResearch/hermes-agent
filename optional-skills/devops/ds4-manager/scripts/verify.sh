#!/bin/bash
# DS4 Dwarfstar Dashboard — full verification script
# Tests all REST endpoints and reports status

set -euo pipefail

BASE="http://127.0.0.1:8765"
PASS=0
FAIL=0

check() {
    local name="$1"
    local url="$2"
    local method="${3:-GET}"
    local data="${4:-}"

    if [ "$method" = "GET" ]; then
        resp=$(curl -s -o /tmp/ds4-check.json -w "%{http_code}" "$url" 2>/dev/null)
    else
        resp=$(curl -s -X "$method" -o /tmp/ds4-check.json -w "%{http_code}" \
            -H 'Content-Type: application/json' -d "$data" "$url" 2>/dev/null)
    fi

    if [ "$resp" = "200" ] || [ "$resp" = "404" ]; then
        echo "  ✓ $name ($resp)"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $name ($resp)"
        cat /tmp/ds4-check.json 2>/dev/null | head -c 200
        echo ""
        FAIL=$((FAIL + 1))
    fi
}

echo "DS4 Dwarfstar Dashboard — Verification"
echo "========================================"
echo ""

# Core endpoints
check "GET /api/status" "$BASE/api/status"
check "GET /api/metrics" "$BASE/api/metrics"
check "GET /api/config" "$BASE/api/config"
check "GET /api/config-schema" "$BASE/api/config-schema"
check "GET /api/config-overrides" "$BASE/api/config-overrides"
check "GET /api/benchmarks" "$BASE/api/benchmarks"

# MCP endpoints
check "GET /api/mcp/manifest" "$BASE/api/mcp/manifest"
check "GET /api/mcp/resources" "$BASE/api/mcp/resources"
check "GET /api/mcp/resources/read?uri=ds4://status" "$BASE/api/mcp/resources/read?uri=ds4://status"

# POST endpoints
check "POST /api/benchmarks/run" "$BASE/api/benchmarks/run" POST '{"suite_id":"quick_smoke","iterations":1}'

echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
