#!/usr/bin/env bash
# verify_composio.sh — End-to-end Composio auth + connectivity check.
#
# Checks: CLI login (whoami), MCP endpoint (curl initialize), and a sample tool search.
# Exit codes: 0 all ok / 1 CLI not authed / 2 MCP not authed / 3 both failed
set -euo pipefail

CLI_OK=0
MCP_OK=0

echo "=== Composio CLI (ak_* path) ==="
if ! command -v composio >/dev/null 2>&1; then
  echo "✗ composio CLI not on PATH. Install: npm i -g composio-core @composio/cli"
else
  if composio whoami >/dev/null 2>&1; then
    echo "✓ CLI authenticated"
    composio whoami 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"  email={d.get(\"email\",\"?\")} org={d.get(\"current_org_name\",\"?\")}")' 2>/dev/null || true
    CLI_OK=1
  else
    echo "✗ CLI not authenticated. Run: composio login"
  fi
fi

echo ""
echo "=== Composio MCP (ck_* path) ==="
CK="${COMPOSIO_CONSUMER_KEY:-}"
if [ -z "$CK" ]; then
  echo "✗ COMPOSIO_CONSUMER_KEY not set. Get ck_* from dashboard → Connect Settings → Sessions & API Key"
else
  echo "  consumer key prefix: ${CK:0:5}…"
  resp=$(curl -sS --max-time 15 -X POST "https://connect.composio.dev/mcp" \
    -H "Content-Type: application/json" \
    -H "x-consumer-api-key: $CK" \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"verify","version":"1.0"}}}' 2>&1 || true)
  if echo "$resp" | grep -q '"result"'; then
    echo "✓ MCP endpoint reachable and authenticated"
    MCP_OK=1
  elif echo "$resp" | grep -q 'Authorization required'; then
    echo "✗ MCP rejected the consumer key (revoked or wrong). Reason: $resp"
  else
    echo "✗ MCP request failed: $resp"
  fi
fi

echo ""
echo "=== Sample tool search (CLI) ==="
if [ "$CLI_OK" -eq 1 ]; then
  composio search "send an email" --limit 3 2>/dev/null | python3 -c '
import json,sys
try:
  d=json.load(sys.stdin)
  tools=d if isinstance(d,list) else d.get("tools",d.get("data",[]))
  for t in tools[:3]:
    print(f"  - {t.get(\"slug\",\"?\")}: {t.get(\"description\",\"\")[:80]}")
  print(f"  ({len(tools)} tools returned)")
except Exception as e:
  print(f"  (could not parse search output: {e})")
' 2>/dev/null || echo "  (search failed)"
fi

echo ""
if [ "$CLI_OK" -eq 1 ] && [ "$MCP_OK" -eq 1 ]; then
  echo "✅ Both CLI and MCP are ready."
  exit 0
elif [ "$CLI_OK" -eq 1 ]; then
  echo "⚠ CLI ok, MCP not ready (fallback path unavailable)."
  exit 2
elif [ "$MCP_OK" -eq 1 ]; then
  echo "⚠ MCP ok, CLI not ready."
  exit 1
else
  echo "❌ Neither CLI nor MCP is ready."
  exit 3
fi
