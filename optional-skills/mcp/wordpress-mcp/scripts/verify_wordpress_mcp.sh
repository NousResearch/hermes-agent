#!/usr/bin/env bash
# verify_wordpress_mcp.sh — End-to-end WordPress MCP connectivity check.
#
# Checks both endpoints (mcp-adapter and AI Engine) by sending
# JSON-RPC initialize + tools/list and reporting tool counts.
#
# Usage:
#   bash verify_wordpress_mcp.sh --url https://yourdomain.com \
#     --wp-user admin --wp-app-password "..." \
#     --ai-engine-token "..."
#
#   # Only one path:
#   bash verify_wordpress_mcp.sh --url https://yourdomain.com --only mcp-adapter \
#     --wp-user admin --wp-app-password "..."
#   bash verify_wordpress_mcp.sh --url https://yourdomain.com --only ai-engine \
#     --ai-engine-token "..."
#
# Exit codes: 0 all ok / 1 mcp-adapter failed / 2 ai-engine failed / 3 both failed
set -euo pipefail

WP_URL=""
WP_USER=""
WP_PASS=""
AE_TOKEN=""
ONLY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --url)             WP_URL="$2"; shift 2 ;;
    --wp-user)         WP_USER="$2"; shift 2 ;;
    --wp-app-password) WP_PASS="$2"; shift 2 ;;
    --ai-engine-token) AE_TOKEN="$2"; shift 2 ;;
    --only)            ONLY="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

DO_ADAPTER=1
DO_AI_ENGINE=1
if [ -n "$ONLY" ]; then
  case "$ONLY" in
    mcp-adapter) DO_AI_ENGINE=0 ;;
    ai-engine)   DO_ADAPTER=0 ;;
    *) echo "Invalid --only: $ONLY" >&2; exit 1 ;;
  esac
fi

if [ -z "$WP_URL" ]; then
  echo "Error: --url is required" >&2
  exit 1
fi

ADAPTER_OK=0
AE_OK=0

# ── mcp-adapter ──────────────────────────────────────────────────────
if [ "$DO_ADAPTER" -eq 1 ]; then
  echo "=== mcp-adapter (official WordPress) ==="
  if [ -z "$WP_USER" ] || [ -z "$WP_PASS" ]; then
    echo "  ✗ Missing --wp-user and --wp-app-password"
  else
    AUTH=$(echo -n "${WP_USER}:${WP_PASS}" | base64)
    URL="${WP_URL%/}/wp-json/mcp/mcp-adapter-default-server"

    # Step 1: initialize
    INIT=$(curl -sS --max-time 15 -i -X POST "$URL" \
      -H "Authorization: Basic $AUTH" \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"verify","version":"1.0"}}}' 2>&1 || true)

    SESSION=$(echo "$INIT" | grep -i "mcp-session-id" | awk '{print $2}' | tr -d '\r\n')

    if echo "$INIT" | grep -q '"result"'; then
      SERVER_NAME=$(echo "$INIT" | python3 -c 'import json,sys; lines=sys.stdin.read().split("\r\n"); body=[l for l in lines if l.startswith("{")][0]; d=json.loads(body); print(d["result"]["serverInfo"]["name"])' 2>/dev/null || echo "?")
      echo "  ✓ Initialize OK: $SERVER_NAME"
    else
      echo "  ✗ Initialize failed"
      echo "    $(echo "$INIT" | head -5)"
    fi

    if [ -n "$SESSION" ]; then
      # Step 2: notifications/initialized
      curl -sS --max-time 10 -X POST "$URL" \
        -H "Authorization: Basic $AUTH" \
        -H "Content-Type: application/json" \
        -H "Mcp-Session-Id: $SESSION" \
        -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' -o /dev/null 2>&1 || true

      # Step 3: tools/list
      TOOLS=$(curl -sS --max-time 15 -X POST "$URL" \
        -H "Authorization: Basic $AUTH" \
        -H "Content-Type: application/json" \
        -H "Mcp-Session-Id: $SESSION" \
        -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' 2>&1 || true)

      if echo "$TOOLS" | grep -q '"tools"'; then
        COUNT=$(echo "$TOOLS" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d.get("result",{}).get("tools",[])))' 2>/dev/null || echo "?")
        echo "  ✓ tools/list OK: $COUNT tools"
        ADAPTER_OK=1
      else
        echo "  ✗ tools/list failed (empty or error)"
        echo "    $(echo "$TOOLS" | head -c 200)"
      fi
    else
      echo "  ✗ No Mcp-Session-Id header in initialize response"
    fi
  fi
  echo ""
fi

# ── AI Engine ────────────────────────────────────────────────────────
if [ "$DO_AI_ENGINE" -eq 1 ]; then
  echo "=== AI Engine ==="
  if [ -z "$AE_TOKEN" ]; then
    echo "  ✗ Missing --ai-engine-token"
  else
    URL="${WP_URL%/}/wp-json/mcp/v1/http"

    # Initialize (no session needed for AI Engine)
    INIT=$(curl -sS --max-time 15 -X POST "$URL" \
      -H "Authorization: Bearer $AE_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"verify","version":"1.0"}}}' 2>&1 || true)

    if echo "$INIT" | grep -q '"result"'; then
      SERVER_NAME=$(echo "$INIT" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["result"]["serverInfo"]["name"])' 2>/dev/null || echo "?")
      echo "  ✓ Initialize OK: $SERVER_NAME"
    else
      echo "  ✗ Initialize failed"
      echo "    $(echo "$INIT" | head -c 200)"
    fi

    # tools/list (no session header needed)
    TOOLS=$(curl -sS --max-time 15 -X POST "$URL" \
      -H "Authorization: Bearer $AE_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' 2>&1 || true)

    if echo "$TOOLS" | grep -q '"tools"'; then
      COUNT=$(echo "$TOOLS" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d.get("result",{}).get("tools",[])))' 2>/dev/null || echo "?")
      echo "  ✓ tools/list OK: $COUNT tools"
      AE_OK=1
    else
      echo "  ✗ tools/list failed"
      echo "    $(echo "$TOOLS" | head -c 200)"
    fi
  fi
  echo ""
fi

# ── Summary ──────────────────────────────────────────────────────────
if [ "$DO_ADAPTER" -eq 1 ] && [ "$DO_AI_ENGINE" -eq 1 ]; then
  if [ "$ADAPTER_OK" -eq 1 ] && [ "$AE_OK" -eq 1 ]; then
    echo "✅ Both mcp-adapter and AI Engine are ready."
    exit 0
  elif [ "$ADAPTER_OK" -eq 1 ]; then
    echo "⚠ mcp-adapter ok, AI Engine not ready."
    exit 2
  elif [ "$AE_OK" -eq 1 ]; then
    echo "⚠ AI Engine ok, mcp-adapter not ready."
    exit 1
  else
    echo "❌ Neither endpoint is ready."
    exit 3
  fi
elif [ "$DO_ADAPTER" -eq 1 ]; then
  if [ "$ADAPTER_OK" -eq 1 ]; then
    echo "✅ mcp-adapter is ready."
    exit 0
  else
    echo "❌ mcp-adapter not ready."
    exit 1
  fi
elif [ "$DO_AI_ENGINE" -eq 1 ]; then
  if [ "$AE_OK" -eq 1 ]; then
    echo "✅ AI Engine is ready."
    exit 0
  else
    echo "❌ AI Engine not ready."
    exit 2
  fi
fi
