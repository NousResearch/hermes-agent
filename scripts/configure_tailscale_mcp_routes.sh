#!/usr/bin/env bash
set -euo pipefail

BROKER_URL="${BROKER_URL:-http://127.0.0.1:8767}"

paths=(
  "/mcp/grain"
  "/mcp/granola"
  "/mcp/notion-api"
  "/mcp/gws-api"
  "/mcp/zoom-api"
  "/mcp/affinity-api"
)

for path in "${paths[@]}"; do
  tailscale serve --bg --set-path "$path" "${BROKER_URL}${path}"
done

tailscale serve status
