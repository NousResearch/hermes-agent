#!/usr/bin/env bash
# Strip Godot-MCP WebSocket subprotocol that breaks Godot 4.5 handshakes.
# Usage: fix-godot-mcp-protocol.sh [godot-mcp-install-dir]
# Default install dir: ${HOME}/.local/share/godot-mcp
set -euo pipefail

ROOT="${1:-${HOME}/.local/share/godot-mcp}"
TARGET="${ROOT}/dist/utils/godot_connection.js"

if [[ ! -f "${TARGET}" ]]; then
  echo "error: missing ${TARGET}" >&2
  echo "Build Godot-MCP and install dist/ to ${ROOT} first." >&2
  exit 1
fi

if grep -q "protocol: 'json'" "${TARGET}" 2>/dev/null \
  || grep -q 'protocol: "json"' "${TARGET}" 2>/dev/null; then
  # portable in-place edit
  tmp="$(mktemp)"
  sed -e "/protocol: 'json',/d" -e '/protocol: "json",/d' "${TARGET}" > "${tmp}"
  mv "${tmp}" "${TARGET}"
  echo "patched: removed protocol json from ${TARGET}"
else
  echo "ok: no protocol json line in ${TARGET} (already patched or different build)"
fi
