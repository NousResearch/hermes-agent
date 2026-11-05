#!/usr/bin/env bash
# setup_composio_mcp.sh — Detect the MCP client platform and patch its config
# with the Composio consumer key (ck_*).
#
# Handles platform-specific config formats:
#   - Claude Code / Claude Desktop / Cursor / Devin CLI: mcpServers.composio with url
#   - Devin Desktop: mcpServers.composio with serverUrl (NOT url)
#   - OpenCode: mcp.composio with type:"remote", environment (NOT env)
#   - Antigravity IDE/CLI: mcpServers.composio with serverUrl (NOT url)
#   - OpenClaw: openclaw mcp add/set CLI (NOT a JSON file patch)
#
# Usage:
#   COMPOSIO_CONSUMER_KEY=ck_xxx bash setup_composio_mcp.sh
#   bash setup_composio_mcp.sh                # prompts for the key
#   bash setup_composio_mcp.sh --dry-run      # show what would change
#   bash setup_composio_mcp.sh --remove       # remove the composio MCP entry
#   bash setup_composio_mcp.sh --platform devin-desktop  # target one platform
#
# Exit codes: 0 ok / 1 missing key / 2 no known config found / 3 python missing
set -euo pipefail

DRY_RUN=0
REMOVE=0
KEY="${COMPOSIO_CONSUMER_KEY:-}"
TARGET_PLATFORM=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --remove)  REMOVE=1; shift ;;
    --platform) TARGET_PLATFORM="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

if [ "$REMOVE" -eq 0 ] && [ -z "$KEY" ]; then
  printf 'Enter your Composio consumer key (ck_...): '
  read -r KEY
  [ -n "$KEY" ] || { echo "No key provided. Aborting." >&2; exit 1; }
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to patch JSON configs." >&2
  exit 3
fi

# ── Platform definitions ─────────────────────────────────────────────
# Format: label|config_path|json_path_to_server|url_field|format
#   format: "standard" = mcpServers with url + headers
#           "serverUrl" = mcpServers with serverUrl + headers (Devin Desktop, Antigravity)
#           "opencode"  = mcp with type:remote, url, headers (OpenCode)
declare -a PLATFORMS=(
  "Claude Code|$HOME/.claude.json|mcpServers.composio|url|standard"
  "Claude Desktop|$HOME/Library/Application Support/Claude/claude_desktop_config.json|mcpServers.composio|url|standard"
  "Cursor|$HOME/.cursor/mcp.json|mcpServers.composio|url|standard"
  "Devin CLI|$HOME/.config/devin/mcp_config.json|mcpServers.composio|url|standard"
  "Devin CLI (legacy)|$HOME/.config/devin/config.json|mcpServers.composio|url|standard"
  "Devin Desktop|$HOME/.devin/mcp_config.json|mcpServers.composio|serverUrl|serverUrl"
  "OpenCode|$HOME/.config/opencode/opencode.json|mcp.composio|url|opencode"
  "OpenCode (project)|$HOME/opencode.json|mcp.composio|url|opencode"
  "Antigravity IDE/CLI|$HOME/.gemini/config/mcp_config.json|mcpServers.composio|serverUrl|serverUrl"
)

patch_json() {
  local label="$1" path="$2" jsonpath="$3" url_field="$4" format="$5"
  [ -f "$path" ] || return 0

  if [ -n "$TARGET_PLATFORM" ] && [ "$label" != "$TARGET_PLATFORM" ] &&
     [ "$label" != "Devin CLI (legacy)" ]; then
    return 0
  fi

  echo "→ Patching $label ($path)"
  if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$REMOVE" -eq 1 ]; then
      echo "  [dry-run] would remove $jsonpath"
    else
      echo "  [dry-run] would set $jsonpath ($format, $url_field)"
    fi
    return 0
  fi

  python3 - "$path" "$jsonpath" "$url_field" "$format" "$KEY" "$REMOVE" <<'PY'
import json, sys
path, jsonpath, url_field, fmt, key, remove = sys.argv[1:7]
remove = int(remove)
with open(path) as f:
    d = json.load(f)

parts = jsonpath.split('.')
parent = d
for p in parts[:-1]:
    parent = parent.setdefault(p, {})
last = parts[-1]

if remove:
    parent.pop(last, None)
else:
    if fmt == "opencode":
        parent[last] = {
            "type": "remote",
            "url": "https://connect.composio.dev/mcp",
            "headers": {"x-consumer-api-key": key},
            "enabled": True
        }
    elif fmt == "serverUrl":
        parent[last] = {
            "serverUrl": "https://connect.composio.dev/mcp",
            "headers": {"x-consumer-api-key": key}
        }
    else:  # standard
        parent[last] = {
            "type": "http",
            "url": "https://connect.composio.dev/mcp",
            "headers": {"x-consumer-api-key": key}
        }

with open(path, 'w') as f:
    json.dump(d, f, indent=2)
print("  done")
PY
}

patch_openclaw() {
  if [ -n "$TARGET_PLATFORM" ] && [ "$TARGET_PLATFORM" != "OpenClaw" ]; then
    return 0
  fi
  if ! command -v openclaw >/dev/null 2>&1; then
    return 0
  fi
  echo "→ Patching OpenClaw (via openclaw mcp CLI)"
  if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$REMOVE" -eq 1 ]; then
      echo "  [dry-run] would run: openclaw mcp unset composio"
    else
      echo "  [dry-run] would run: openclaw mcp add composio --transport streamable-http ..."
    fi
    return 0
  fi
  if [ "$REMOVE" -eq 1 ]; then
    openclaw mcp unset composio 2>/dev/null && echo "  done" || echo "  (openclaw mcp unset failed — entry may not exist)"
  else
    openclaw mcp add composio \
      --transport streamable-http \
      --url https://connect.composio.dev/mcp \
      --header "x-consumer-api-key: $KEY" 2>/dev/null && echo "  done" || \
    openclaw mcp set composio \
      --transport streamable-http \
      --url https://connect.composio.dev/mcp \
      --header "x-consumer-api-key: $KEY" 2>/dev/null && echo "  done" || \
    echo "  (openclaw mcp add/set failed — try manually: openclaw mcp add composio --transport streamable-http --url https://connect.composio.dev/mcp --header 'x-consumer-api-key: $KEY')"
  fi
}

# ── Main ─────────────────────────────────────────────────────────────
found=0

for p in "${PLATFORMS[@]}"; do
  IFS='|' read -r label path jsonpath url_field format <<< "$p"
  if [ -f "$path" ]; then
    patch_json "$label" "$path" "$jsonpath" "$url_field" "$format"
    found=1
  fi
done

# OpenClaw is handled via CLI, not JSON patch
patch_openclaw
command -v openclaw >/dev/null 2>&1 && found=1

if [ "$found" -eq 0 ]; then
  echo "No known MCP config file found." >&2
  echo "Supported platforms: Claude Code, Claude Desktop, Cursor, Devin CLI/Desktop, OpenCode, Antigravity, OpenClaw" >&2
  echo "See references/mcp-config.md for manual setup." >&2
  exit 2
fi

if [ "$REMOVE" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
  echo ""
  echo "Done. Restart your agent for the change to take effect."
  echo "Verify with: bash $(dirname "$0")/verify_composio.sh"
  echo ""
  echo "⚠ Antigravity users: also clear the MCP cache:"
  echo "  rm -rf ~/.gemini/antigravity*/mcp/composio"
fi
