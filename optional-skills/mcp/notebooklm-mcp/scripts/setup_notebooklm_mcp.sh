#!/usr/bin/env bash
# setup_notebooklm_mcp.sh — Detect the MCP client platform and patch its config
# with the NotebookLM MCP server (notebooklm-mcp / nlm mcp start).
#
# Handles platform-specific config formats:
#   - Claude Code / Claude Desktop / Cursor / Devin CLI / Devin Desktop: mcpServers with command+args
#   - OpenCode: mcp with type:"local", command as single array, environment (NOT env)
#   - Antigravity IDE/CLI: mcpServers with command+args (cache must be cleared separately)
#   - OpenClaw: openclaw mcp add/set CLI
#
# Usage:
#   bash setup_notebooklm_mcp.sh                # patch all detected platforms
#   bash setup_notebooklm_mcp.sh --dry-run      # show what would change
#   bash setup_notebooklm_mcp.sh --remove       # remove the notebooklm-mcp entry
#   bash setup_notebooklm_mcp.sh --platform cursor  # target one platform
#   bash setup_notebooklm_mcp.sh --binary notebooklm-mcp  # use bare binary instead of nlm wrapper
#
# Exit codes: 0 ok / 2 no known config found / 3 python missing
set -euo pipefail

DRY_RUN=0
REMOVE=0
TARGET_PLATFORM=""
# Use "nlm mcp start" wrapper by default (ensures profile/auth layer is initialized)
# Override with --binary notebooklm-mcp for the bare server binary
USE_WRAPPER=1

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --remove)  REMOVE=1; shift ;;
    --platform) TARGET_PLATFORM="$2"; shift 2 ;;
    --binary)
      if [ "$2" = "notebooklm-mcp" ]; then USE_WRAPPER=0; fi
      shift 2 ;;
    -h|--help)
      sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to patch JSON configs." >&2
  exit 3
fi

# Build the command/args based on --binary flag
if [ "$USE_WRAPPER" -eq 1 ]; then
  CMD="nlm"
  ARGS_JSON='["mcp", "start"]'
  ARGS_OPCODE='["nlm", "mcp", "start"]'
else
  CMD="notebooklm-mcp"
  ARGS_JSON='[]'
  ARGS_OPCODE='["notebooklm-mcp"]'
fi

# ── Platform definitions ─────────────────────────────────────────────
# Format: label|config_path|json_path_to_server|format
#   format: "standard" = mcpServers with command + args
#           "opencode"  = mcp with type:local, command as single array
declare -a PLATFORMS=(
  "Claude Code|$HOME/.claude.json|mcpServers.notebooklm-mcp|standard"
  "Claude Desktop|$HOME/Library/Application Support/Claude/claude_desktop_config.json|mcpServers.notebooklm-mcp|standard"
  "Cursor|$HOME/.cursor/mcp.json|mcpServers.notebooklm-mcp|standard"
  "Devin CLI|$HOME/.config/devin/mcp_config.json|mcpServers.notebooklm-mcp|standard"
  "Devin CLI (legacy)|$HOME/.config/devin/config.json|mcpServers.notebooklm-mcp|standard"
  "Devin Desktop|$HOME/.devin/mcp_config.json|mcpServers.notebooklm-mcp|standard"
  "OpenCode|$HOME/.config/opencode/opencode.json|mcp.notebooklm-mcp|opencode"
  "OpenCode (project)|$HOME/opencode.json|mcp.notebooklm-mcp|opencode"
  "Antigravity IDE/CLI|$HOME/.gemini/config/mcp_config.json|mcpServers.notebooklm-mcp|standard"
)

patch_json() {
  local label="$1" path="$2" jsonpath="$3" format="$4"
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
      echo "  [dry-run] would set $jsonpath ($format, cmd=$CMD)"
    fi
    return 0
  fi

  CMD="$CMD" ARGS_JSON="$ARGS_JSON" ARGS_OPCODE="$ARGS_OPCODE" \
  python3 - "$path" "$jsonpath" "$format" "$REMOVE" <<'PY'
import json, os, sys
path, jsonpath, fmt, remove = sys.argv[1:5]
remove = int(remove)
cmd = os.environ["CMD"]
args_json = os.environ["ARGS_JSON"]
args_opcode = os.environ["ARGS_OPCODE"]

import json as j
args_list = j.loads(args_json)
args_opc = j.loads(args_opcode)

with open(path) as f:
    d = j.load(f)

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
            "type": "local",
            "command": args_opc,
            "enabled": True
        }
    else:  # standard
        parent[last] = {
            "command": cmd,
            "args": args_list
        }

with open(path, 'w') as f:
    j.dump(d, f, indent=2)
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
      echo "  [dry-run] would run: openclaw mcp unset notebooklm-mcp"
    else
      echo "  [dry-run] would run: openclaw mcp add notebooklm-mcp --transport stdio ..."
    fi
    return 0
  fi
  if [ "$REMOVE" -eq 1 ]; then
    openclaw mcp unset notebooklm-mcp 2>/dev/null && echo "  done" || echo "  (entry may not exist)"
  else
    if [ "$USE_WRAPPER" -eq 1 ]; then
      openclaw mcp add notebooklm-mcp --transport stdio --command nlm --args mcp,start 2>/dev/null && echo "  done" || \
      openclaw mcp set notebooklm-mcp --transport stdio --command nlm --args mcp,start 2>/dev/null && echo "  done" || \
      echo "  (try manually: openclaw mcp add notebooklm-mcp --transport stdio --command nlm --args mcp,start)"
    else
      openclaw mcp add notebooklm-mcp --transport stdio --command notebooklm-mcp 2>/dev/null && echo "  done" || \
      openclaw mcp set notebooklm-mcp --transport stdio --command notebooklm-mcp 2>/dev/null && echo "  done" || \
      echo "  (try manually: openclaw mcp add notebooklm-mcp --transport stdio --command notebooklm-mcp)"
    fi
  fi
}

# ── Main ─────────────────────────────────────────────────────────────
found=0

for p in "${PLATFORMS[@]}"; do
  IFS='|' read -r label path jsonpath format <<< "$p"
  if [ -f "$path" ]; then
    patch_json "$label" "$path" "$jsonpath" "$format"
    found=1
  fi
done

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
  echo "Authenticate with: nlm login   (manual cookie file is an emergency fallback: nlm login --manual --file cookies.txt)"
  echo "Verify with: bash $(dirname "$0")/verify_notebooklm.sh"
  echo ""
  echo "⚠ Antigravity users: also clear the MCP cache:"
  echo "  rm -rf ~/.gemini/antigravity*/mcp/notebooklm-mcp"
fi
