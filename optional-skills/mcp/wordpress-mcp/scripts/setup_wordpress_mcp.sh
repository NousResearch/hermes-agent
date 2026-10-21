#!/usr/bin/env bash
# setup_wordpress_mcp.sh — Detect all installed MCP client platforms and patch
# each with the WordPress MCP server config (mcp-adapter and/or AI Engine).
#
# Handles platform-specific config formats:
#   - Claude Code: mcpServers.<name> with url, type:"http", headers
#   - Devin CLI: mcpServers.<name> with url, headers (via devin mcp add CLI)
#   - OpenCode: mcp.<name> with type:"remote", url, headers (via opencode mcp add CLI)
#   - Gemini CLI: mcpServers.<name> with httpUrl (NOT url), headers
#   - AGY: mcpServers.<name> with httpUrl (NOT url), headers
#   - Codex: [mcp_servers.<name>] with url + [mcp_servers.<name>.headers] sub-table
#   - OpenClaw: openclaw mcp add CLI (NOT JSON patching)
#
# Usage:
#   bash setup_wordpress_mcp.sh --url https://yourdomain.com \
#     --wp-user admin --wp-app-password "xxxx xxxx xxxx xxxx xxxx xxxx" \
#     --ai-engine-token "your_bearer_token"
#
#   # Only one path:
#   bash setup_wordpress_mcp.sh --url https://yourdomain.com --only mcp-adapter \
#     --wp-user admin --wp-app-password "..."
#   bash setup_wordpress_mcp.sh --url https://yourdomain.com --only ai-engine \
#     --ai-engine-token "..."
#
#   # Dry-run:
#   bash setup_wordpress_mcp.sh ... --dry-run
#
#   # Remove:
#   bash setup_wordpress_mcp.sh --remove
#
#   # Target one platform:
#   bash setup_wordpress_mcp.sh ... --platform claude-code
#
# Exit codes: 0 ok / 1 missing args / 2 no known config found / 3 python missing
set -euo pipefail

DRY_RUN=0
REMOVE=0
WP_URL=""
WP_USER=""
WP_PASS=""
AE_TOKEN=""
ONLY=""
TARGET_PLATFORM=""

MCP_ADAPTER_NAME="wordpress-mcp"
AI_ENGINE_NAME="wordpress-ai-engine"
MCP_ADAPTER_PATH="/wp-json/mcp/mcp-adapter-default-server"
AI_ENGINE_PATH="/wp-json/mcp/v1/http"

while [ $# -gt 0 ]; do
  case "$1" in
    --url)             WP_URL="$2"; shift 2 ;;
    --wp-user)         WP_USER="$2"; shift 2 ;;
    --wp-app-password) WP_PASS="$2"; shift 2 ;;
    --ai-engine-token) AE_TOKEN="$2"; shift 2 ;;
    --only)            ONLY="$2"; shift 2 ;;
    --dry-run)         DRY_RUN=1; shift ;;
    --remove)          REMOVE=1; shift ;;
    --platform)        TARGET_PLATFORM="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

# Determine which paths to configure
DO_ADAPTER=1
DO_AI_ENGINE=1
if [ -n "$ONLY" ]; then
  case "$ONLY" in
    mcp-adapter) DO_AI_ENGINE=0 ;;
    ai-engine)   DO_ADAPTER=0 ;;
    *) echo "Invalid --only value: $ONLY (use 'mcp-adapter' or 'ai-engine')" >&2; exit 1 ;;
  esac
fi

# Validate args
if [ "$REMOVE" -eq 0 ] && [ -z "$WP_URL" ]; then
  echo "Error: --url is required (e.g. --url https://yourdomain.com)" >&2
  exit 1
fi
if [ "$REMOVE" -eq 0 ] && [ "$DO_ADAPTER" -eq 1 ]; then
  if [ -z "$WP_USER" ] || [ -z "$WP_PASS" ]; then
    printf 'Enter WordPress username: '; read -r WP_USER
    printf 'Enter Application Password: '; read -r WP_PASS
    [ -n "$WP_USER" ] && [ -n "$WP_PASS" ] || { echo "Missing username or password. Aborting." >&2; exit 1; }
  fi
fi
if [ "$REMOVE" -eq 0 ] && [ "$DO_AI_ENGINE" -eq 1 ] && [ -z "$AE_TOKEN" ]; then
  printf 'Enter AI Engine Bearer Token: '; read -r AE_TOKEN
  [ -n "$AE_TOKEN" ] || { echo "No token provided. Skipping AI Engine." >&2; DO_AI_ENGINE=0; }
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to patch JSON configs." >&2
  exit 3
fi

# Compute derived values
ADAPTER_URL="${WP_URL%/}${MCP_ADAPTER_PATH}"
AI_ENGINE_URL="${WP_URL%/}${AI_ENGINE_PATH}"
BASIC_AUTH=""
if [ "$DO_ADAPTER" -eq 1 ] && [ -n "$WP_USER" ] && [ -n "$WP_PASS" ]; then
  BASIC_AUTH=$(echo -n "${WP_USER}:${WP_PASS}" | base64)
fi

# ── Platform definitions ─────────────────────────────────────────────
# Format: label|config_path|json_path_prefix|url_field|format
#   format: "claude"     = mcpServers with url, type:"http", headers
#           "gemini"     = mcpServers with httpUrl (NOT url), headers
#           "codex"      = TOML, handled separately
declare -a JSON_PLATFORMS=(
  "claude-code|$HOME/.claude.json|mcpServers|url|claude"
  "gemini-cli|$HOME/.gemini/settings.json|mcpServers|httpUrl|gemini"
  "agy|$HOME/.gemini/antigravity-cli/settings.json|mcpServers|httpUrl|gemini"
)

patch_json() {
  local label="$1" path="$2" jsonpath_root="$3" url_field="$4" format="$5"
  [ -f "$path" ] || return 0

  if [ -n "$TARGET_PLATFORM" ] && [ "$label" != "$TARGET_PLATFORM" ]; then
    return 0
  fi

  echo "→ Patching $label ($path)"
  if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$REMOVE" -eq 1 ]; then
      echo "  [dry-run] would remove ${jsonpath_root}.${MCP_ADAPTER_NAME} and ${jsonpath_root}.${AI_ENGINE_NAME}"
    else
      [ "$DO_ADAPTER" -eq 1 ] && echo "  [dry-run] would add ${jsonpath_root}.${MCP_ADAPTER_NAME} ($format, $url_field)"
      [ "$DO_AI_ENGINE" -eq 1 ] && echo "  [dry-run] would add ${jsonpath_root}.${AI_ENGINE_NAME} ($format, $url_field)"
    fi
    return 0
  fi

  python3 - "$path" "$jsonpath_root" "$url_field" "$format" \
    "$REMOVE" "$DO_ADAPTER" "$DO_AI_ENGINE" \
    "$MCP_ADAPTER_NAME" "$AI_ENGINE_NAME" \
    "$ADAPTER_URL" "$AI_ENGINE_URL" "$BASIC_AUTH" "$AE_TOKEN" <<'PY'
import json, os, sys
path, root, url_field, fmt, remove, do_adapter, do_ae, \
  adapter_name, ae_name, adapter_url, ae_url, basic_auth, ae_token = sys.argv[1:15]
remove = int(remove); do_adapter = int(do_adapter); do_ae = int(do_ae)

with open(path) as f:
    d = json.load(f)

parent = d.setdefault(root, {})

if remove:
    parent.pop(adapter_name, None)
    parent.pop(ae_name, None)
else:
    if do_adapter and basic_auth:
        entry = {"headers": {"Authorization": f"Basic {basic_auth}"}}
        if fmt == "claude":
            entry["type"] = "http"
            entry["url"] = adapter_url
        elif fmt == "gemini":
            entry["httpUrl"] = adapter_url
        parent[adapter_name] = entry

    if do_ae and ae_token:
        entry = {"headers": {"Authorization": f"Bearer {ae_token}"}}
        if fmt == "claude":
            entry["type"] = "http"
            entry["url"] = ae_url
        elif fmt == "gemini":
            entry["httpUrl"] = ae_url
        parent[ae_name] = entry

with open(path, 'w') as f:
    json.dump(d, f, indent=2)
# Restrict access to the file because it contains credentials
os.chmod(path, 0o600)
print("  done")
PY
}

patch_codex() {
  if [ -n "$TARGET_PLATFORM" ] && [ "$TARGET_PLATFORM" != "codex" ]; then
    return 0
  fi
  local path="$HOME/.codex/config.toml"
  [ -f "$path" ] || return 0

  echo "→ Patching Codex ($path)"
  if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$REMOVE" -eq 1 ]; then
      echo "  [dry-run] would remove [mcp_servers.${MCP_ADAPTER_NAME}] and [mcp_servers.${AI_ENGINE_NAME}]"
    else
      [ "$DO_ADAPTER" -eq 1 ] && echo "  [dry-run] would add [mcp_servers.${MCP_ADAPTER_NAME}] with headers sub-table"
      [ "$DO_AI_ENGINE" -eq 1 ] && echo "  [dry-run] would add [mcp_servers.${AI_ENGINE_NAME}] with headers sub-table"
    fi
    return 0
  fi

  if [ "$REMOVE" -eq 1 ]; then
    python3 - "$path" "$MCP_ADAPTER_NAME" "$AI_ENGINE_NAME" <<'PY'
import re, sys
path, adapter_name, ae_name = sys.argv[1:4]
with open(path) as f:
    content = f.read()
for name in [adapter_name, ae_name]:
    # Remove the [mcp_servers.<name>] section and any sub-tables
    pattern = rf'\[mcp_servers\.{re.escape(name)}[^\]]*\][^\[]*'
    content = re.sub(pattern, '', content)
with open(path, 'w') as f:
    f.write(content)
print("  done")
PY
  else
    python3 - "$path" "$REMOVE" "$DO_ADAPTER" "$DO_AI_ENGINE" \
      "$MCP_ADAPTER_NAME" "$AI_ENGINE_NAME" \
      "$ADAPTER_URL" "$AI_ENGINE_URL" "$BASIC_AUTH" "$AE_TOKEN" <<'PY'
import os, sys
path, remove, do_adapter, do_ae, \
  adapter_name, ae_name, adapter_url, ae_url, basic_auth, ae_token = sys.argv[1:11]
do_adapter = int(do_adapter); do_ae = int(do_ae)

with open(path) as f:
    content = f.read()

blocks = []
if do_adapter and basic_auth:
    blocks.append(f'''
[mcp_servers.{adapter_name}]
url = "{adapter_url}"

[mcp_servers.{adapter_name}.headers]
Authorization = "Basic {basic_auth}"
''')
if do_ae and ae_token:
    blocks.append(f'''
[mcp_servers.{ae_name}]
url = "{ae_url}"

[mcp_servers.{ae_name}.headers]
Authorization = "Bearer {ae_token}"
''')

with open(path, 'a') as f:
    for b in blocks:
        f.write(b)
# Restrict access to the file because it contains credentials
os.chmod(path, 0o600)
print("  done")
PY
  fi
}

patch_devin_cli() {
  if [ -n "$TARGET_PLATFORM" ] && [ "$TARGET_PLATFORM" != "devin-cli" ]; then
    return 0
  fi
  command -v devin >/dev/null 2>&1 || return 0

  echo "→ Patching Devin CLI (via devin mcp add)"
  if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$REMOVE" -eq 1 ]; then
      echo "  [dry-run] would run: devin mcp remove ${MCP_ADAPTER_NAME} / ${AI_ENGINE_NAME}"
    else
      [ "$DO_ADAPTER" -eq 1 ] && echo "  [dry-run] would run: devin mcp add -s user ${MCP_ADAPTER_NAME} ..."
      [ "$DO_AI_ENGINE" -eq 1 ] && echo "  [dry-run] would run: devin mcp add -s user ${AI_ENGINE_NAME} ..."
    fi
    return 0
  fi

  if [ "$REMOVE" -eq 1 ]; then
    devin mcp remove "$MCP_ADAPTER_NAME" 2>/dev/null && echo "  removed ${MCP_ADAPTER_NAME}" || true
    devin mcp remove "$AI_ENGINE_NAME" 2>/dev/null && echo "  removed ${AI_ENGINE_NAME}" || true
  else
    if [ "$DO_ADAPTER" -eq 1 ] && [ -n "$BASIC_AUTH" ]; then
      devin mcp add -s user "$MCP_ADAPTER_NAME" "$ADAPTER_URL" \
        --header "Authorization: Basic $BASIC_AUTH" 2>/dev/null \
        && echo "  added ${MCP_ADAPTER_NAME}" || echo "  (devin mcp add failed for ${MCP_ADAPTER_NAME})"
    fi
    if [ "$DO_AI_ENGINE" -eq 1 ] && [ -n "$AE_TOKEN" ]; then
      devin mcp add -s user "$AI_ENGINE_NAME" "$AI_ENGINE_URL" \
        --header "Authorization: Bearer $AE_TOKEN" 2>/dev/null \
        && echo "  added ${AI_ENGINE_NAME}" || echo "  (devin mcp add failed for ${AI_ENGINE_NAME})"
    fi
  fi
}

patch_opencode() {
  if [ -n "$TARGET_PLATFORM" ] && [ "$TARGET_PLATFORM" != "opencode" ]; then
    return 0
  fi
  command -v opencode >/dev/null 2>&1 || return 0

  echo "→ Patching OpenCode (via opencode mcp add)"
  if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$REMOVE" -eq 1 ]; then
      echo "  [dry-run] would remove ${MCP_ADAPTER_NAME} / ${AI_ENGINE_NAME} from OpenCode"
    else
      [ "$DO_ADAPTER" -eq 1 ] && echo "  [dry-run] would run: opencode mcp add ${MCP_ADAPTER_NAME} ..."
      [ "$DO_AI_ENGINE" -eq 1 ] && echo "  [dry-run] would run: opencode mcp add ${AI_ENGINE_NAME} ..."
    fi
    return 0
  fi

  if [ "$REMOVE" -eq 1 ]; then
    echo "  (OpenCode removal: edit ~/.config/opencode/opencode.json manually)"
  else
    if [ "$DO_ADAPTER" -eq 1 ] && [ -n "$BASIC_AUTH" ]; then
      opencode mcp add "$MCP_ADAPTER_NAME" \
        --url "$ADAPTER_URL" \
        --header "Authorization=Basic $BASIC_AUTH" 2>/dev/null \
        && echo "  added ${MCP_ADAPTER_NAME}" || echo "  (opencode mcp add failed for ${MCP_ADAPTER_NAME})"
    fi
    if [ "$DO_AI_ENGINE" -eq 1 ] && [ -n "$AE_TOKEN" ]; then
      opencode mcp add "$AI_ENGINE_NAME" \
        --url "$AI_ENGINE_URL" \
        --header "Authorization=Bearer $AE_TOKEN" 2>/dev/null \
        && echo "  added ${AI_ENGINE_NAME}" || echo "  (opencode mcp add failed for ${AI_ENGINE_NAME})"
    fi
  fi
}

patch_openclaw() {
  if [ -n "$TARGET_PLATFORM" ] && [ "$TARGET_PLATFORM" != "openclaw" ]; then
    return 0
  fi
  command -v openclaw >/dev/null 2>&1 || return 0

  echo "→ Patching OpenClaw (via openclaw mcp add)"
  if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$REMOVE" -eq 1 ]; then
      echo "  [dry-run] would remove ${MCP_ADAPTER_NAME} / ${AI_ENGINE_NAME} from OpenClaw"
    else
      [ "$DO_ADAPTER" -eq 1 ] && echo "  [dry-run] would run: openclaw mcp add ${MCP_ADAPTER_NAME} ..."
      [ "$DO_AI_ENGINE" -eq 1 ] && echo "  [dry-run] would run: openclaw mcp add ${AI_ENGINE_NAME} ..."
    fi
    return 0
  fi

  if [ "$REMOVE" -eq 1 ]; then
    echo "  (OpenClaw removal: use openclaw mcp configure or edit ~/.openclaw/openclaw.json)"
  else
    if [ "$DO_ADAPTER" -eq 1 ] && [ -n "$BASIC_AUTH" ]; then
      openclaw mcp add "$MCP_ADAPTER_NAME" \
        --url "$ADAPTER_URL" \
        --header "Authorization=Basic $BASIC_AUTH" \
        --transport streamable-http \
        --no-probe 2>/dev/null \
        && echo "  added ${MCP_ADAPTER_NAME}" || echo "  (openclaw mcp add failed for ${MCP_ADAPTER_NAME})"
    fi
    if [ "$DO_AI_ENGINE" -eq 1 ] && [ -n "$AE_TOKEN" ]; then
      openclaw mcp add "$AI_ENGINE_NAME" \
        --url "$AI_ENGINE_URL" \
        --header "Authorization=Bearer $AE_TOKEN" \
        --transport streamable-http \
        --no-probe 2>/dev/null \
        && echo "  added ${AI_ENGINE_NAME}" || echo "  (openclaw mcp add failed for ${AI_ENGINE_NAME})"
    fi
  fi
}

# ── Main ─────────────────────────────────────────────────────────────
found=0

# JSON-based platforms (Claude Code, Gemini CLI, AGY)
for p in "${JSON_PLATFORMS[@]}"; do
  IFS='|' read -r label path jsonpath_root url_field format <<< "$p"
  if [ -f "$path" ]; then
    patch_json "$label" "$path" "$jsonpath_root" "$url_field" "$format"
    found=1
  fi
done

# Codex (TOML)
if [ -f "$HOME/.codex/config.toml" ]; then
  patch_codex
  found=1
fi

# CLI-managed platforms
patch_devin_cli
command -v devin >/dev/null 2>&1 && found=1

patch_opencode
command -v opencode >/dev/null 2>&1 && found=1

patch_openclaw
command -v openclaw >/dev/null 2>&1 && found=1

if [ "$found" -eq 0 ]; then
  echo "No known MCP config file or CLI found." >&2
  echo "Supported platforms: Claude Code, Devin CLI, OpenCode, Gemini CLI, AGY, Codex, OpenClaw" >&2
  echo "See references/mcp-config.md for manual setup." >&2
  exit 2
fi

if [ "$REMOVE" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
  echo ""
  echo "Done. Restart your agent(s) for the change to take effect."
  echo "Verify with: bash $(dirname "$0")/verify_wordpress_mcp.sh --url $WP_URL ..."
fi
