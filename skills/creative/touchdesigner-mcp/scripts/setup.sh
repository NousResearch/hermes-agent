#!/usr/bin/env bash
# setup.sh — Automated setup for twozero MCP plugin for TouchDesigner
# Idempotent: safe to run multiple times.
set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
OK="${GREEN}✔${NC}"; FAIL="${RED}✘${NC}"; WARN="${YELLOW}⚠${NC}"

TWOZERO_URL="https://www.404zero.com/pisang/twozero.tox"
TOX_PATH="$HOME/Downloads/twozero.tox"
HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
HERMES_CFG="${HERMES_HOME_DIR}/config.yaml"
MCP_PORT=40404
MCP_ENDPOINT="http://localhost:${MCP_PORT}/mcp"

manual_steps=()

echo -e "\n${CYAN}═══ twozero MCP for TouchDesigner — Setup ═══${NC}\n"

# ── 1. Check if TouchDesigner is running ──
# Match on process *name* (not full cmdline) to avoid self-matching shells
# that happen to have "TouchDesigner" in their args. macOS and Linux pgrep
# both support -x for exact name match.
if pgrep -x TouchDesigner >/dev/null 2>&1 || pgrep -x TouchDesignerFTE >/dev/null 2>&1; then
    echo -e " ${OK} TouchDesigner is running"
    td_running=true
else
    echo -e " ${WARN} TouchDesigner is not running"
    td_running=false
fi

# ── 2. Ensure twozero.tox exists ──
if [[ -f "$TOX_PATH" ]]; then
    echo -e " ${OK} twozero.tox already exists at ${TOX_PATH}"
else
    echo -e " ${WARN} twozero.tox not found — downloading..."
    if curl -fSL -o "$TOX_PATH" "$TWOZERO_URL" 2>/dev/null; then
        echo -e " ${OK} Downloaded twozero.tox to ${TOX_PATH}"
    else
        echo -e " ${FAIL} Failed to download twozero.tox from ${TWOZERO_URL}"
        echo "       Please download manually and place at ${TOX_PATH}"
        manual_steps+=("Download twozero.tox from ${TWOZERO_URL} to ${TOX_PATH}")
    fi
fi

# ── 3. Ensure Hermes config has twozero_td MCP entry ──
if [[ ! -f "$HERMES_CFG" ]]; then
    echo -e " ${FAIL} Hermes config not found at ${HERMES_CFG}"
    manual_steps+=("Create ${HERMES_CFG} with twozero_td MCP server entry")
elif grep -q 'twozero_td' "$HERMES_CFG" 2>/dev/null; then
    echo -e " ${OK} twozero_td MCP entry exists in Hermes config"
else
    echo -e " ${WARN} Adding twozero_td MCP entry to Hermes config..."
    python3 -c "
import yaml, sys, copy

cfg_path = '$HERMES_CFG'
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f) or {}

if 'mcp_servers' not in cfg:
    cfg['mcp_servers'] = {}

if 'twozero_td' not in cfg['mcp_servers']:
    cfg['mcp_servers']['twozero_td'] = {
        'url': '${MCP_ENDPOINT}',
        'timeout': 120,
        'connect_timeout': 60
    }
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
" 2>/dev/null && echo -e " ${OK} twozero_td MCP entry added to config" \
              || { echo -e " ${FAIL} Could not update config (is PyYAML installed?)"; \
                   manual_steps+=("Add twozero_td MCP entry to ${HERMES_CFG} manually"); }
    manual_steps+=("Restart Hermes session to pick up config change")
fi

# ── 4. Test if MCP port is responding ──
if nc -z 127.0.0.1 "$MCP_PORT" 2>/dev/null; then
    echo -e " ${OK} Port ${MCP_PORT} is open"

    # ── 5. Verify MCP endpoint responds ──
    resp=$(curl -s --max-time 3 "$MCP_ENDPOINT" 2>/dev/null || true)
    if [[ -n "$resp" ]]; then
        echo -e " ${OK} MCP endpoint responded at ${MCP_ENDPOINT}"
    else
        echo -e " ${WARN} Port open but MCP endpoint returned empty response"
        manual_steps+=("Verify MCP is enabled in twozero settings")
    fi
else
    echo -e " ${WARN} Port ${MCP_PORT} is not open"
    if [[ "$td_running" == true ]]; then
        manual_steps+=("In TD: drag twozero.tox into network editor → click Install")
        manual_steps+=("Enable MCP: twozero icon → Settings → mcp → 'auto start MCP' → Yes")
    else
        manual_steps+=("Launch TouchDesigner")
        manual_steps+=("Drag twozero.tox into the TD network editor and click Install")
        manual_steps+=("Enable MCP: twozero icon → Settings → mcp → 'auto start MCP' → Yes")
    fi
fi

# ── Status Report ──
echo -e "\n${CYAN}═══ Status Report ═══${NC}\n"

if [[ ${#manual_steps[@]} -eq 0 ]]; then
    echo -e " ${OK} ${GREEN}Fully configured! twozero MCP is ready to use.${NC}\n"
    exit 0
else
    echo -e " ${WARN} ${YELLOW}Manual steps remaining:${NC}\n"
    for i in "${!manual_steps[@]}"; do
        echo -e "   $((i+1)). ${manual_steps[$i]}"
    done
    echo ""
    exit 1
fi
