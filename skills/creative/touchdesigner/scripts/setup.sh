#!/usr/bin/env bash
# TouchDesigner MCP Setup Verification Script
# Checks all prerequisites and guides configuration

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
info() { echo -e "  ${BLUE}→${NC} $1"; }

echo ""
echo "TouchDesigner MCP Setup Check"
echo "=============================="
echo ""

ERRORS=0

# 1. Check Node.js
echo "1. Node.js"
if command -v node &>/dev/null; then
    NODE_VER=$(node --version 2>/dev/null || echo "unknown")
    MAJOR=$(echo "$NODE_VER" | sed 's/^v//' | cut -d. -f1)
    if [ "$MAJOR" -ge 18 ] 2>/dev/null; then
        pass "Node.js $NODE_VER (>= 18 required)"
    else
        fail "Node.js $NODE_VER (>= 18 required, please upgrade)"
        ERRORS=$((ERRORS + 1))
    fi
else
    fail "Node.js not found"
    info "Install: https://nodejs.org/ or 'brew install node'"
    ERRORS=$((ERRORS + 1))
fi

# 2. Check npm/npx
echo "2. npm/npx"
if command -v npx &>/dev/null; then
    NPX_VER=$(npx --version 2>/dev/null || echo "unknown")
    pass "npx $NPX_VER"
else
    fail "npx not found (usually comes with Node.js)"
    ERRORS=$((ERRORS + 1))
fi

# 3. Check MCP Python package
echo "3. MCP Python package"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
VENV_PYTHON=""

# Try to find the Hermes venv Python
if [ -f "$HERMES_HOME/hermes-agent/.venv/bin/python" ]; then
    VENV_PYTHON="$HERMES_HOME/hermes-agent/.venv/bin/python"
elif [ -f "$HERMES_HOME/hermes-agent/venv/bin/python" ]; then
    VENV_PYTHON="$HERMES_HOME/hermes-agent/venv/bin/python"
fi

if [ -n "$VENV_PYTHON" ]; then
    if $VENV_PYTHON -c "import mcp" 2>/dev/null; then
        MCP_VER=$($VENV_PYTHON -c "import importlib.metadata; print(importlib.metadata.version('mcp'))" 2>/dev/null || echo "installed")
        pass "mcp package ($MCP_VER) in Hermes venv"
    else
        fail "mcp package not installed in Hermes venv"
        info "Install: $VENV_PYTHON -m pip install mcp"
        ERRORS=$((ERRORS + 1))
    fi
else
    warn "Could not find Hermes venv — check mcp package manually"
fi

# 4. Check TouchDesigner
echo "4. TouchDesigner"
TD_FOUND=false

# macOS
if [ -d "/Applications/TouchDesigner.app" ]; then
    TD_FOUND=true
    pass "TouchDesigner found at /Applications/TouchDesigner.app"
fi

# Linux (common install locations)
if command -v TouchDesigner &>/dev/null; then
    TD_FOUND=true
    pass "TouchDesigner found in PATH"
fi

if [ -d "$HOME/TouchDesigner" ]; then
    TD_FOUND=true
    pass "TouchDesigner found at ~/TouchDesigner"
fi

if [ "$TD_FOUND" = false ]; then
    warn "TouchDesigner not detected (may be installed elsewhere)"
    info "Download from: https://derivative.ca/download"
    info "Free Non-Commercial license available"
fi

# 5. Check TD WebServer DAT reachability
echo "5. TouchDesigner WebServer DAT"
TD_URL="${TD_API_URL:-http://127.0.0.1:9981}"
if command -v curl &>/dev/null; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "$TD_URL/api/td/server/td" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        TD_INFO=$(curl -s --connect-timeout 3 "$TD_URL/api/td/server/td" 2>/dev/null || echo "{}")
        pass "TD WebServer DAT responding at $TD_URL"
        info "Response: $TD_INFO"
    elif [ "$HTTP_CODE" = "000" ]; then
        warn "Cannot reach TD WebServer DAT at $TD_URL"
        info "Make sure TouchDesigner is running with mcp_webserver_base.tox imported"
    else
        warn "TD WebServer DAT returned HTTP $HTTP_CODE at $TD_URL"
    fi
else
    warn "curl not found — cannot test TD connection"
fi

# 6. Check Hermes config
echo "6. Hermes MCP config"
CONFIG_FILE="$HERMES_HOME/config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    if grep -q "touchdesigner" "$CONFIG_FILE" 2>/dev/null; then
        pass "TouchDesigner MCP server configured in config.yaml"
    else
        warn "No 'touchdesigner' entry found in mcp_servers config"
        info "Add a touchdesigner entry under mcp_servers: in $CONFIG_FILE"
        info "See references/mcp-tools.md for the configuration block"
    fi
else
    warn "No Hermes config.yaml found at $CONFIG_FILE"
fi

# Summary
echo ""
echo "=============================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All critical checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Open TouchDesigner and import mcp_webserver_base.tox"
    echo "  2. Add the MCP server config to Hermes (see references/mcp-tools.md)"
    echo "  3. Restart Hermes and test: 'Get TouchDesigner server info'"
else
    echo -e "${RED}$ERRORS critical issue(s) found.${NC}"
    echo "Fix the issues above, then re-run this script."
fi
echo ""
