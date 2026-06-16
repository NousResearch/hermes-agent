#!/usr/bin/env bash
set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${CYAN}ℹ${NC} $1"; }
ok() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
err() { echo -e "${RED}✗${NC} $1" >&2; }

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
MEMORY_DIR="$HERMES_HOME/memory"
PLUGIN_DIR="${HERMES_HOME}/plugins/layered"

main() {
  echo ""
  echo -e "${BOLD}Layered Memory Provider — Install${NC}"
  echo ""

  if [ "${1:-}" = "--uninstall" ]; then
    uninstall
    exit 0
  fi

  # 1. Create memory directories and template files
  mkdir -p "$MEMORY_DIR"

  if [ ! -f "$MEMORY_DIR/L0_core.md" ]; then
    cat > "$MEMORY_DIR/L0_core.md" << 'EOF'
# Core Memory (L0)
> Loaded every session. Keep under ~200 tokens.

## User Profile
- Timezone: UTC+8
- Language: English / 中文
- Work style: [edit this]

## Environment
- OS: Linux
- Shell: bash

## Workflows
- Coding: [your editor]
- File management: [your tools]
EOF
    ok "Created L0_core.md"
  else
    info "L0_core.md already exists, skipped"
  fi

  if [ ! -f "$MEMORY_DIR/L1_context.md" ]; then
    cat > "$MEMORY_DIR/L1_context.md" << 'EOF'
# Context Memory (L1)
> Loaded on demand when API keys, versions, or skills are mentioned.

## API Keys
- Service: key in ~/.env

## Versions
- Python: 3.x
- Node: 20.x
EOF
    ok "Created L1_context.md"
  else
    info "L1_context.md already exists, skipped"
  fi

  mkdir -p "$MEMORY_DIR/L2_archive"
  for f in workflows environment decisions; do
    if [ ! -f "$MEMORY_DIR/L2_archive/${f}.md" ]; then
      echo "# ${f}" > "$MEMORY_DIR/L2_archive/${f}.md"
    fi
  done
  ok "Created L2_archive/ with default files"

  # 2. Install plugin files
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  mkdir -p "$PLUGIN_DIR"
  cp "$SCRIPT_DIR/__init__.py" "$PLUGIN_DIR/"
  cp "$SCRIPT_DIR/loader.py" "$PLUGIN_DIR/"
  ok "Installed plugin to $PLUGIN_DIR"

  # 3. Show instructions
  echo ""
  echo -e "${BOLD}Installation complete.${NC}"
  echo ""
  echo "To enable the layered memory provider, run:"
  echo ""
  echo -e "  ${GREEN}hermes config set memory.memory_enabled false${NC}"
  echo -e "  ${GREEN}hermes config set memory.user_profile_enabled false${NC}"
  echo -e "  ${GREEN}hermes config set memory.provider layered${NC}"
  echo ""
  echo "Then edit your memory files:"
  echo -e "  $MEMORY_DIR/L0_core.md"
  echo -e "  $MEMORY_DIR/L1_context.md"
  echo -e "  $MEMORY_DIR/L2_archive/"
  echo ""
  echo "To uninstall:"
  echo -e "  ${YELLOW}rm -rf $PLUGIN_DIR${NC}"
  echo -e "  ${YELLOW}hermes config set memory.provider \"\"${NC}"
}

uninstall() {
  echo ""
  warn "Uninstalling layered memory provider..."
  rm -rf "$PLUGIN_DIR"
  ok "Removed $PLUGIN_DIR"
  echo ""
  info "Run these to restore native memory:"
  echo "  hermes config set memory.memory_enabled true"
  echo "  hermes config set memory.user_profile_enabled true"
  echo "  hermes config set memory.provider \"\""
}

main "$@"
