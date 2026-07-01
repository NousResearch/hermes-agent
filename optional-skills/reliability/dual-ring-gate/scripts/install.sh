#!/bin/bash
# Dual-Ring Gate — One-command setup script
set -e

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"

echo "=== Dual-Ring Gate Install ==="

# 1. Create skill directory
SKILL_DIR="$HERMES_HOME/skills/knowledge/dual-ring-gate"
mkdir -p "$SKILL_DIR/scripts"
mkdir -p "$SKILL_DIR/templates"

# 2. Copy SKILL.md
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cp "$SCRIPT_DIR/SKILL.md" "$SKILL_DIR/"
echo "[OK] SKILL.md installed"

# 3. Copy scripts
cp "$SCRIPT_DIR/scripts/pre-session-check.sh" "$SKILL_DIR/scripts/"
chmod +x "$SKILL_DIR/scripts/pre-session-check.sh"
echo "[OK] pre-session-check.sh installed"

# 4. Copy templates
cp "$SCRIPT_DIR/templates/hot-rules.json" "$SKILL_DIR/templates/"
echo "[OK] hot-rules.json installed"

# 5. Inject inner ring into SOUL.md
SOUL_FILE="$HERMES_HOME/SOUL.md"
INNER_RING=$(cat << 'EOF'

## 🔴 Dual-Ring Gate · Inner Ring (auto-injected · cannot skip)
- **Time check**: terminal('date') before every response
- **Gateway check**: verify gateway status on first response
- **Rule update**: every fix must also update the error rule database
EOF
)

if [ -f "$SOUL_FILE" ]; then
    if grep -q "Dual-Ring Gate" "$SOUL_FILE" 2>/dev/null; then
        echo "[SKIP] Inner ring already in SOUL.md"
    else
        echo "$INNER_RING" >> "$SOUL_FILE"
        echo "[OK] Inner ring injected into SOUL.md"
    fi
else
    echo "$INNER_RING" > "$SOUL_FILE"
    echo "[OK] SOUL.md created with inner ring"
fi

echo "=== Install Complete ==="
echo "Restart your Hermes session to activate."
