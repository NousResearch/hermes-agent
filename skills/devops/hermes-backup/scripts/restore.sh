#!/usr/bin/env bash
# Hermes Restore Script
# Usage: bash restore.sh
#
# Restores Hermes config, skills, sessions, memories from backup.
# Run this AFTER downloading the backup folder from cloud storage.
set -euo pipefail

PBACKUP_DIR="$(cd "$(dirname "$0")" && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"

echo "============================================"
echo "  Hermes Restore from Backup"
echo "============================================"
echo "Backup dir:   $PBACKUP_DIR"
echo "Hermes home:  $HERMES_HOME"
echo ""

# Safety check
if [ ! -f "$PBACKUP_DIR/config.yaml" ]; then
  echo "❌ ERROR: config.yaml not found in $PBACKUP_DIR"
  echo "   Make sure you're running this from inside the backup folder."
  exit 1
fi

echo "The following will be restored:"
echo "  • config.yaml        → $HERMES_HOME/config.yaml"
echo "  • .env (API keys)    → $HERMES_HOME/.env"
echo "  • skills/            → $HERMES_HOME/skills/"
echo "  • memories/          → $HERMES_HOME/memories/"
echo "  • sessions/ (tar.gz) → $HERMES_HOME/sessions/"
echo "  • engram/            → $HERMES_HOME/engram/"
echo "  • scripts/           → $HERMES_HOME/scripts/"
echo "  • cron/              → $HERMES_HOME/cron/"
echo "  • state/             → $HERMES_HOME/state/"
echo ""
echo "Existing files will be backed up to .bak"
echo ""
read -p "Proceed? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
  echo "Aborted."
  exit 0
fi

mkdir -p "$HERMES_HOME"

restore_file() {
  local src="$1"
  local dst="$2"
  if [ -f "$dst" ]; then
    cp "$dst" "${dst}.bak.$(date +%s)"
    echo "  ⚠️  Existing file backed up: ${dst}.bak"
  fi
  cp "$src" "$dst"
  echo "  ✅ $dst"
}

restore_dir() {
  local src="$1"
  local dst="$2"
  if [ -d "$dst" ]; then
    mv "$dst" "${dst}.bak.$(date +%s)"
    echo "  ⚠️  Existing dir backed up: ${dst}.bak"
  fi
  cp -r "$src" "$dst"
  echo "  ✅ $dst"
}

echo ""
echo "--- Restoring files ---"

# 1. Config
[ -f "$PBACKUP_DIR/config.yaml" ] && restore_file "$PBACKUP_DIR/config.yaml" "$HERMES_HOME/config.yaml"

# 2. .env
if [ -f "$PBACKUP_DIR/hermes-env" ]; then
  restore_file "$PBACKUP_DIR/hermes-env" "$HERMES_HOME/.env"
  chmod 600 "$HERMES_HOME/.env"
  echo "  🔒 chmod 600 $HERMES_HOME/.env"
fi

# 3. Directories
for d in skills scripts cron state career memories engram; do
  [ -d "$PBACKUP_DIR/$d" ] && restore_dir "$PBACKUP_DIR/$d" "$HERMES_HOME/$d"
done

# 4. Sessions
if [ -f "$PBACKUP_DIR/sessions.tar.gz" ]; then
  if [ -d "$HERMES_HOME/sessions" ]; then
    mv "$HERMES_HOME/sessions" "$HERMES_HOME/sessions.bak.$(date +%s)"
    echo "  ⚠️  Existing sessions backed up"
  fi
  mkdir -p "$HERMES_HOME"
  tar xzf "$PBACKUP_DIR/sessions.tar.gz" -C "$HERMES_HOME"
  echo "  ✅ $HERMES_HOME/sessions/ (extracted)"
elif [ -d "$PBACKUP_DIR/sessions" ]; then
  restore_dir "$PBACKUP_DIR/sessions" "$HERMES_HOME/sessions"
fi

# --- Verification ---
echo ""
echo "============================================"
echo "  Verification"
echo "============================================"

PASS=0; FAIL=0
check() {
  if eval "$2" 2>/dev/null; then
    echo "  ✅ $1"; PASS=$((PASS + 1))
  else
    echo "  ❌ $1"; FAIL=$((FAIL + 1))
  fi
}

check "Config exists"   "[ -f $HERMES_HOME/config.yaml ]"
check "Env file exists" "[ -f $HERMES_HOME/.env ]"
check "Skills restored" "[ -d $HERMES_HOME/skills ]"
check "Memories restored" "[ -d $HERMES_HOME/memories ]"
check "Sessions restored" "[ -d $HERMES_HOME/sessions ]"

echo ""
echo "Passed: $PASS  Failed: $FAIL"

echo ""
echo "Next steps:"
echo "  1. source ~/.hermes/.env"
echo "  2. hermes config show"
echo "  3. hermes secrets audit"
echo "  4. Re-setup backup cron (if desired)"
echo ""
echo "Welcome back. 🔄"
