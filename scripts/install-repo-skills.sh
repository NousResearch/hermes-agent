#!/usr/bin/env bash
# install-repo-skills.sh — Copy repo-bundled skills to ~/.hermes/skills/
#
# Skills shipped in .hermes/skills/ within the hermes-agent repo are
# fork/prod-specific operational knowledge that should be available to
# all hosts running this fork. This script syncs them.
#
# Usage:
#   ./scripts/install-repo-skills.sh          # copy/update all
#   ./scripts/install-repo-skills.sh --list    # show what would be copied
#   ./scripts/install-repo-skills.sh --dry-run # preview mode

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

REPO_SKILLS_DIR=".hermes/skills"
LOCAL_SKILLS_DIR="${HERMES_HOME:-$HOME/.hermes}/skills"

MODE="${1:-}"

# Colors (bash 3.2 compatible — no \e in echo -e on some systems)
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { printf "${GREEN}[skills]${NC} %s\n" "$*"; }
log2() { printf "  ${CYAN}%s${NC}\n" "$*"; }

if [ ! -d "$REPO_SKILLS_DIR" ]; then
  log "No repo-bundled skills directory (${REPO_SKILLS_DIR}). Nothing to do."
  exit 0
fi

# Find skill directories (contain SKILL.md) — bash 3.2 compatible (no mapfile)
SKILL_DIRS=()
while IFS= read -r -d '' d; do
  if [ -f "$d/SKILL.md" ]; then
    SKILL_DIRS+=("$d")
  fi
done < <(find "$REPO_SKILLS_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

if [ ${#SKILL_DIRS[@]} -eq 0 ]; then
  log "No repo-bundled skills found."
  exit 0
fi

log "Found ${#SKILL_DIRS[@]} repo-bundled skill(s):"
for d in "${SKILL_DIRS[@]}"; do
  log2 "$(basename "$d")"
done

if [ "$MODE" = "--list" ] || [ "$MODE" = "--dry-run" ]; then
  exit 0
fi

mkdir -p "$LOCAL_SKILLS_DIR"

COPIED=0
UPDATED=0
for SRC_DIR in "${SKILL_DIRS[@]}"; do
  SKILL_NAME="$(basename "$SRC_DIR")"
  DST_DIR="$LOCAL_SKILLS_DIR/$SKILL_NAME"

  if [ -d "$DST_DIR" ]; then
    # Check if different
    if diff -q "$SRC_DIR/SKILL.md" "$DST_DIR/SKILL.md" >/dev/null 2>&1; then
      log2 "$SKILL_NAME: up to date"
    else
      cp -R "$SRC_DIR/"* "$DST_DIR/"
      log2 "$SKILL_NAME: updated ✓"
      UPDATED=$((UPDATED + 1))
    fi
  else
    cp -R "$SRC_DIR" "$DST_DIR"
    log2 "$SKILL_NAME: installed ✓"
    COPIED=$((COPIED + 1))
  fi

  # Also copy linked files (references/, templates/, scripts/, assets/)
  for subdir in references templates scripts assets; do
    if [ -d "$SRC_DIR/$subdir" ]; then
      mkdir -p "$DST_DIR/$subdir"
      cp -R "$SRC_DIR/$subdir/"* "$DST_DIR/$subdir/" 2>/dev/null || true
    fi
  done
done

echo ""
log "Done. $COPIED installed, $UPDATED updated."
