#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECOND_BRAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SECOND_BRAIN_DIR/../.." && pwd)"
SKILL_DIR="$REPO_ROOT/skills/productivity/company-second-brain"
STATIC_DIR="$SECOND_BRAIN_DIR/services/company-ai-gateway/static"
DIST_DIR="$SECOND_BRAIN_DIR/dist"

if [[ ! -f "$SKILL_DIR/SKILL.md" ]]; then
  echo "Cannot find skill source at $SKILL_DIR" >&2
  exit 1
fi

mkdir -p "$STATIC_DIR" "$DIST_DIR"

tar --no-xattrs -czf "$DIST_DIR/company-second-brain-skill.tar.gz" \
  -C "$SKILL_DIR/.." company-second-brain

cp "$DIST_DIR/company-second-brain-skill.tar.gz" \
  "$STATIC_DIR/company-second-brain-skill.tar.gz"
cp "$SECOND_BRAIN_DIR/install-company-second-brain-skill.sh" \
  "$STATIC_DIR/install-company-second-brain-skill.sh"
chmod +x "$STATIC_DIR/install-company-second-brain-skill.sh"

echo "Built install assets:"
echo "- $STATIC_DIR/company-second-brain-skill.tar.gz"
echo "- $STATIC_DIR/install-company-second-brain-skill.sh"
