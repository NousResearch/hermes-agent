#!/usr/bin/env bash
# Sync root development package into skills/kubrick for Hermes hub/tap discovery.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/skills/kubrick"
mkdir -p "${DEST}"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.github/' \
  --exclude 'skills/' \
  --exclude 'out/' \
  --exclude 'dist/' \
  --exclude '__pycache__/' \
  --exclude '**/__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'PR_BODY.md' \
  --exclude 'scripts/package_optional_skill.sh' \
  --exclude 'scripts/sync_hub_skill.sh' \
  --exclude 'tests/' \
  "${ROOT}/" "${DEST}/"
find "${DEST}" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
echo "Synced root → ${DEST}"
echo "files: $(find "${DEST}" -type f | wc -l)"
