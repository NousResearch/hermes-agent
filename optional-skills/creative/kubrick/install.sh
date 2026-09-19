#!/usr/bin/env bash
# Kubrick — atomic Hermes Skill Installer
# Usage:
#   ./install.sh                  # ~/.hermes/skills/kubrick
#   ./install.sh creative         # ~/.hermes/skills/creative/kubrick
#   ./install.sh --dry-run [creative]
#   ./install.sh --rollback
#   ./install.sh --version
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_ROOT="${HERMES_HOME:-$HOME/.hermes}"
TARGET_BASE="${HERMES_ROOT}/skills"
BACKUP_BASE="${HERMES_ROOT}/backups/skills"
STAGING_BASE="${HERMES_ROOT}/staging"
RECEIPT_BASE="${HERMES_ROOT}/receipts"
DRY_RUN=0
ROLLBACK=0
SUBDIR=""

usage() {
  sed -n '2,8p' "$0" | sed 's/^# //'
}

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --rollback) ROLLBACK=1 ;;
    --version) cat "${ROOT}/VERSION"; exit 0 ;;
    creative|categorized) SUBDIR="creative/" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

DEST="${TARGET_BASE}/${SUBDIR}kubrick"
LAST_DEST_FILE="${RECEIPT_BASE}/kubrick-last-destination"
LAST_BACKUP_FILE="${RECEIPT_BASE}/kubrick-last-backup"

if [[ "$DRY_RUN" -eq 1 ]]; then
  if [[ "$ROLLBACK" -eq 1 ]]; then
    echo "DRY RUN: would restore the last Kubrick backup to ${DEST}"
  else
    echo "DRY RUN: would stage, validate, and atomically install Kubrick to ${DEST}"
    [[ -d "$DEST" ]] && echo "DRY RUN: would back up the current installation outside the skills tree"
  fi
  exit 0
fi

mkdir -p "$BACKUP_BASE" "$STAGING_BASE" "$RECEIPT_BASE" "$(dirname "$DEST")"

write_receipt() {
  local status="$1" backup_path="${2:-}" displaced_path="${3:-}"
  KUBRICK_INSTALL_STATUS="$status" \
  KUBRICK_INSTALL_DEST="$DEST" \
  KUBRICK_INSTALL_BACKUP="$backup_path" \
  KUBRICK_INSTALL_DISPLACED="$displaced_path" \
  KUBRICK_INSTALL_VERSION="$(cat "${ROOT}/VERSION")" \
  KUBRICK_RECEIPT_PATH="${RECEIPT_BASE}/kubrick-install-$(date -u +%Y%m%dT%H%M%SZ)-$$.json" \
  python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path
payload = {
    "schema_version": "1.0.0",
    "status": os.environ["KUBRICK_INSTALL_STATUS"],
    "version": os.environ["KUBRICK_INSTALL_VERSION"],
    "destination": os.environ["KUBRICK_INSTALL_DEST"],
    "backup_path": os.environ["KUBRICK_INSTALL_BACKUP"] or None,
    "displaced_path": os.environ["KUBRICK_INSTALL_DISPLACED"] or None,
    "validated_before_activation": True,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
Path(os.environ["KUBRICK_RECEIPT_PATH"]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

if [[ "$ROLLBACK" -eq 1 ]]; then
  if [[ -f "$LAST_DEST_FILE" ]]; then
    DEST="$(cat "$LAST_DEST_FILE")"
  fi
  if [[ ! -f "$LAST_BACKUP_FILE" ]]; then
    echo "No Kubrick backup receipt is available for rollback." >&2
    exit 1
  fi
  BACKUP_PATH="$(cat "$LAST_BACKUP_FILE")"
  if [[ ! -d "$BACKUP_PATH" ]]; then
    echo "Recorded Kubrick backup is missing: ${BACKUP_PATH}" >&2
    exit 1
  fi
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  DISPLACED_PATH="${BACKUP_BASE}/kubrick-rollback-displaced-${STAMP}"
  if [[ -d "$DEST" ]]; then
    mv "$DEST" "$DISPLACED_PATH"
  else
    DISPLACED_PATH=""
  fi
  if ! mv "$BACKUP_PATH" "$DEST"; then
    [[ -n "$DISPLACED_PATH" && -d "$DISPLACED_PATH" ]] && mv "$DISPLACED_PATH" "$DEST"
    echo "Rollback failed; active installation was restored." >&2
    exit 1
  fi
  if [[ -n "$DISPLACED_PATH" ]]; then
    printf '%s\n' "$DISPLACED_PATH" > "$LAST_BACKUP_FILE"
  else
    rm -f "$LAST_BACKUP_FILE"
  fi
  printf '%s\n' "$DEST" > "$LAST_DEST_FILE"
  write_receipt "ROLLED_BACK" "$BACKUP_PATH" "$DISPLACED_PATH"
  echo "Kubrick rollback completed: ${DEST}"
  exit 0
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
STAGE="${STAGING_BASE}/kubrick-${STAMP}"
BACKUP_PATH=""

cleanup_stage() {
  [[ -d "$STAGE" ]] && rm -rf "$STAGE"
}
trap cleanup_stage EXIT
mkdir -p "$STAGE"

copy_skill() {
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude '.git/' --exclude '.github/' --exclude 'out/' --exclude 'dist/' \
      --exclude '__pycache__/' --exclude '*.pyc' --exclude '.pytest_cache/' \
      --exclude 'references/usage/' --exclude 'references/reports/' \
      "${ROOT}/" "$STAGE/"
  else
    cp -R "${ROOT}/." "$STAGE/"
    rm -rf "$STAGE/.git" "$STAGE/.github" "$STAGE/out" "$STAGE/dist" \
      "$STAGE/references/usage" "$STAGE/references/reports" 2>/dev/null || true
    find "$STAGE" -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} + 2>/dev/null || true
    find "$STAGE" -type f -name '*.pyc' -delete 2>/dev/null || true
  fi
}

copy_skill
chmod +x "$STAGE/scripts/"*.py "$STAGE/scripts/"*.sh "$STAGE/install.sh" 2>/dev/null || true

echo "Validating staged Kubrick skill..."
python3 "$STAGE/scripts/validate_manifest.py"
python3 "$STAGE/scripts/validate_hermes_skill.py"
python3 "$STAGE/scripts/validate_pattern_corpus.py"
find "$STAGE" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -type f -name '*.pyc' -delete 2>/dev/null || true

if [[ -d "$DEST" ]]; then
  BACKUP_PATH="${BACKUP_BASE}/kubrick-${STAMP}"
  mv "$DEST" "$BACKUP_PATH"
fi

if ! mv "$STAGE" "$DEST"; then
  [[ -n "$BACKUP_PATH" && -d "$BACKUP_PATH" ]] && mv "$BACKUP_PATH" "$DEST"
  echo "Activation failed; previous Kubrick installation was restored." >&2
  exit 1
fi
trap - EXIT

printf '%s\n' "$DEST" > "$LAST_DEST_FILE"
if [[ -n "$BACKUP_PATH" ]]; then
  printf '%s\n' "$BACKUP_PATH" > "$LAST_BACKUP_FILE"
else
  rm -f "$LAST_BACKUP_FILE"
fi
write_receipt "INSTALLED" "$BACKUP_PATH"

echo "Kubrick installed successfully."
echo "Location: ${DEST}"
echo "Validated before activation: manifest, Hermes skill, pattern corpus"
echo "Standalone Hermes skill — Continuity Forge optional."
