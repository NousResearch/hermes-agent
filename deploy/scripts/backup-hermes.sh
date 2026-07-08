#!/usr/bin/env bash
set -Eeuo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/hermes-backups}"
INCLUDE_SECRETS="${INCLUDE_SECRETS:-0}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="${BACKUP_DIR}/hermes-${TIMESTAMP}.tar.gz"

if [[ ! -d "$HERMES_HOME" ]]; then
  echo "HERMES_HOME does not exist: $HERMES_HOME" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

EXCLUDES=(
  "--exclude=.cache"
  "--exclude=audio_cache"
  "--exclude=cache"
  "--exclude=tmp"
  "--exclude=*.sock"
)

if [[ "$INCLUDE_SECRETS" != "1" ]]; then
  EXCLUDES+=(
    "--exclude=.env"
    "--exclude=auth.json"
    "--exclude=config/*.env"
    "--exclude=*.key"
    "--exclude=*.pem"
  )
fi

MANIFEST="$TMPDIR/MANIFEST.txt"
{
  echo "Hermes backup"
  echo "Created UTC: $TIMESTAMP"
  echo "Source: $HERMES_HOME"
  echo "Include secrets: $INCLUDE_SECRETS"
} > "$MANIFEST"

tar -C "$(dirname "$HERMES_HOME")" "${EXCLUDES[@]}" -czf "$ARCHIVE" "$(basename "$HERMES_HOME")" -C "$TMPDIR" MANIFEST.txt
chmod 0600 "$ARCHIVE"
echo "$ARCHIVE"
