#!/usr/bin/env bash
# Generates a manifest with SHA256 checksums for the backup
# Included in every backup so restore can verify integrity
# Usage: bash generate-manifest.sh /path/to/backup/dir
set -euo pipefail

BACKUP_DIR="${1:-.}"
OUTPUT="$BACKUP_DIR/MANIFEST.txt"

echo "# Hermes Backup Manifest" > "$OUTPUT"
echo "# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTPUT"
echo "# Format: SHA256  SIZE  PATH" >> "$OUTPUT"
echo "" >> "$OUTPUT"

cd "$BACKUP_DIR"
find . -type f ! -name "MANIFEST.txt" -print0 | while IFS= read -r -d '' f; do
  HASH=$(sha256sum "$f" | awk '{print $1}')
  SIZE=$(stat --format='%s' "$f")
  CLEAN="${f#./}"
  echo "$HASH  $SIZE  $CLEAN" >> "$OUTPUT"
done

TOTAL_FILES=$(grep -cE '^[a-f0-9]' "$OUTPUT" || true)
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | awk '{print $1}')
echo "" >> "$OUTPUT"
echo "# Total files: $TOTAL_FILES" >> "$OUTPUT"
echo "# Total size: $TOTAL_SIZE" >> "$OUTPUT"

echo "Manifest written: $OUTPUT ($TOTAL_FILES files, $TOTAL_SIZE)"
