#!/usr/bin/env bash
set -euo pipefail
#
# batch_generate.sh
# Generate multiple images from a prompt file, each with a shared reference image.
#
# Usage:
#   ./batch_generate.sh <reference_image> <output_dir> <prompt_file>
#
# prompt_file format: one prompt per line. Lines starting with # are skipped.
# Output files: output_dir/001.png, output_dir/002.png, ...
#
# Example:
#   ./batch_generate.sh ~/references/primary.jpg ~/output/prompts.txt
#

REFERENCE="$1"
OUTPUT_DIR="$2"
PROMPT_FILE="$3"

mkdir -p "$OUTPUT_DIR"

COUNTER=0
while IFS= read -r line; do
  [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

  COUNTER=$((COUNTER + 1))
  PADDED=$(printf '%03d' "$COUNTER")
  OUT_FILE="$OUTPUT_DIR/${PADDED}.png"

  echo "=== Generating #$COUNTER: $line ==="

  REFERENCE_DIR="$(dirname "$REFERENCE")"
  REFERENCE_FILE="$(basename "$REFERENCE")"

  cd "$OUTPUT_DIR"
  printf '%s' "$line" | \
    codex exec --skip-git-repo-check -s workspace-write \
      --add-dir "$REFERENCE_DIR" \
      -i "$REFERENCE_FILE" \
      --add-dir "$OUTPUT_DIR" \
      -

  SESSION=$(ls -lt "$HOME/.codex/generated_images/" 2>/dev/null | head -1 | awk '{print $NF}')
  if [[ -n "$SESSION" ]]; then
    LATEST=$(find "$HOME/.codex/generated_images/$SESSION" -name '*.png' -type f 2>/dev/null | sort | tail -1)
    if [[ -n "$LATEST" ]]; then
      cp "$LATEST" "$OUT_FILE"
      echo "  → Saved: $OUT_FILE"
    fi
  fi

  echo "---"
done < "$PROMPT_FILE"

echo "BATCH DONE: $COUNTER images generated in $OUTPUT_DIR"
