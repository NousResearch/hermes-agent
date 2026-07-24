#!/usr/bin/env bash
set -euo pipefail
#
# generate_with_reference.sh
# Generate an image via Codex CLI with a reference image for identity preservation.
#
# Usage:
#   ./generate_with_reference.sh <reference_image> <output_path> <prompt>
#
# Example:
#   ./generate_with_reference.sh \
#     ~/references/primary.jpg \
#     ~/output/portrait.jpg \
#     "Generate a portrait: woman, morning light on deck, natural, warm tones"
#
# Requirements:
#   - codex CLI installed and authenticated (codex auth login)
#   - ChatGPT Plus subscription (for image generation)
#   - macOS with sips (for format conversion). On Linux, install ImageMagick.
#

REFERENCE="$1"
OUTPUT="$2"
PROMPT="$3"
OUTDIR="$(dirname "$OUTPUT")"

if [[ ! -f "$REFERENCE" ]]; then
  echo "ERROR: Reference image not found: $REFERENCE" >&2
  exit 1
fi

# Check for iCloud placeholder
if file "$REFERENCE" | grep -qi "dataless\|placeholder"; then
  echo "ERROR: Reference image appears to be an iCloud placeholder (dataless). Use a local copy." >&2
  exit 1
fi

mkdir -p "$OUTDIR"

NOW=$(date '+%Y-%m-%d %H:%M %Z')
FULL_PROMPT="$PROMPT [Generated at $NOW. Match lighting and atmosphere to the time of day.]"

REFERENCE_DIR="$(dirname "$REFERENCE")"
REFERENCE_FILE="$(basename "$REFERENCE")"

cd "$OUTDIR"
printf '%s' "$FULL_PROMPT" | \
  codex exec --skip-git-repo-check -s workspace-write \
    --add-dir "$REFERENCE_DIR" \
    -i "$REFERENCE_FILE" \
    --add-dir "$OUTDIR" \
    -

SESSION=$(ls -lt "$HOME/.codex/generated_images/" 2>/dev/null | head -1 | awk '{print $NF}')
if [[ -n "$SESSION" ]]; then
  LATEST_PNG=$(find "$HOME/.codex/generated_images/$SESSION" -name '*.png' -type f 2>/dev/null | sort | tail -1)
  if [[ -n "$LATEST_PNG" ]]; then
    case "$OUTPUT" in
      *.png) cp "$LATEST_PNG" "$OUTPUT" ;;
      *.jpg|*.jpeg) sips -s format jpeg "$LATEST_PNG" --out "$OUTPUT" >/dev/null 2>&1 || \
                    convert "$LATEST_PNG" "$OUTPUT" ;;
      *) cp "$LATEST_PNG" "$OUTPUT.png" ;;
    esac
    echo "Generated: $OUTPUT"
    file "$OUTPUT"
    exit 0
  fi
fi

echo "Codex completed but no generated image was found." >&2
exit 3
