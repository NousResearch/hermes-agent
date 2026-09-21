#!/usr/bin/env bash
set -euo pipefail

MIRROR_DIR="$HOME/references/hermes-docs"
LLMS_FILE="$MIRROR_DIR/_llms.txt"
LLMS_FULL_FILE="$MIRROR_DIR/_llms-full.txt"

echo "═══ Hermes Docs Sync ═══"
echo "Date: $(date '+%d/%m/%y %H:%M')"
echo "Mirror: $MIRROR_DIR"
echo ""

# 1. Run the scraper
echo "--- Scraper check ---"
cd "$MIRROR_DIR"
python3 scripts/scrape-docs.py 2>&1

# 2. Download the official llms.txt index for reference
echo ""
echo "--- llms.txt ---"
curl -sL "https://hermes-agent.nousresearch.com/docs/llms.txt" -o "$LLMS_FILE"
PAGE_COUNT=$(grep -c '^\- \[' "$LLMS_FILE" || true)
echo "Downloaded: $PAGE_COUNT pages indexed in llms.txt"

# 3. Silently refresh llms-full.txt for local grep access
curl -sL "https://hermes-agent.nousresearch.com/docs/llms-full.txt" -o "$LLMS_FULL_FILE"
FULL_SIZE=$(wc -c < "$LLMS_FULL_FILE")
echo "llms-full.txt: $FULL_SIZE bytes"

echo ""
echo "Done."
