#!/bin/bash
# Video Toolkit Render Helper
# Usage: ./render.sh [draft|preview|production] [composition-id]
set -e

QUALITY=${1:-preview}
COMP=${2:-MainVideo}

case $QUALITY in
  draft)
    echo "Rendering DRAFT (half resolution, low quality)..."
    npx remotion render src/index.ts "$COMP" \
      --output="draft.mp4" \
      --scale=0.5 \
      --quality=50 \
      --concurrency=100%
    echo "Output: draft.mp4"
    ;;
  preview)
    echo "Rendering PREVIEW (full resolution, standard quality)..."
    npx remotion render src/index.ts "$COMP" \
      --output="preview.mp4" \
      --quality=80 \
      --concurrency=100%
    echo "Output: preview.mp4"
    ;;
  production)
    echo "Rendering PRODUCTION (maximum quality)..."
    npx remotion render src/index.ts "$COMP" \
      --output="final.mp4" \
      --codec=h264 \
      --crf=18 \
      --pixel-format=yuv420p \
      --concurrency=100%
    echo "Output: final.mp4"
    ;;
  *)
    echo "Usage: ./render.sh [draft|preview|production] [composition-id]"
    echo ""
    echo "Quality presets:"
    echo "  draft      — half resolution, low quality, fast"
    echo "  preview    — full resolution, standard quality"
    echo "  production — full resolution, maximum quality"
    exit 1
    ;;
esac
