#!/usr/bin/env bash
set -euo pipefail
#
# analyze_image.sh
# Analyze an image via Codex CLI vision.
#
# Usage:
#   ./analyze_image.sh <image_path> ["question"]
#
# Example:
#   ./analyze_image.sh /path/to/photo.jpg
#   ./analyze_image.sh /path/to/photo.jpg "Who is in this photo? Describe the scene."
#

IMAGE="$1"
QUESTION="${2:-Describe this image in detail — who/what is depicted, style, mood, colors, composition}"

if [[ ! -f "$IMAGE" ]]; then
  echo "ERROR: Image not found: $IMAGE" >&2
  exit 1
fi

IMGDIR="$(dirname "$IMAGE")"
IMGFILE="$(basename "$IMAGE")"

cd "$IMGDIR"
codex exec "$QUESTION" --image "$IMGFILE" --skip-git-repo-check
