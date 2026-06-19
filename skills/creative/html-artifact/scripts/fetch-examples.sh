#!/usr/bin/env bash
# Fetch Anthropic's html-effectiveness gallery — 20 self-contained reference HTML
# files demonstrating the artifact patterns this skill teaches. MIT licensed
# (https://github.com/anthropics/html-effectiveness).
#
# Idempotent: clones on first run, pulls latest on subsequent runs. Files land in
# this skill's references/examples/ dir so you can read_file them directly.
#
# Usage:  bash scripts/fetch-examples.sh
# Then:   read_file references/examples/03-code-review-pr.html   (etc.)
set -euo pipefail

REPO_URL="https://github.com/anthropics/html-effectiveness"
# Resolve the skill dir from this script's location (scripts/ -> skill root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$SKILL_DIR/references/examples"

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required but not found on PATH" >&2
  exit 1
fi

if [ -d "$DEST/.git" ]; then
  echo "Refreshing existing gallery in $DEST ..."
  git -C "$DEST" pull --ff-only --quiet || {
    echo "warn: pull failed; re-cloning" >&2
    rm -rf "$DEST"
  }
fi

if [ ! -d "$DEST/.git" ]; then
  echo "Cloning $REPO_URL ..."
  rm -rf "$DEST"
  git clone --depth 1 --quiet "$REPO_URL" "$DEST"
fi

# Report what landed (the 20 numbered examples + index).
COUNT="$(find "$DEST" -maxdepth 1 -name '[0-9]*.html' | wc -l | tr -d ' ')"
echo "Done. $COUNT example HTML files in: $DEST"
echo "Open the index (categorized) or read any file directly:"
echo "  read_file references/examples/index.html"
echo "  read_file references/examples/03-code-review-pr.html"
