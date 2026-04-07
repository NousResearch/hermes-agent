#!/usr/bin/env bash
# Verify dependencies for manim-kids skill.
# Same requirements as manim-video.

set -e

echo "Checking manim-kids dependencies..."
echo ""

check() {
  if command -v "$1" &>/dev/null; then
    printf "  %-12s %s\n" "$1" "$(eval "$2" 2>&1 | head -1)"
  else
    printf "  %-12s MISSING -- %s\n" "$1" "$3"
    return 1
  fi
}

MISSING=0

check python3  "python3 --version"        "Install Python 3.10+"           || MISSING=$((MISSING+1))
check manim    "manim --version"           "pip install manim"              || MISSING=$((MISSING+1))
check pdflatex "pdflatex --version"        "Install texlive-full or mactex" || MISSING=$((MISSING+1))
check ffmpeg   "ffmpeg -version"           "Install ffmpeg"                 || MISSING=$((MISSING+1))

echo ""
if [ $MISSING -eq 0 ]; then
  echo "All dependencies satisfied."
else
  echo "$MISSING missing dependency(ies). Install them and re-run."
  exit 1
fi
