#!/bin/bash
# Video Toolkit Setup — verify prerequisites and install dependencies
set -e

echo "Video Toolkit Setup"
echo "==================="
echo ""

# Check Node.js
if ! command -v node &>/dev/null; then
  echo "ERROR: Node.js is required but not installed."
  echo "  Install: https://nodejs.org/ (v18+ required)"
  exit 1
fi

NODE_VERSION=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
  echo "ERROR: Node.js 18+ required, found v$(node -v)"
  exit 1
fi
echo "Node.js: $(node -v)"

# Check npm
if ! command -v npm &>/dev/null; then
  echo "ERROR: npm not found"
  exit 1
fi
echo "npm: $(npm -v)"

# Check ffmpeg (optional but recommended)
if command -v ffmpeg &>/dev/null; then
  echo "ffmpeg: $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
else
  echo "ffmpeg: not found (optional — needed for post-processing and GIF conversion)"
  echo "  Install: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
fi

echo ""
echo "Ready. To create a video project:"
echo "  cp -r skills/creative/video-toolkit/templates/announcement ./my-video"
echo "  cd my-video && npm install"
echo "  npx remotion studio  # preview in browser"
echo "  npm run render        # render to MP4"
