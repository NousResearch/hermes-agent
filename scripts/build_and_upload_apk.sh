#!/usr/bin/env bash
# Script to build Android APK and upload to GitHub release
# This creates a proper release with the APK attached

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[+]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

error() {
    echo -e "${RED}[-]${NC} $1"
}

# Build APK
log "Building Android APK..."
cd "$REPO_ROOT"
VERSION="1.0.0"
./scripts/build_android_apk.sh --version "$VERSION" --debug

# Find the APK file
APK_FILE="$REPO_ROOT/Hermes-Agent-${VERSION}-debug.apk"
if [ ! -f "$APK_FILE" ]; then
    error "APK file not found at $APK_FILE"
    exit 1
fi

log "APK built successfully: $APK_FILE"

# Create a release and upload
log "Creating GitHub release v${VERSION}..."

# Get the current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Create tag and push
git tag "v${VERSION}"
git push origin "v${VERSION}"

log "Tag v${VERSION} created and pushed"
warn "GitHub Actions will now automatically build and create a release with the APK"
warn "You can check the release at: https://github.com/luciano2-web/hermes-agent/releases/tag/v${VERSION}"
