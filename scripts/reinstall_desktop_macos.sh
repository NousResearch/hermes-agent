#!/usr/bin/env bash
# Replace /Applications/Hermes.app with a freshly built, signed bundle.
# Never copy only apps/desktop/dist into the installed app — that breaks
# macOS code signing and causes SIGKILL (Code Signature Invalid) on launch.
set -euo pipefail

HERMES_ROOT="${HERMES_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DESKTOP_DIR="$HERMES_ROOT/apps/desktop"
RELEASE_APP="$DESKTOP_DIR/release/mac-arm64/Hermes.app"
TARGET="/Applications/Hermes.app"
BUILD=1

usage() {
  cat <<'EOF'
Usage: scripts/reinstall_desktop_macos.sh [--skip-build]

Build (unless --skip-build) and atomically replace /Applications/Hermes.app
with the signed release bundle. Verifies codesign before opening.

Environment:
  HERMES_ROOT  Hermes agent checkout (default: repo root)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) BUILD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ "$BUILD" -eq 1 ]]; then
  echo "→ Building desktop app (hermes desktop --build-only --force-build)…"
  (
    cd "$HERMES_ROOT"
    # shellcheck disable=SC1091
    source venv/bin/activate
    hermes desktop --build-only --force-build
  )
fi

if [[ ! -f "$RELEASE_APP/Contents/MacOS/Hermes" ]]; then
  echo "Missing release app: $RELEASE_APP" >&2
  exit 1
fi

if ! codesign -vv --deep "$RELEASE_APP" >/dev/null 2>&1; then
  echo "Release app failed codesign verification: $RELEASE_APP" >&2
  codesign -vv --deep "$RELEASE_APP" >&2 || true
  exit 1
fi

if pgrep -x Hermes >/dev/null 2>&1; then
  echo "→ Quitting Hermes…"
  osascript -e 'quit app "Hermes"' >/dev/null 2>&1 || true
  sleep 2
fi

if [[ -d "$TARGET" ]]; then
  backup="$TARGET.backup-$(date +%Y%m%d-%H%M%S)"
  echo "→ Backing up to $backup"
  cp -R "$TARGET" "$backup"
fi

echo "→ Installing $RELEASE_APP → $TARGET"
rm -rf "$TARGET"
cp -R "$RELEASE_APP" "$TARGET"

if ! codesign -vv --deep "$TARGET" >/dev/null 2>&1; then
  echo "Installed app failed codesign verification." >&2
  codesign -vv --deep "$TARGET" >&2 || true
  exit 1
fi

echo "✓ Hermes.app installed and signature valid"
open -a "$TARGET"
