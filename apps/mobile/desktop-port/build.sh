#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PIN_TAG="v2026.7.7.2"
VENDOR="vendor"

# Two modes:
#   - Standalone (default): clone hermes-agent at PIN_TAG into vendor/, reset it
#     pristine on every build, apply patches/, then build the renderer.
#   - In-tree (HERMES_AGENT_SRC set): build the renderer from that local
#     hermes-agent checkout directly — no clone, no reset, no patches (in-tree
#     the fixes live in apps/desktop itself). Used when this lives as apps/mobile.
if [ -n "${HERMES_AGENT_SRC:-}" ]; then
  DESKTOP_DIR="$HERMES_AGENT_SRC/apps/desktop"
  [ -d "$DESKTOP_DIR" ] || { echo "HERMES_AGENT_SRC set but $DESKTOP_DIR not found." >&2; exit 1; }
  echo "== In-tree renderer source: $DESKTOP_DIR (no clone/patch) =="
else
  if [ ! -d "$VENDOR/.git" ]; then
    git clone --depth 1 --branch "$PIN_TAG" https://github.com/nousresearch/hermes-agent "$VENDOR"
  else
    CURRENT=$(git -C "$VENDOR" describe --tags --exact-match 2>/dev/null || echo "")
    if [ "$CURRENT" != "$PIN_TAG" ]; then
      echo "Vendor is at '$CURRENT', expected '$PIN_TAG' — please delete vendor/ and rebuild." >&2
      exit 1
    fi
  fi
  # Reset vendor to the pristine tag state before every build (discard manual
  # edits; any documented patches are applied afterwards)
  git -C "$VENDOR" checkout -- .
  git -C "$VENDOR" clean -fd

  # Optional extension point: renderer fixes needed against the pinned tag but
  # not yet released. Empty by default — the fixes this port needs now live in
  # apps/desktop upstream, and the in-tree build (above) uses them directly.
  if compgen -G "patches/*.patch" > /dev/null; then
    echo "== Applying vendor patches =="
    for patch in patches/*.patch; do
      echo "  -> $patch"
      git -C "$VENDOR" apply "../$patch"
    done
  fi
  DESKTOP_DIR="$VENDOR/apps/desktop"
fi

echo "== Building renderer (apps/desktop) =="
(cd "$DESKTOP_DIR" && npm install && npx vite build)

rm -rf dist
cp -R "$DESKTOP_DIR/dist" dist

echo "== Injecting browser shim (window.hermesDesktop) =="
node scripts/inject-shim.mjs

echo "== Fixing assets (rebasing hero font paths) =="
node scripts/fix-assets.mjs "$DESKTOP_DIR"

echo "== Build ok: desktop-port/dist =="
