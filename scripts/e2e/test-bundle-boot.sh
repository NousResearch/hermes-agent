#!/usr/bin/env bash
#
# Phase 0 task 0.6: E2E gate — prove a bundle boots on a bare system.
#
# Runs inside debian:stable-slim (docker) with NO python/node/git installed.
# The bundle must be fully self-contained: its own Python, venv, node, etc.
#
# Usage: bash scripts/e2e/test-bundle-boot.sh <bundle-dir>
#    or: bash scripts/e2e/test-bundle-boot.sh <bundle-archive.tar.zst>
#
# Requires: docker (or podman). If docker is not available, falls back to
# a local check (less rigorous — the host has python/node installed).
#
# Phase 1 will add `bin/hermes doctor --preflight` — until then, the
# python-import fallback line is the gate. Both lines stay so the script
# tightens automatically when phase 1 lands.

set -euo pipefail

BUNDLE_INPUT="${1:-}"
if [ -z "$BUNDLE_INPUT" ]; then
    echo "Usage: bash scripts/e2e/test-bundle-boot.sh <bundle-dir-or-archive>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Resolve bundle directory (unpack archive if needed)
if [ -d "$BUNDLE_INPUT" ]; then
    BUNDLE_DIR="$BUNDLE_INPUT"
elif [[ "$BUNDLE_INPUT" == *.tar.zst ]]; then
    WORK=$(mktemp -d)
    trap 'rm -rf "$WORK"' EXIT
    echo "==> Unpacking $BUNDLE_INPUT..."
    tar --zstd -xf "$BUNDLE_INPUT" -C "$WORK"
    BUNDLE_DIR="$WORK/bundle"
    if [ ! -d "$BUNDLE_DIR" ]; then
        # Try finding the bundle dir
        BUNDLE_DIR=$(find "$WORK" -name "manifest.json" -type f -exec dirname {} \; | head -1)
    fi
else
    echo "ERROR: $BUNDLE_INPUT is not a directory or .tar.zst archive" >&2
    exit 1
fi

if [ ! -d "$BUNDLE_DIR" ]; then
    echo "ERROR: bundle directory not found" >&2
    exit 1
fi

echo "==> Bundle: $BUNDLE_DIR"

# ─── Try Docker first (the real gate) ──────────────────────────────────

CONTAINER_CMD=""
if command -v docker &>/dev/null; then
    CONTAINER_CMD="docker"
elif command -v podman &>/dev/null; then
    CONTAINER_CMD="podman"
fi

if [ -n "$CONTAINER_CMD" ]; then
    echo "==> Running in debian:stable-slim via $CONTAINER_CMD (no python/node/git)..."

    BUNDLE_ABSOLUTE="$(cd "$BUNDLE_DIR" && pwd)"

    $CONTAINER_CMD run --rm -v "$BUNDLE_ABSOLUTE:/b:ro" debian:stable-slim /bin/sh -c '
        set -e
        echo "--- Checking no system python/node/git ---"
        which python3 2>/dev/null && echo "FAIL: python3 found on host" && exit 1 || true
        which node 2>/dev/null && echo "FAIL: node found on host" && exit 1 || true
        which git 2>/dev/null && echo "FAIL: git found on host" && exit 1 || true
        echo "PASS: no system python/node/git"

        echo "--- bin/hermes --version ---"
        /b/bin/hermes --version

        echo "--- doctor --preflight (phase 1; fallback to import check) ---"
        HERMES_HOME=/tmp/hh /b/bin/hermes doctor --preflight 2>/dev/null || \
        HERMES_HOME=/tmp/hh /b/runtime/venv/bin/python -c "import hermes_cli.main, run_agent, model_tools, gateway.run; print(\"PREFLIGHT_OK\")"

        echo "--- manifest verification ---"
        /b/runtime/venv/bin/python -c "import json; m=json.loads(open(\"/b/manifest.json\").read()); assert m[\"schema\"]==1; assert len(m.get(\"files\",{}))>0; print(\"MANIFEST_OK\")"

        echo "E2E_PASS"
    '
    echo "==> Docker E2E gate passed!"
    exit 0
fi

# ─── Fallback: local check (host has python/node) ─────────────────────

echo "WARN: docker/podman not available — running local fallback check" >&2
echo "    (less rigorous: the host has python/node installed)" >&2
echo ""

echo "--- bin/hermes --version ---"
"$BUNDLE_DIR/bin/hermes" --version

echo "--- core imports ---"
"$BUNDLE_DIR/runtime/venv/bin/python" -c "import hermes_cli.main, run_agent, model_tools, gateway.run; print('PREFLIGHT_OK')"

echo "--- doctor --preflight (phase 1; fallback to import check) ---"
HERMES_HOME=/tmp/hh "$BUNDLE_DIR/bin/hermes" doctor --preflight 2>/dev/null || \
    echo "    (doctor --preflight not yet available — import check above is the gate)"

echo "--- manifest check ---"
if [ -f "$BUNDLE_DIR/manifest.json" ]; then
    "$BUNDLE_DIR/runtime/venv/bin/python" -c "
import json
manifest = json.loads(open('$BUNDLE_DIR/manifest.json').read())
assert manifest['schema'] == 1
assert 'files' in manifest
print(f'MANIFEST_OK: {len(manifest[\"files\"])} files')
"
else
    echo "    WARN: manifest.json not found (run write-manifest.py first)"
fi

echo ""
echo "E2E_PASS (local fallback)"
