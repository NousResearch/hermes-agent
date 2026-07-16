#!/usr/bin/env bash
#
# Phase 0 task 0.3: Build a self-contained Hermes release bundle.
#
# Produces the layout from docs/updater-world.md §2.1:
#
#   dist/bundle/
#   ├── manifest.json        # written by task 0.4 (write-manifest.py)
#   ├── runtime/
#   │   ├── python/          # uv-managed CPython (relocatable)
#   │   ├── venv/            # fully resolved site-packages from uv.lock (non-editable)
#   │   ├── node/            # Node LTS runtime
#   │   └── tools/           # bundled native CLIs (ripgrep)
#   ├── app/                 # git archive of source (no .git), .pyc precompiled
#   ├── ui/
#   │   ├── tui/dist/        # pre-built Ink bundle
#   │   └── web/dist/        # pre-built dashboard SPA
#   ├── desktop/             # pre-built electron app (optional)
#   └── bin/hermes           # launcher shim (phase 0: placeholder)
#
# Usage: bash scripts/release/build-bundle.sh [--out dist/bundle] [--no-desktop]
#
# Everything is best-effort EXCEPT runtime/ + app/: a bundle without desktop/
# is valid (flag it in the manifest as "desktop": false).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUT_DIR="${OUT_DIR:-dist/bundle}"
INCLUDE_DESKTOP=true
UV="${UV:-$HOME/.hermes/bin/uv}"
PYTHON_VERSION="3.11"
NODE_VERSION="22"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)       OUT_DIR="$2"; shift 2 ;;
        --no-desktop) INCLUDE_DESKTOP=false; shift ;;
        --help|-h)
            echo "Usage: bash scripts/release/build-bundle.sh [--out dist/bundle] [--no-desktop]"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "==> Building bundle to: $OUT_DIR"
echo "    Repo: $REPO_ROOT"
echo "    Python: $PYTHON_VERSION, Node: $NODE_VERSION"
echo "    Desktop: $INCLUDE_DESKTOP"

# Clean + create output dir
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# ─── app/ — source tree (git archive, no .git) ──────────────────────────

echo "==> [1/7] Creating app/ from git archive..."
mkdir -p "$OUT_DIR/app"
git -C "$REPO_ROOT" archive HEAD | tar -x -C "$OUT_DIR/app"

# Precompile .pyc with unchecked-hash invalidation (timestamps don't matter
# in an immutable tree).
echo "==> [2/7] Precompiling .pyc files..."
# We need a python for compileall — use uv-managed python
if ! command -v python3 &>/dev/null && [ ! -x "$UV" ]; then
    echo "ERROR: need python3 or uv to precompile" >&2
    exit 1
fi
COMPILE_PY=$(command -v python3 2>/dev/null || echo "")
if [ -z "$COMPILE_PY" ] && [ -x "$UV" ]; then
    COMPILE_PY=$("$UV" python find "$PYTHON_VERSION" 2>/dev/null || echo "")
fi
if [ -z "$COMPILE_PY" ]; then
    echo "WARN: No python found for compileall — skipping .pyc precompilation" >&2
else
    "$COMPILE_PY" -m compileall -j0 "$OUT_DIR/app" \
        --invalidation-mode unchecked-hash 2>/dev/null || \
        echo "WARN: compileall had errors (non-fatal)" >&2
fi

# ─── runtime/python/ — relocatable CPython ─────────────────────────────

echo "==> [3/7] Staging Python runtime..."
if [ ! -x "$UV" ]; then
    echo "ERROR: uv not found at $UV — set UV env var" >&2
    exit 1
fi
PYTHON_INSTALL_DIR="$OUT_DIR/runtime/python"
mkdir -p "$PYTHON_INSTALL_DIR"
# uv installs CPython with a versioned directory structure:
#   <install-dir>/cpython-3.11.15-linux-x86_64-gnu/bin/python
"$UV" python install "$PYTHON_VERSION" --install-dir "$PYTHON_INSTALL_DIR"
# Find the python binary in the nested structure
BUNDLE_PYTHON=$(find "$PYTHON_INSTALL_DIR" -name "python3.11" -type f 2>/dev/null | head -1)
if [ -z "$BUNDLE_PYTHON" ]; then
    BUNDLE_PYTHON=$(find "$PYTHON_INSTALL_DIR" -name "python" -type f 2>/dev/null | head -1)
fi
if [ -z "$BUNDLE_PYTHON" ] || [ ! -x "$BUNDLE_PYTHON" ]; then
    echo "ERROR: bundle python not found in $PYTHON_INSTALL_DIR" >&2
    exit 1
fi
echo "    Python at: $BUNDLE_PYTHON"

# ─── runtime/venv/ — fully resolved, non-editable ──────────────────────

echo "==> [4/7] Creating non-editable venv from uv.lock..."
VENV_DIR="$OUT_DIR/runtime/venv"
"$UV" venv --python "$BUNDLE_PYTHON" --relocatable "$VENV_DIR"

# Build from a throwaway copy (like check-relocatable.sh) so the source
# tree is unreachable after the build — proves the venv carries everything.
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
git -C "$REPO_ROOT" archive HEAD | tar -x -C "$WORK"
cp "$REPO_ROOT/uv.lock" "$WORK/uv.lock" 2>/dev/null || true

VIRTUAL_ENV="$VENV_DIR" "$UV" sync --extra all --locked --no-editable --active \
    --project "$WORK" --python "$VENV_DIR/bin/python"

# Fix: uv --relocatable makes entry-point scripts relative but leaves the
# python symlink absolute. In a bundle mounted at a different path (e.g.
# docker /b instead of /tmp/e2e-bundle), the absolute symlink breaks.
# Replace it with a relative symlink to the runtime python.
echo "    Fixing venv python symlink to be relative..."
PYTHON_SYMLINK="$VENV_DIR/bin/python"
if [ -L "$PYTHON_SYMLINK" ]; then
    TARGET=$(readlink "$PYTHON_SYMLINK")
    if [[ "$TARGET" == /* ]]; then
        # Absolute — convert to relative
        REL_TARGET=$(python3 -c "import os.path; print(os.path.relpath('$TARGET', '$VENV_DIR/bin'))" 2>/dev/null || echo "")
        if [ -n "$REL_TARGET" ]; then
            ln -sf "$REL_TARGET" "$PYTHON_SYMLINK"
        fi
    fi
fi

echo "==> [5/7] Staging Node.js $NODE_VERSION runtime..."
NODE_DIR="$OUT_DIR/runtime/node"
mkdir -p "$NODE_DIR"

ARCH=$(uname -m)
NODE_ARCH="x64"
case "$ARCH" in
    x86_64)        NODE_ARCH="x64"    ;;
    aarch64|arm64) NODE_ARCH="arm64"  ;;
    *) echo "WARN: Unsupported arch $ARCH for Node — skipping" >&2 ;;
esac
NODE_OS="linux"
case "$(uname -s)" in
    Linux)  NODE_OS="linux"  ;;
    Darwin) NODE_OS="darwin" ;;
    *) echo "WARN: Unsupported OS for Node — skipping" >&2 ;;
esac

if [ -n "$NODE_ARCH" ] && [ -n "$NODE_OS" ]; then
    INDEX_URL="https://nodejs.org/dist/latest-v${NODE_VERSION}.x/"
    TARBALL=$(curl -fsSL "$INDEX_URL" 2>/dev/null \
        | grep -oE "node-v${NODE_VERSION}\.[0-9]+\.[0-9]+-${NODE_OS}-${NODE_ARCH}\.tar\.xz" \
        | head -1)
    if [ -z "$TARBALL" ]; then
        TARBALL=$(curl -fsSL "$INDEX_URL" 2>/dev/null \
            | grep -oE "node-v${NODE_VERSION}\.[0-9]+\.[0-9]+-${NODE_OS}-${NODE_ARCH}\.tar\.gz" \
            | head -1)
    fi
    if [ -n "$TARBALL" ]; then
        DOWNLOAD_URL="https://nodejs.org/dist/latest-v${NODE_VERSION}.x/$TARBALL"
        echo "    Downloading $TARBALL..."
        TMP_TAR=$(mktemp)
        curl -fsSL "$DOWNLOAD_URL" -o "$TMP_TAR"
        tar -xf "$TMP_TAR" -C "$NODE_DIR" --strip-components=1
        rm -f "$TMP_TAR"
        echo "    Node staged at $NODE_DIR"
    else
        echo "WARN: Could not find Node.js $NODE_VERSION tarball — skipping" >&2
    fi
fi

# ─── runtime/tools/ — bundled native CLIs ──────────────────────────────

echo "==> [6/7] Staging bundled native CLIs (ripgrep)..."
TOOLS_DIR="$OUT_DIR/runtime/tools"
mkdir -p "$TOOLS_DIR"
# Ripgrep: use system rg if available (it's a static binary on most distros)
if command -v rg &>/dev/null; then
    cp "$(command -v rg)" "$TOOLS_DIR/rg"
    echo "    rg copied from $(command -v rg)"
else
    echo "    WARN: rg not found on system — bundle will lack ripgrep" >&2
fi

# ─── ui/ — pre-built TUI + web ────────────────────────────────────────

echo "==> [7/7] Building UI surfaces..."

# TUI (Ink) build
TUI_DIR="$REPO_ROOT/ui-tui"
if [ -d "$TUI_DIR" ]; then
    echo "    Building TUI..."
    (cd "$TUI_DIR" && npm ci --ignore-scripts 2>/dev/null || npm install --ignore-scripts 2>/dev/null)
    (cd "$TUI_DIR" && npm run build 2>/dev/null) || echo "    WARN: TUI build failed" >&2
    if [ -d "$TUI_DIR/dist" ]; then
        mkdir -p "$OUT_DIR/ui/tui"
        cp -r "$TUI_DIR/dist" "$OUT_DIR/ui/tui/dist"
        echo "    TUI dist staged"
    fi
else
    echo "    WARN: ui-tui/ not found — skipping TUI build" >&2
fi

# Web dashboard build
WEB_DIR="$REPO_ROOT/web"
if [ -d "$WEB_DIR" ]; then
    echo "    Building web dashboard..."
    (cd "$WEB_DIR" && npm ci --ignore-scripts 2>/dev/null || npm install --ignore-scripts 2>/dev/null)
    (cd "$WEB_DIR" && npm run build 2>/dev/null) || echo "    WARN: web build failed" >&2
    WEB_DIST="$REPO_ROOT/hermes_cli/web_dist"
    if [ -d "$WEB_DIST" ]; then
        mkdir -p "$OUT_DIR/ui/web"
        cp -r "$WEB_DIST" "$OUT_DIR/ui/web/dist"
        echo "    Web dist staged"
    fi
else
    echo "    WARN: web/ not found — skipping web build" >&2
fi

# ─── desktop/ — pre-built electron app (optional) ──────────────────────

if [ "$INCLUDE_DESKTOP" = true ] && [ -d "$REPO_ROOT/apps/desktop" ]; then
    echo "==> Building desktop app..."
    DESKTOP_DIR="$REPO_ROOT/apps/desktop"
    (cd "$DESKTOP_DIR" && npm ci --ignore-scripts 2>/dev/null || npm install --ignore-scripts 2>/dev/null)
    (cd "$DESKTOP_DIR" && CSC_IDENTITY_AUTO_DISCOVERY=false npm run pack 2>/dev/null) || {
        echo "    WARN: desktop build failed — bundle will be valid without it" >&2
    }
    UNPACKED=$(find "$DESKTOP_DIR/release" -maxdepth 1 -name "*-unpacked" -type d 2>/dev/null | head -1)
    if [ -n "$UNPACKED" ]; then
        mkdir -p "$OUT_DIR/desktop"
        cp -r "$UNPACKED"/* "$OUT_DIR/desktop/"
        echo "    Desktop app staged"
    else
        echo "    WARN: No unpacked desktop build found — bundle will lack desktop/" >&2
    fi
else
    echo "==> Skipping desktop build (--no-desktop or no apps/desktop)"
fi

# ─── bin/hermes — placeholder launcher shim ────────────────────────────

echo "==> Creating bin/hermes launcher shim..."
mkdir -p "$OUT_DIR/bin"
cat > "$OUT_DIR/bin/hermes" << 'STUB'
#!/bin/sh
# Phase 0 placeholder launcher shim.
# Phase 1 replaces this with the native Rust launcher binary.
# This shim execs the venv python, which knows about site-packages.
# The venv's bin/python is a relative symlink to the bundle's CPython
# (made relative by build-bundle.sh's post-venv fixup).
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/../runtime/venv/bin/python" -m hermes_cli.main "$@"
STUB
chmod +x "$OUT_DIR/bin/hermes"

# ─── Summary ──────────────────────────────────────────────────────────

echo ""
echo "==> Bundle built at: $OUT_DIR"
echo "    Size: $(du -sh "$OUT_DIR" 2>/dev/null | cut -f1)"
echo ""

# Verify the bundle boots
echo "==> Verifying bundle..."
"$OUT_DIR/bin/hermes" --version 2>/dev/null && echo "    PASS: bin/hermes --version" || \
    echo "    WARN: bin/hermes --version failed (may need manifest.json from task 0.4)"

"$OUT_DIR/runtime/venv/bin/python" -c "import hermes_cli, run_agent, model_tools; print('    PASS: core imports')" 2>/dev/null || \
    echo "    WARN: core import check failed"

echo ""
echo "==> Done. Next: run scripts/release/write-manifest.py to add manifest.json"
