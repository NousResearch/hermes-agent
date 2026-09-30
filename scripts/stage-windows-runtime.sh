#!/usr/bin/env bash
# Składa apps/desktop/build/runtime dla Windows w CI wydania (Git Bash na windows-latest):
# przenośny Python 3.11 + zależności z uv.lock + PortableGit + Node.js LTS.
# Wymaga: uv w PATH, gh (GH_TOKEN), 7z. Lokalnie zob. docs/CZESIEK_INSTALLATION.md.
set -euo pipefail

PYTHON_VERSION="3.11"
NODE_VERSION="${NODE_RUNTIME_VERSION:-v24.20.0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP="$REPO/apps/desktop"
WORK="$DESKTOP/build/win-runtime-work"
win() { cygpath -w "$1"; }

rm -rf "$WORK" "$DESKTOP/build/runtime"
mkdir -p "$WORK"

echo "Python $PYTHON_VERSION…"
export UV_PYTHON_INSTALL_DIR="$(win "$WORK/python")"
uv python install "$PYTHON_VERSION"
PY_BIN="$(uv python find "$PYTHON_VERSION")"
PY_ROOT="$(cd "$(dirname "$(cygpath -u "$PY_BIN")")" && pwd -P)"

echo "Zależności (uv.lock)…"
unset UV_PYTHON
UV_PROJECT_ENVIRONMENT="$(win "$WORK/venv")" uv sync --project "$(win "$REPO")" --python "$(win "$PY_ROOT/python.exe")" --extra all --locked

echo "Node.js $NODE_VERSION…"
curl -fsSL -o "$WORK/node.zip" "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-win-x64.zip"
7z x -y -o"$WORK" "$WORK/node.zip" >/dev/null
NODE_ROOT="$WORK/node-$NODE_VERSION-win-x64"

echo "PortableGit (najnowsze wydanie Git for Windows)…"
gh release download --repo git-for-windows/git --pattern 'PortableGit-*-64-bit.7z.exe' --dir "$WORK"
7z x -y -o"$WORK/git" "$WORK"/PortableGit-*-64-bit.7z.exe >/dev/null

node "$DESKTOP/scripts/stage-windows-runtime.mjs" \
    --python-root="$(win "$PY_ROOT")" \
    --site-packages="$(win "$WORK/venv/Lib/site-packages")" \
    --git-root="$(win "$WORK/git")" \
    --node-root="$(win "$NODE_ROOT")"
rm -rf "$WORK"
