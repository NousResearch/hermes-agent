#!/usr/bin/env bash
# Składa apps/desktop/build/runtime dla macOS (Python + zależności z uv.lock + kod).
# Używane przez scripts/build-macos-installer.sh i przez CI wydania (jedna architektura na przebieg).
set -euo pipefail

[ "$(uname -s)" = "Darwin" ] || { echo "Uruchom na macOS." >&2; exit 1; }
PYTHON_VERSION="3.11"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP="$REPO/apps/desktop"
WORK="$DESKTOP/build/mac-runtime-work"

if ! command -v uv >/dev/null; then
    echo "▸ Instaluję uv (menedżer Pythona)…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
rm -rf "$WORK" "$DESKTOP/build/runtime"
mkdir -p "$WORK"

echo "▸ Pobieram przenośnego Pythona $PYTHON_VERSION…"
export UV_PYTHON_INSTALL_DIR="$WORK/python"
uv python install "$PYTHON_VERSION"
PY_BIN="$(uv python find "$PYTHON_VERSION")"
# …/cpython-<wersja>-macos-<arch>-none/bin/python3 → katalog główny dystrybucji
PY_ROOT="$(cd "$(dirname "$PY_BIN")/.." && pwd -P)"

echo "▸ Instaluję zależności silnika (uv.lock)…"
unset UV_PYTHON
UV_PROJECT_ENVIRONMENT="$WORK/venv" uv sync --project "$REPO" --python "$PY_ROOT/bin/python3" --extra all --locked
SITE="$(echo "$WORK"/venv/lib/python3.*/site-packages)"

echo "▸ Składam silnik…"
node "$DESKTOP/scripts/stage-macos-runtime.mjs" --python-root="$PY_ROOT" --site-packages="$SITE"
rm -rf "$WORK"
