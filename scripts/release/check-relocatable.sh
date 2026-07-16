#!/usr/bin/env bash
#
# Phase 0 task 0.2: Prove (or find) that a CI-built venv works after being
# moved to a different absolute path — the property managed-install slots
# depend on.
#
# Two traps make a naive check false-pass, and both MUST be handled:
#
# 1. uv sync installs the project EDITABLE by default. An editable install's
#    .pth points back at the source checkout — which still exists after you
#    move the venv — so imports succeed via the *source tree*, not the venv.
#    Pass --no-editable.
#
# 2. The source tree must be unreachable during the probe. Even a
#    non-editable install can false-pass if cwd is the repo root ('' on
#    sys.path resolves top-level modules from the checkout). Copy the tree,
#    build from the copy, then DELETE the copy before probing, and probe
#    from a neutral cwd.
#
# Usage: bash scripts/release/check-relocatable.sh
# Exit: 0 = RELOCATABLE_OK; 1 = import failed (findings printed to stderr)

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
WORK=$(mktemp -d) && DST=$(mktemp -d)
trap 'rm -rf "$WORK" "$DST"' EXIT

echo "==> Building venv from a throwaway COPY of the source (no .git)..."
git -C "$REPO_ROOT" archive HEAD | tar -x -C "$WORK"
cp "$REPO_ROOT/uv.lock" "$WORK/uv.lock" 2>/dev/null || true

UV=${UV:-"$HOME/.hermes/bin/uv"}
if [ ! -x "$UV" ]; then
    echo "ERROR: uv not found at $UV — set UV env var or install managed uv" >&2
    exit 1
fi

echo "==> Creating relocatable venv with Python 3.11..."
"$UV" venv --python 3.11 --relocatable "$WORK/venv"

echo "==> Syncing dependencies (non-editable, hash-verified via uv.lock)..."
# --active: install into VIRTUAL_ENV instead of uv's default <project>/.venv.
#   Without it, uv creates $WORK/.venv and leaves our venv empty — the warning
#   "VIRTUAL_ENV=... does not match the project environment path ... will be
#   ignored" is the tell.
VIRTUAL_ENV="$WORK/venv" "$UV" sync --extra all --locked --no-editable --active \
    --project "$WORK" --python "$WORK/venv/bin/python"

echo "==> Moving venv to a different path and DELETING the source copy..."
mv "$WORK/venv" "$DST/venv"
rm -rf "$WORK"

echo "==> Probing core imports from neutral cwd (/)..."
cd /
"$DST/venv/bin/python" -c "import hermes_cli, run_agent, model_tools; print('RELOCATABLE_OK')"
