#!/usr/bin/env bash
# Smoke-drives the Hermes Agent CLI: build, version/doctor checks,
# a one-shot agent invocation (with and without a provider key), and
# a quick pytest subset. Run from the repo root (git-bash / WSL / Linux).
#
# Why this file exists instead of just typing the commands: this repo
# lives under a OneDrive-synced path on the reference machine, and
# `uv sync` intermittently fails there (see UV_PROJECT_ENVIRONMENT
# below) — this script encodes the fix so it isn't rediscovered every run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.." # repo root

# uv's package-replace step collides with OneDrive's file-lock-on-sync
# behavior ("error: failed to remove directory ... Access is denied.
# (os error 5)") when .venv lives inside a OneDrive folder. Relocating
# the venv outside OneDrive fixes it. Harmless (and unnecessary) if
# your checkout isn't under OneDrive.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/hermes-agent}"
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"

# Isolate the two oneshot invocations below from any real config the
# developer already has at ~/.hermes/.env (or HERMES_HOME elsewhere).
# main.py loads that file unconditionally on every invocation, so without
# this, "no provider configured" / "bogus key" runs against a machine with
# a real key already set up would silently call a real provider instead of
# exercising the failure paths this script is meant to prove out.
HERMES_SMOKE_HOME="$(mktemp -d)"
trap 'rm -rf "$HERMES_SMOKE_HOME"' EXIT
export HERMES_HOME="$HERMES_SMOKE_HOME"

echo "== uv python pin =="
uv python pin 3.12.13

echo "== uv sync (base + dev extras) =="
uv sync --extra dev

echo "== hermes --version =="
uv run hermes --version

echo "== hermes doctor =="
uv run hermes doctor || true # exits 0 normally; `|| true` only guards flaky CI hosts

echo "== one-shot invocation, no provider configured (expect exit 1, clear error) =="
set +e
uv run hermes -z "say hi"
echo "exit: $?"
set -e

echo "== one-shot invocation, bogus provider key (expect exit 0, upstream 401 text as the 'response') =="
set +e
OPENROUTER_API_KEY=not-a-real-key \
  uv run hermes -z "say hi"
echo "exit: $?"
set -e

echo "== pytest smoke subset (skip anything symlink-based — see Gotchas) =="
# AGENTS.md requires scripts/run_tests.sh over raw pytest: it enforces
# hermetic CI-parity (unset credential vars, TZ=UTC, LANG=C.UTF-8,
# per-file subprocess isolation) that a bare `pytest` invocation skips.
# It only probes ./.venv, ./venv, and $HERMES_PYTHON for an interpreter
# with pytest — since UV_PROJECT_ENVIRONMENT above relocates the venv
# outside OneDrive, point HERMES_PYTHON at it so run_tests.sh finds it.
if [ -x "$UV_PROJECT_ENVIRONMENT/Scripts/python.exe" ]; then
  export HERMES_PYTHON="$UV_PROJECT_ENVIRONMENT/Scripts/python.exe"
elif [ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  export HERMES_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
fi
scripts/run_tests.sh tests/test_account_usage.py -q

echo "== done =="
