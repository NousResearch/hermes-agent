#!/usr/bin/env bash
# Canonical test runner for hermes-agent. Run this instead of calling
# `pytest` directly to guarantee your local run matches CI behavior.
#
# What this script enforces:
#   * -n 4 xdist workers (CI has 4 cores; -n auto diverges locally)
#   * TZ=UTC, LANG=C.UTF-8, PYTHONHASHSEED=0 (deterministic)
#   * Credential env vars blanked (conftest.py also does this, but this
#     is belt-and-suspenders for anyone running `pytest` outside of
#     our conftest path — e.g. calling pytest on a single file)
#   * Locked exact `uv sync` + `uv run` in this checkout's project environment
#   * `HERMES_TEST_NO_SYNC=1` opt-out for hot loops after a successful sync
#
# Usage:
#   scripts/run_tests.sh                     # full suite
#   scripts/run_tests.sh tests/agent/        # one directory
#   scripts/run_tests.sh tests/agent/test_foo.py::TestClass::test_method
#   scripts/run_tests.sh --tb=long -v        # pass-through pytest args
#   HERMES_TEST_NO_SYNC=1 scripts/run_tests.sh tests/agent/ -q

set -euo pipefail

# ── Locate repo root ────────────────────────────────────────────────────────
# Works whether this is the main checkout or a worktree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Ensure uv is available ──────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found on PATH. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

# ── Select the project env for this checkout ────────────────────────────────
# Prefer the default `.venv`, preserve legacy `venv` if it already exists,
# and keep the environment scoped to the current checkout/worktree.
ENV_DIR="$REPO_ROOT/.venv"
if [ -d "$REPO_ROOT/venv" ] && [ ! -d "$REPO_ROOT/.venv" ]; then
  ENV_DIR="$REPO_ROOT/venv"
fi
export UV_PROJECT_ENVIRONMENT="$ENV_DIR"

# ── Hermetic environment ────────────────────────────────────────────────────
# Mirror what CI does in .github/workflows/tests.yml + what conftest.py does.
# Unset every credential-shaped var currently in the environment.
while IFS='=' read -r name _; do
  case "$name" in
    *_API_KEY|*_TOKEN|*_SECRET|*_PASSWORD|*_CREDENTIALS|*_ACCESS_KEY| \
    *_SECRET_ACCESS_KEY|*_PRIVATE_KEY|*_OAUTH_TOKEN|*_WEBHOOK_SECRET| \
    *_ENCRYPT_KEY|*_APP_SECRET|*_CLIENT_SECRET|*_CORP_SECRET|*_AES_KEY| \
    AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|FAL_KEY| \
    GH_TOKEN|GITHUB_TOKEN)
      unset "$name"
      ;;
  esac
done < <(env)

# Unset HERMES_* behavioral vars too.
unset HERMES_YOLO_MODE HERMES_INTERACTIVE HERMES_QUIET HERMES_TOOL_PROGRESS \
      HERMES_TOOL_PROGRESS_MODE HERMES_MAX_ITERATIONS HERMES_SESSION_PLATFORM \
      HERMES_SESSION_CHAT_ID HERMES_SESSION_CHAT_NAME HERMES_SESSION_THREAD_ID \
      HERMES_SESSION_SOURCE HERMES_SESSION_KEY HERMES_GATEWAY_SESSION \
      HERMES_PLATFORM HERMES_INFERENCE_PROVIDER HERMES_MANAGED HERMES_DEV \
      HERMES_CONTAINER HERMES_EPHEMERAL_SYSTEM_PROMPT HERMES_TIMEZONE \
      HERMES_REDACT_SECRETS HERMES_BACKGROUND_NOTIFICATIONS HERMES_EXEC_ASK \
      HERMES_HOME_MODE 2>/dev/null || true

# Pin deterministic runtime.
export TZ=UTC
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONHASHSEED=0

# ── Worker count ────────────────────────────────────────────────────────────
# CI uses `-n auto` on ubuntu-latest which gives 4 workers. A 20-core
# workstation with `-n auto` gets 20 workers and exposes test-ordering
# flakes that CI will never see. Pin to 4 so local matches CI.
WORKERS="${HERMES_TEST_WORKERS:-4}"

# ── Run pytest ──────────────────────────────────────────────────────────────
cd "$REPO_ROOT"

# Pass all arguments straight through to pytest.
ARGS=("$@")

if [ "${HERMES_TEST_NO_SYNC:-0}" = "1" ]; then
  echo "▶ skipping uv sync (HERMES_TEST_NO_SYNC=1); using $UV_PROJECT_ENVIRONMENT"
else
  echo "▶ syncing locked test environment via uv in $REPO_ROOT"
  echo "  (env: $UV_PROJECT_ENVIRONMENT)"
  uv sync --locked --extra all --extra dev
fi

echo "▶ running pytest with $WORKERS workers, hermetic env, in $REPO_ROOT"
echo "  (TZ=UTC LANG=C.UTF-8 PYTHONHASHSEED=0; all credential env vars unset)"

# -o "addopts=" clears pyproject.toml's `-n auto` so our -n wins.
exec uv run --no-sync pytest \
  -o "addopts=" \
  -n "$WORKERS" \
  --ignore=tests/integration \
  --ignore=tests/e2e \
  -m "not integration" \
  "${ARGS[@]}"
