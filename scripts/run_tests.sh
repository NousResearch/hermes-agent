#!/usr/bin/env bash
# Canonical test runner for hermes-agent. Run this instead of calling
# `pytest` directly to guarantee your local run matches CI behavior.
#
# What this script enforces:
#   * Per-file isolation via scripts/run_tests_parallel.py — each test
#     file runs in its own freshly-spawned `python -m pytest <file>`
#     subprocess. No xdist, no shared workers, no module-level leakage
#     between files.
#   * TZ=UTC, LANG=C.UTF-8, PYTHONHASHSEED=0 (deterministic)
#   * Env vars blanked (conftest.py also does this, but this
#     is belt-and-suspenders for anyone running pytest outside our
#     conftest path — e.g. on a single file)
#   * Proper venv activation (probes .venv, venv, then ~/.hermes/...)
#
# Usage:
#   scripts/run_tests.sh                            # full suite
#   scripts/run_tests.sh -j 4                       # cap parallelism
#   scripts/run_tests.sh tests/agent/               # discover only here
#   scripts/run_tests.sh tests/agent/ tests/acp/    # multiple roots
#   scripts/run_tests.sh tests/foo.py               # single file
#   scripts/run_tests.sh tests/foo.py -- --tb=long  # path + pytest args
#   scripts/run_tests.sh -- -v --tb=long            # pytest args only
#
# Everything after a literal '--' is passed through to each per-file
# pytest invocation. Positional path arguments before '--' override
# the default discovery root (tests/).

set -euo pipefail

# ── Locate repo root ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Activate venv ───────────────────────────────────────────────────────────
VENV=""
for candidate in "$REPO_ROOT/.venv" "$REPO_ROOT/venv" "$HOME/.hermes/hermes-agent/venv"; do
  if [ -f "$candidate/bin/activate" ]; then
    VENV="$candidate"
    break
  fi
done

if [ -z "$VENV" ]; then
  echo "error: no virtualenv found in $REPO_ROOT/.venv or $REPO_ROOT/venv" >&2
  exit 1
fi

PYTHON="$VENV/bin/python"


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
      HERMES_CRON_SESSION \
      HERMES_PLATFORM HERMES_INFERENCE_PROVIDER HERMES_MANAGED HERMES_DEV \
      HERMES_CONTAINER HERMES_EPHEMERAL_SYSTEM_PROMPT HERMES_TIMEZONE \
      HERMES_REDACT_SECRETS HERMES_BACKGROUND_NOTIFICATIONS HERMES_EXEC_ASK \
      HERMES_HOME_MODE 2>/dev/null || true

# Pin deterministic runtime.
export TZ=UTC
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONHASHSEED=0

# ── File-descriptor budget ───────────────────────────────────────────────────
# macOS login shells commonly default to 256 open files. The full pytest suite
# uses subprocess/PTY/socket-heavy tests; at 256 it can cascade into unrelated
# ``OSError: [Errno 24] Too many open files`` errors. Raising the soft limit is
# process-local to this test runner and its children.
_current_nofile="$(ulimit -n 2>/dev/null || echo 0)"
case "$_current_nofile" in
  ''|*[!0-9]*) _current_nofile=0 ;;
esac
if [ "$_current_nofile" -gt 0 ] && [ "$_current_nofile" -lt 4096 ]; then
  ulimit -n 4096 2>/dev/null || {
    _hard_nofile="$(ulimit -Hn 2>/dev/null || echo 0)"
    case "$_hard_nofile" in
      ''|*[!0-9]*|0) ;;
      *) ulimit -n "$_hard_nofile" 2>/dev/null || true ;;
    esac
  }
fi

# ── Live-gateway plugin (computed before we drop env) ───────────────────────
EXTRA_PYTHONPATH=""
EXTRA_PYTEST_PLUGINS=""

# ── Live-gateway test guard (developer machines) ────────────────────────────
# If a system-wide hermes pytest_live_guard plugin is installed at
# $HOME/.hermes/pytest_live_guard.py, force-load it here so every test run
# from this script gets the protection regardless of which worktree is
# checked out (in-tree tests/conftest.py guard may be missing on stale
# branches). Harmless on CI / fresh machines that don't have the file.
if [ -f "$HOME/.hermes/pytest_live_guard.py" ]; then
  EXTRA_PYTHONPATH="$HOME/.hermes"
  EXTRA_PYTEST_PLUGINS="pytest_live_guard"
fi


# ── Run in hermetic env ──────────────────────────────────────────────────────
# env -i: start with empty environment, opt-in only what we need.
# No credential var can leak — you'd have to explicitly add it here.
echo "▶ running per-file parallel test suite via run_tests_parallel.py"
echo "  (TZ=UTC LANG=C.UTF-8 PYTHONHASHSEED=0; clean env)"

cd "$REPO_ROOT"

exec env -i \
  PATH="$PATH" \
  HOME="$HOME" \
  TZ=UTC \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONHASHSEED=0 \
  ${EXTRA_PYTHONPATH:+PYTHONPATH="$EXTRA_PYTHONPATH"} \
  ${EXTRA_PYTEST_PLUGINS:+PYTEST_PLUGINS="$EXTRA_PYTEST_PLUGINS"} \
  "$PYTHON" "$SCRIPT_DIR/run_tests_parallel.py" "$@"
