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
#   * Proper venv Python selection (probes POSIX and Windows layouts)
#
# Usage:
#   scripts/run_tests.sh                            # full suite
#   scripts/run_tests.sh -j 4                       # cap parallelism
#   scripts/run_tests.sh tests/agent/               # discover only here
#   scripts/run_tests.sh tests/agent/ tests/acp/    # multiple roots
#   scripts/run_tests.sh tests/foo.py               # single file
#   scripts/run_tests.sh tests/foo.py -q            # path + bare pytest flag
#   scripts/run_tests.sh tests/foo.py -v --tb=long  # bare flags "just work"
#   scripts/run_tests.sh -k 'pattern'               # value flags pass through too
#   scripts/run_tests.sh tests/foo.py -- --tb=long  # explicit '--' still works
#
# Bare pytest flags (anything starting with '-' that isn't one of this
# runner's own options: -j/--jobs, --paths, --slice, --file-timeout, etc.)
# are forwarded to each per-file pytest invocation automatically — no '--'
# separator required. The explicit '--' form still works and stacks with
# bare flags. Positional path arguments override the default discovery
# root (tests/).

set -euo pipefail

# ── Locate repo root ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Locate venv ─────────────────────────────────────────────────────────────
VENV=""
PYTHON=""
venv_candidates=(
  "$REPO_ROOT/.venv"
  "$REPO_ROOT/venv"
  "$HOME/.hermes/hermes-agent/venv"
)

# Git Bash exposes Windows env vars with Windows-style paths; normalize the
# managed install venv path before probing for Scripts/python.exe.
if [ -n "${LOCALAPPDATA:-}" ]; then
  local_appdata="$LOCALAPPDATA"
  if command -v cygpath >/dev/null 2>&1; then
    local_appdata="$(cygpath -u "$LOCALAPPDATA" 2>/dev/null || printf '%s' "$LOCALAPPDATA")"
  fi
  venv_candidates+=("$local_appdata/hermes/hermes-agent/venv")
fi

for candidate in "${venv_candidates[@]}"; do
  if [ -x "$candidate/bin/python" ]; then
    VENV="$candidate"
    PYTHON="$candidate/bin/python"
    break
  fi
  if [ -x "$candidate/Scripts/python.exe" ]; then
    VENV="$candidate"
    PYTHON="$candidate/Scripts/python.exe"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "error: no virtualenv Python found in:" >&2
  for candidate in "${venv_candidates[@]}"; do
    echo "  - $candidate" >&2
  done
  exit 1
fi

# Windows Python ignores Git Bash's POSIX HOME when computing Path.home().
# Provide isolated Windows user-data roots so collection-time imports can call
# Path.home() / LOCALAPPDATA without seeing the developer's real Hermes state.
RUNNER_TMPDIR=""
WINDOWS_HOME_ENV=()
cleanup_runner_tmpdir() {
  if [ -n "$RUNNER_TMPDIR" ]; then
    rm -rf "$RUNNER_TMPDIR"
  fi
}

if "$PYTHON" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.platform == "win32" else 1)
PY
then
  RUNNER_TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/hermes-test-env.XXXXXX")"
  trap cleanup_runner_tmpdir EXIT

  userprofile_posix="$RUNNER_TMPDIR/UserProfile"
  appdata_posix="$userprofile_posix/AppData/Roaming"
  local_appdata_posix="$userprofile_posix/AppData/Local"
  mkdir -p "$appdata_posix" "$local_appdata_posix"

  if command -v cygpath >/dev/null 2>&1; then
    userprofile_env="$(cygpath -w "$userprofile_posix")"
    appdata_env="$(cygpath -w "$appdata_posix")"
    local_appdata_env="$(cygpath -w "$local_appdata_posix")"
  else
    userprofile_env="$userprofile_posix"
    appdata_env="$appdata_posix"
    local_appdata_env="$local_appdata_posix"
  fi

  WINDOWS_HOME_ENV=(
    "USERPROFILE=$userprofile_env"
    "APPDATA=$appdata_env"
    "LOCALAPPDATA=$local_appdata_env"
  )
fi

# ── Live-gateway plugin (computed before we drop env) ───────────────────────
EXTRA_PYTHONPATH=""
EXTRA_PYTEST_PLUGINS=""
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

env -i \
  PATH="$PATH" \
  HOME="$HOME" \
  TZ=UTC \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONIOENCODING=utf-8 \
  PYTHONHASHSEED=0 \
  PYTHONDONTWRITEBYTECODE=1 \
  "${WINDOWS_HOME_ENV[@]}" \
  ${HERMES_RUN_SLOW_PET_TESTS:+HERMES_RUN_SLOW_PET_TESTS="$HERMES_RUN_SLOW_PET_TESTS"} \
  ${EXTRA_PYTHONPATH:+PYTHONPATH="$EXTRA_PYTHONPATH"} \
  ${EXTRA_PYTEST_PLUGINS:+PYTEST_PLUGINS="$EXTRA_PYTEST_PLUGINS"} \
  "$PYTHON" "$SCRIPT_DIR/run_tests_parallel.py" "$@"
