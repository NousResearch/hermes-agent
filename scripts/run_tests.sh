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
#   * Proper venv activation (exact Python 3.12 only: probes .venv, venv,
#     then HERMES_PYTHON; fails closed with no python/PATH fallback)
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

# ── Locate python (exact Python 3.12 only) ───────────────────────────────────
# Probe repo-local venvs first; fall back to the Nix devShell's editable venv
# (HERMES_PYTHON is exported by the devShell hook and ships [dev] extras:
# pytest, pytest-asyncio, pytest-timeout, ruff, ty).
#
# A candidate is accepted ONLY if its executable probes as exactly
# Python 3.12 (sys.version_info[:2] == (3, 12)) AND imports pytest.
# Anything else — missing pytest, or a 3.11/3.13/3.14 interpreter even with
# pytest installed — is skipped (repo venvs) or rejected (HERMES_PYTHON),
# and the runner fails closed. There is deliberately no `python`/`python3`/
# PATH fallback: an unpinned lookup resolves to whatever the host provides.
#
# The live release venv at $HOME/.hermes/hermes-agent/venv is NEVER probed
# here. It is a protected 3.11 runtime, so accepting it whenever a checkout
# has no local .venv lets the canonical runner select a forbidden
# interpreter. Create $REPO_ROOT/.venv with Python 3.12 (plus dev extras)
# or enter the Nix devShell instead.
VENV=""
VENV_PYTHON=""
SKIPPED_VENVS=""
_PY312_PYTEST_PROBE='import sys, pytest; sys.exit(sys.version_info[:2] != (3, 12))'
for candidate in "$REPO_ROOT/.venv" "$REPO_ROOT/venv"; do
  if [ -f "$candidate/bin/activate" ]; then
    if "$candidate/bin/python" -c "$_PY312_PYTEST_PROBE" 2>/dev/null; then
      VENV="$candidate"
      VENV_PYTHON="$candidate/bin/python"
      break
    fi
    SKIPPED_VENVS="$SKIPPED_VENVS $candidate"
  fi
  # Native Windows venv layout: python.exe and activate live under
  # Scripts/, and there is no bin/. Anyone running this script from
  # Git Bash / MSYS with a `python -m venv`- or uv-created venv hits
  # this branch — without it the canonical runner refuses to start.
  # The same exact-Python-3.12-plus-pytest gate applies here.
  if [ -f "$candidate/Scripts/activate" ]; then
    if "$candidate/Scripts/python.exe" -c "$_PY312_PYTEST_PROBE" 2>/dev/null; then
      VENV="$candidate"
      VENV_PYTHON="$candidate/Scripts/python.exe"
      break
    fi
    SKIPPED_VENVS="$SKIPPED_VENVS $candidate"
  fi
done

if [ -n "$SKIPPED_VENVS" ]; then
  for skipped in $SKIPPED_VENVS; do
    echo "▶ skipping venv without exactly Python 3.12 + pytest: $skipped" >&2
  done
fi

if [ -n "$VENV" ]; then
  PYTHON="$VENV_PYTHON"
elif [ -n "${HERMES_PYTHON:-}" ] && [ -x "$HERMES_PYTHON" ] \
    && "$HERMES_PYTHON" -c "$_PY312_PYTEST_PROBE" 2>/dev/null; then
  # Guard with a version + import check: HERMES_PYTHON may point at the
  # live RELEASE venv (protected 3.11 runtime, with pytest installed) when
  # inherited from a wrapped `hermes` binary rather than the devShell hook.
  # A non-3.12 interpreter is rejected even when pytest imports.
  PYTHON="$HERMES_PYTHON"
  echo "▶ no local venv — using Nix dev venv via HERMES_PYTHON: $PYTHON (Python 3.12 with pytest)"
else
  if [ -n "${HERMES_PYTHON:-}" ]; then
    echo "▶ rejecting HERMES_PYTHON=${HERMES_PYTHON} (requires exactly Python 3.12 with pytest)" >&2
  fi
  echo "error: no Python 3.12 virtualenv with pytest found in $REPO_ROOT/.venv or $REPO_ROOT/venv," >&2
  echo "       and HERMES_PYTHON is not an exactly-Python-3.12 python with pytest (enter the Nix devShell or create a 3.12 venv)" >&2
  if [ -n "$SKIPPED_VENVS" ]; then
    echo "       (skipped — not exactly Python 3.12 with pytest:$SKIPPED_VENVS — install dev extras into a 3.12 venv, or create $REPO_ROOT/.venv with Python 3.12)" >&2
  fi
  exit 1
fi


# ── Live-gateway plugin (computed before we drop env) ───────────────────────
EXTRA_PYTHONPATH=""
EXTRA_PYTEST_PLUGINS=""
if [ -f "$HOME/.hermes/pytest_live_guard.py" ]; then
  EXTRA_PYTHONPATH="$HOME/.hermes"
  EXTRA_PYTEST_PLUGINS="pytest_live_guard"
fi


# ── Windows location variables (computed before we drop env) ───────────────
# `env -i` forwards HOME, which is enough on POSIX. Native Windows CPython
# resolves Path.home() from USERPROFILE (or HOMEDRIVE+HOMEPATH), stdlib
# platform paths come from LOCALAPPDATA/APPDATA, ssl/sockets need SYSTEMROOT,
# and tempfile needs TEMP/TMP. Dropping them breaks collection on native
# Windows (issues #67385, #70813). These are location variables, not
# credentials, so forwarding them keeps the isolation intent intact. Each is
# only forwarded when actually set, so POSIX runs are byte-for-byte unchanged.
WIN_ENV=()
for _win_var in USERPROFILE HOMEDRIVE HOMEPATH LOCALAPPDATA APPDATA SYSTEMROOT TEMP TMP; do
  if [ -n "${!_win_var:-}" ]; then
    WIN_ENV+=("$_win_var=${!_win_var}")
  fi
done

# ── Test-runner knobs (computed before we drop env) ────────────────────────
# The runner's own documented environment knobs must survive the hermetic
# `env -i` below, or they are silent no-ops for anyone invoking this script:
#
#   * HERMES_TEST_WORKERS / PATHS / FILE_TIMEOUT / FILE_RETRIES / SLICE are
#     read by run_tests_parallel.py at argparse-default time — inside the
#     stripped environment.
#   * HERMES_TEST_IMAGE is read by tests/docker/conftest.py to skip its
#     session-scoped `docker build`. CI's docker.yml sets it to the image
#     the build step just loaded; stripping it made every per-file pytest
#     subprocess rebuild the 5GB image from a cold builder cache instead
#     (~4 min per worker per run, and the rebuilt image lacked the
#     HERMES_GIT_SHA build-arg the workflow bakes in).
#
# These are test-infrastructure knobs, not credentials — same class as the
# HERMES_RUN_SLOW_PET_TESTS / HERMES_E2E_BROWSER opt-ins already forwarded.
# Keep this an explicit allowlist (no HERMES_TEST_* glob) so the "no
# credential can leak" property stays auditable at a glance.
TEST_ENV=()
for _test_var in HERMES_TEST_IMAGE HERMES_TEST_WORKERS HERMES_TEST_PATHS \
  HERMES_TEST_FILE_TIMEOUT HERMES_TEST_FILE_RETRIES HERMES_TEST_SLICE; do
  if [ -n "${!_test_var:-}" ]; then
    TEST_ENV+=("$_test_var=${!_test_var}")
  fi
done

# ── Run in hermetic env ──────────────────────────────────────────────────────
# env -i: start with empty environment, opt-in only what we need.
# No credential var can leak — you'd have to explicitly add it here.
echo "▶ running per-file parallel test suite via run_tests_parallel.py"
echo "  (TZ=UTC LANG=C.UTF-8 PYTHONHASHSEED=0; clean env)"

cd "$REPO_ROOT"

# ── Pre-compile .pyc bytecode cache ─────────────────────────────────────────
# Each test file runs in its own subprocess via run_tests_parallel.py.
# Pre-building the bytecode cache once here (instead of each subprocess
# compiling on first import) avoids redundant work across ~2000 processes.
# Uses git to list tracked .py files (skips venv, node_modules, etc).
echo "▶ pre-compiling bytecode cache"
"$PYTHON" -m compileall -q -j 0 -- $(git ls-files '*.py') >/dev/null 2>&1 || true

echo "▶ launching test runner"
exec env -i \
  PATH="$PATH" \
  HOME="$HOME" \
  ${WIN_ENV[@]+"${WIN_ENV[@]}"} \
  ${TEST_ENV[@]+"${TEST_ENV[@]}"} \
  TZ=UTC \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONHASHSEED=0 \
  PYTHONUTF8=1 \
  ${HERMES_RUN_SLOW_PET_TESTS:+HERMES_RUN_SLOW_PET_TESTS="$HERMES_RUN_SLOW_PET_TESTS"} \
  ${HERMES_E2E_BROWSER:+HERMES_E2E_BROWSER="$HERMES_E2E_BROWSER"} \
  ${EXTRA_PYTHONPATH:+PYTHONPATH="$EXTRA_PYTHONPATH"} \
  ${EXTRA_PYTEST_PLUGINS:+PYTEST_PLUGINS="$EXTRA_PYTEST_PLUGINS"} \
  "$PYTHON" "$SCRIPT_DIR/run_tests_parallel.py" "$@"
