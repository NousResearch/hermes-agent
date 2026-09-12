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

# ── Locate python ───────────────────────────────────────────────────────────
# Probe local venvs first; fall back to the Nix devShell's editable venv
# (HERMES_PYTHON is exported by the devShell hook and ships [dev] extras:
# pytest, pytest-asyncio, pytest-timeout, ruff, ty).
#
# A candidate must have pytest INSTALLED, not merely exist. The release venv
# at ~/.hermes/hermes-agent/venv has bin/activate but no pytest, so an
# existence-only probe selected it in checkouts/worktrees without a local
# .venv — every file then died with "No module named pytest" and the run
# reported "0 tests passed" (which reads green at a glance even though the
# exit code is 1). Skip such a venv and keep probing instead.
#
# pytest alone does not make the suite TRUSTWORTHY, though: a venv missing
# other declared deps is selected, runs, and reports ordinary assertion
# failures that read as code regressions. Absent croniter costs ~22 tests/cron
# failures whose cause shows up only as a "'croniter' is not installed" log
# warning; absent psutil stops conftest's _is_own_subtree() walking the process
# tree, so _live_system_guard refuses the test's OWN children and tempts you
# into @pytest.mark.live_system_guard_bypass — disabling a real guard to hide a
# missing package; absent pytest-asyncio fails every async test with
# "Unknown pytest.mark.asyncio".
#
# So: prefer a candidate satisfying the full set, fall back to pytest-only
# (a lean venv still serves a narrow selection, and run_tests_parallel.py is
# stdlib-only), and name the gap after selecting. Warning, not a gate — the
# failure mode being prevented is a plausible wall of red, not a crash.
VENV=""
VENV_PYTHON=""
FALLBACK_VENV=""
FALLBACK_PYTHON=""
SKIPPED_VENVS=""
SUITE_IMPORTS="pytest croniter psutil pytest_asyncio yaml"

# Echo the subset of SUITE_IMPORTS that $1 cannot import, space-separated.
missing_suite_imports() {
  "$1" - "$SUITE_IMPORTS" 2>/dev/null <<'PROBE_EOF' || true
import importlib.util, sys
missing = [m for m in sys.argv[1].split() if importlib.util.find_spec(m) is None]
print(" ".join(missing))
PROBE_EOF
}

# One pass, two tiers: a complete venv wins outright; otherwise the first
# pytest-capable one is held as fallback. Both venv layouts are probed —
# POSIX bin/python and native Windows Scripts/python.exe (Git Bash / MSYS,
# where there is no bin/ at all).
for candidate in "$REPO_ROOT/.venv" "$REPO_ROOT/venv" "$HOME/.hermes/hermes-agent/venv"; do
  for _py in "$candidate/bin/python" "$candidate/Scripts/python.exe"; do
    [ -x "$_py" ] || continue
    if ! "$_py" -c 'import pytest' 2>/dev/null; then
      SKIPPED_VENVS="$SKIPPED_VENVS $candidate"
      continue
    fi
    if [ -z "$(missing_suite_imports "$_py")" ]; then
      VENV="$candidate"
      VENV_PYTHON="$_py"
      break 2
    fi
    [ -n "$FALLBACK_VENV" ] || { FALLBACK_VENV="$candidate"; FALLBACK_PYTHON="$_py"; }
  done
done
if [ -z "$VENV" ] && [ -n "$FALLBACK_VENV" ]; then
  VENV="$FALLBACK_VENV"
  VENV_PYTHON="$FALLBACK_PYTHON"
fi

if [ -n "$SKIPPED_VENVS" ]; then
  for skipped in $SKIPPED_VENVS; do
    echo "▶ skipping venv without pytest: $skipped" >&2
  done
fi

if [ -n "$VENV" ]; then
  PYTHON="$VENV_PYTHON"
  # Selection succeeded, but say so plainly if the venv cannot give
  # trustworthy results. Without this the run proceeds and misattributes
  # missing packages as code failures (see the SUITE_IMPORTS note above).
  MISSING_SUITE="$(missing_suite_imports "$PYTHON")"
  if [ -n "$MISSING_SUITE" ]; then
    echo "▶ WARNING: selected venv is missing suite dependencies:$(printf ' %s' $MISSING_SUITE)" >&2
    echo "  $VENV" >&2
    echo "  Tests will RUN but failures may be caused by these missing packages," >&2
    echo "  not by your changes. Install the project's declared dependencies:" >&2
    echo "    uv pip install --python '$PYTHON' -e '.[dev]'" >&2
    echo "  (a uv-created venv has no pip module, so 'pip install' fails there)" >&2
  fi
elif [ -n "${HERMES_PYTHON:-}" ] && [ -x "$HERMES_PYTHON" ] \
    && "$HERMES_PYTHON" -c 'import pytest' 2>/dev/null; then
  # Guard with an import check: HERMES_PYTHON may point at the RELEASE
  # venv (no pytest) when inherited from a wrapped `hermes` binary rather
  # than the devShell hook.
  PYTHON="$HERMES_PYTHON"
  echo "▶ no local venv — using Nix dev venv via HERMES_PYTHON: $PYTHON"
else
  echo "error: no virtualenv with pytest found in $REPO_ROOT/.venv or $REPO_ROOT/venv," >&2
  echo "       and HERMES_PYTHON is not a python with pytest (enter the Nix devShell or create a venv)" >&2
  if [ -n "$SKIPPED_VENVS" ]; then
    echo "       (skipped for missing pytest:$SKIPPED_VENVS — install dev extras there, or create $REPO_ROOT/.venv)" >&2
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
