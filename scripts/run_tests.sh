#!/usr/bin/env bash
# Canonical test runner for hermes-agent. Run this instead of calling
# `pytest` directly to guarantee your local run matches CI behavior.
#
# One runner on every host: per-file subprocess isolation via
# scripts/run_tests_parallel.py — each test FILE runs in its own
# freshly-spawned `python -m pytest <file>` process. The spawn floor is
# ~15ms on POSIX; on Windows it is ~0.5-1.5s per file (a real cost,
# ~a 6-minute floor over the full suite, paid for the state isolation
# below). There is no cross-file state pollution and each file is
# collected exactly once (pytest's per-item fixture-closure machinery —
# tens of millions of dict walks over ~42k items against the conftest's
# autouse fixtures — is paid once, not once per worker; measured 37-65s
# of pure collection that a persistent-worker model multiplies by the
# worker count).
#
# Both paths enforce the same hermetic environment: TZ=UTC, LANG=C.UTF-8,
# PYTHONHASHSEED=0, `env -i` scrubbing (credential vars can't leak), and
# proper venv activation (probes .venv, venv, then ~/.hermes/...).
#
# Usage:
#   scripts/run_tests.sh                            # full suite
#   scripts/run_tests.sh -j 4                       # cap workers/parallelism
#   scripts/run_tests.sh tests/agent/               # discover only here
#   scripts/run_tests.sh tests/foo.py               # single file
#   scripts/run_tests.sh tests/foo.py -q            # path + bare pytest flag
#   scripts/run_tests.sh -k 'pattern'               # value flags pass through too

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
VENV=""
VENV_PYTHON=""
SKIPPED_VENVS=""
for candidate in "$REPO_ROOT/.venv" "$REPO_ROOT/venv" "$HOME/.hermes/hermes-agent/venv"; do
  if [ -f "$candidate/bin/activate" ]; then
    if "$candidate/bin/python" -c 'import pytest' 2>/dev/null; then
      VENV="$candidate"
      VENV_PYTHON="$candidate/bin/python"
      break
    fi
    SKIPPED_VENVS="$SKIPPED_VENVS $candidate"
  elif [ -f "$candidate/Scripts/activate" ]; then
    if "$candidate/Scripts/python.exe" -c 'import pytest' 2>/dev/null; then
      VENV="$candidate"
      VENV_PYTHON="$candidate/Scripts/python.exe"
      break
    fi
    SKIPPED_VENVS="$SKIPPED_VENVS $candidate"
  fi
done
if [ -z "$VENV_PYTHON" ]; then
  if [ -n "${HERMES_PYTHON:-}" ] && "${HERMES_PYTHON}" -c 'import pytest' 2>/dev/null; then
    VENV_PYTHON="$HERMES_PYTHON"
  else
    echo "✗ No venv with pytest found. Install dev extras:" >&2
    echo "    uv sync --extra dev" >&2
    if [ -n "$SKIPPED_VENVS" ]; then
      echo "       (skipped for missing pytest:$SKIPPED_VENVS — install dev extras there, or create $REPO_ROOT/.venv)" >&2
    fi
    exit 1
  fi
fi
PYTHON="$VENV_PYTHON"

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

# ── Live-gateway plugin (computed before we drop env) ───────────────────────
EXTRA_PYTHONPATH=""
EXTRA_PYTEST_PLUGINS=""
if [ -f "$HOME/.hermes/pytest_live_guard.py" ]; then
  EXTRA_PYTHONPATH="$HOME/.hermes"
  EXTRA_PYTEST_PLUGINS="pytest_live_guard"
fi

# ── Our -j/--jobs flag: consumed here, forwarded via HERMES_TEST_WORKERS ────
# (run_tests_parallel.py reads that env knob as its worker cap).
JOBS="${HERMES_TEST_WORKERS:-}"
PASS_THROUGH=()
while [ $# -gt 0 ]; do
  case "$1" in
    -j|--jobs)
      JOBS="$2"; shift 2 ;;
    -j*)
      JOBS="${1#-j}"; shift ;;
    --jobs=*)
      JOBS="${1#--jobs=}"; shift ;;
    *)
      PASS_THROUGH+=("$1"); shift ;;
  esac
done
set -- ${PASS_THROUGH[@]+"${PASS_THROUGH[@]}"}
if [ -n "$JOBS" ]; then
  export HERMES_TEST_WORKERS="$JOBS"
  TEST_ENV_KNOB="HERMES_TEST_WORKERS"
fi

# ── Test-runner knobs (computed before we drop env) ──────────────────────────
#   * HERMES_TEST_IMAGE is read by tests/docker/conftest.py to skip its
#     session-scoped `docker build`. CI's docker.yml sets it to the image
#     the build step just loaded; stripping it made every per-file pytest
#     subprocess rebuild the 5GB image from a cold builder cache instead
#     (~4 min per worker per run, and the rebuilt image lacked the
#     install stamp the workflow bakes in).
#   * session-scoped `docker build`.
#   * POSIX per-file path: HERMES_TEST_WORKERS / PATHS / FILE_TIMEOUT /
#     FILE_RETRIES / SLICE are read by run_tests_parallel.py at argparse-
#     default time — inside the stripped environment.

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
#
# __NIXOS_SET_ENVIRONMENT_DONE is a NixOS platform guard, not a credential:
# without it, every login shell (bash -l) that a test spawns re-runs
# /etc/set-environment and rebuilds PATH from the system profile — which
# evicts the dev shell's python3/rg and makes terminal, process-registry,
# and ripgrep-backed search tests fail with exit 127 on NixOS hosts.
#
# HERMES_PYTHON_SRC_ROOT is the Nix dev shell's editable-install root: the
# venv's editable finder reads it at runtime to locate first-party modules.
# Stripping it breaks "import tools" in every test subprocess whose cwd is
# not the repo root (the import-guard probe runs from a tempdir).
echo "▶ running per-file parallel test suite via run_tests_parallel.py"
echo "  (TZ=UTC LANG=C.UTF-8 PYTHONHASHSEED=0; clean env)"


cd "$REPO_ROOT"

echo "▶ pre-compiling bytecode cache"
"$PYTHON" -m compileall -q -j 0 -- $(git ls-files '*.py') >/dev/null 2>&1 || true

HERMETIC_ENV=(
  PATH="$PATH"
  HOME="$HOME"
  ${WIN_ENV[@]+"${WIN_ENV[@]}"}
  ${TEST_ENV[@]+"${TEST_ENV[@]}"}
  TZ=UTC
  LANG=C.UTF-8
  LC_ALL=C.UTF-8
  PYTHONHASHSEED=0
  PYTHONUTF8=1
  ${HERMES_RUN_SLOW_PET_TESTS:+HERMES_RUN_SLOW_PET_TESTS="$HERMES_RUN_SLOW_PET_TESTS"}
  ${HERMES_E2E_BROWSER:+HERMES_E2E_BROWSER="$HERMES_E2E_BROWSER"}
  ${__NIXOS_SET_ENVIRONMENT_DONE:+__NIXOS_SET_ENVIRONMENT_DONE="$__NIXOS_SET_ENVIRONMENT_DONE"}
  ${HERMES_PYTHON_SRC_ROOT:+HERMES_PYTHON_SRC_ROOT="$HERMES_PYTHON_SRC_ROOT"}
  ${EXTRA_PYTHONPATH:+PYTHONPATH="$EXTRA_PYTHONPATH"}
  ${EXTRA_PYTEST_PLUGINS:+PYTEST_PLUGINS="$EXTRA_PYTEST_PLUGINS"}
)

exec env -i "${HERMETIC_ENV[@]}" \
  "$PYTHON" "$SCRIPT_DIR/run_tests_parallel.py" "$@"
