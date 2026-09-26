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
#   * The activated checkout's test environment (activates when needed)
#
# Usage:
#   scripts/run_tests.sh                            # full suite
#   scripts/run_tests.sh -j 4                       # cap parallelism
#   scripts/run_tests.sh tests/agent/               # discover only here
#   scripts/run_tests.sh tests/agent/ tests/acp_adapter/    # multiple roots
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
# The suite runs under the activated checkout's isolated test environment
# (pm.testenv: `activate` builds it beside the checkout's install state, and CI
# activates the same way). An inherited activation is re-checked against its
# inputs (scripts/_activation.sh) and re-sourced when stale, so a branch switch
# or lock edit never runs the suite against the previous dependency set.
#
# Without an activation, an explicit HERMES_PYTHON that has pytest is honored:
# the Nix devShell's editable venv and CI's minimal installer lanes provide
# one on purpose. The import check matters: a wrapped `hermes` binary exports
# HERMES_PYTHON pointing at a release venv without pytest.
_has_pytest() { [ -n "$1" ] && [ -x "$1" ] && "$1" -c 'import pytest' 2>/dev/null; }
# shellcheck source=scripts/_activation.sh
. "$SCRIPT_DIR/_activation.sh"
if [ -z "${__HERMES_ACTIVATED:-}" ] && _has_pytest "${HERMES_PYTHON:-}"; then
  PYTHON="$HERMES_PYTHON"
  echo "▶ not activated — using HERMES_PYTHON: $PYTHON"
else
  test_stamp="${__HERMES_ACTIVATED:-}"
  test_stamp="${test_stamp//\\//}"
  if ! hermes_activation_current "$REPO_ROOT" ||
     [ ! -f "${test_stamp%/*}/inputs/.test-environment" ] ||
     ! _has_pytest "${__HERMES_TEST_PYTHON:-}"; then
    echo "▶ activating $REPO_ROOT (environment missing or stale)" >&2
    # activate is written for interactive shells, not errexit/nounset.
    set +euo pipefail
    # shellcheck source=/dev/null
    . "$REPO_ROOT/activate" --
    activated=$?
    set -euo pipefail
    if [ "$activated" != 0 ]; then
      echo "error: activation failed (see above)" >&2
      exit 1
    fi
  fi
  PYTHON="${__HERMES_TEST_PYTHON:-}"
  if ! _has_pytest "$PYTHON"; then
    echo "error: activation provided no test interpreter with pytest (__HERMES_TEST_PYTHON=${PYTHON:-unset})" >&2
    exit 1
  fi
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
# Windows (issues #67385, #70813). PATHEXT is also required: without .EXE,
# PowerShell opens a native child as a document without waiting for its exit.
# These are location variables, not
# credentials, so forwarding them keeps the isolation intent intact. Each is
# only forwarded when actually set, so POSIX runs are byte-for-byte unchanged.
WIN_ENV=()
for _win_var in USERPROFILE HOMEDRIVE HOMEPATH LOCALAPPDATA APPDATA SYSTEMROOT TEMP TMP \
    ComSpec PATHEXT PROGRAMFILES ProgramFiles PROGRAMDATA ProgramData; do
  if [ -n "${!_win_var:-}" ]; then
    WIN_ENV+=("$_win_var=${!_win_var}")
  fi
done
# Native build toolchain (Windows arm64 has no wheels for every pinned C extension, so
# `uv sync` inside a PM test compiles ruamel-yaml-clib and friends). The MSVC developer
# environment is exported by scripts/build/windows-deps.ps1 into the job env; without
# INCLUDE/LIB/VSINSTALLDIR the build backend reports "Visual C++ 14.0 or greater is
# required". These describe compiler locations, not credentials.
for _tool_var in INCLUDE LIB LIBPATH VSINSTALLDIR VCINSTALLDIR VCToolsInstallDir VCToolsVersion \
    VCToolsRedistDir WindowsSdkDir WindowsSDKVersion WindowsSdkBinPath WindowsSdkVerBinPath \
    WindowsLibPath UCRTVersion UniversalCRTSdkDir VSCMD_ARG_HOST_ARCH VSCMD_ARG_TGT_ARCH VSCMD_VER \
    DevEnvDir ExtensionSdkDir Platform CARGO_HOME RUSTUP_HOME RUSTUP_TOOLCHAIN \
    CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_LINKER CC_aarch64_pc_windows_msvc CC CXX AR \
    VCPKG_ROOT OPENSSL_DIR OPENSSL_STATIC OPENSSL_LIB_DIR OPENSSL_INCLUDE_DIR; do
  if [ -n "${!_tool_var:-}" ]; then
    WIN_ENV+=("$_tool_var=${!_tool_var}")
  fi
done
# setuptools locates the compiler through vswhere under "%ProgramFiles(x86)%\Microsoft Visual
# Studio\Installer"; without that variable a primed INCLUDE/LIB still reads as "Visual C++ 14.0
# or greater is required". The parenthesised name cannot be read with ${!var}.
_pf86="$(env | sed -n 's/^ProgramFiles(x86)=//p' | head -n1)"
[ -z "$_pf86" ] || WIN_ENV+=("ProgramFiles(x86)=$_pf86")

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
#   * HERMES_E2E_REQUIRE_TUI turns a missing Ink TUI build into a failure in
#     tests/e2e/core/terminal instead of a skip (set by the e2e CI job).
#   * CI / GITHUB_ACTIONS tell suites they run on a disposable runner (e.g.
#     tests/e2e/core/upgrade runs the real updater unsandboxed only there).
#
# These are test-infrastructure knobs, not credentials — same class as the
# HERMES_RUN_SLOW_PET_TESTS / HERMES_E2E_BROWSER / HERMES_RUN_E2E opt-ins
# forwarded below.
# SSL_CERT_FILE/DIR are trust-store locations: the pinned interpreter's
# OpenSSL has no compiled-in bundle path on NixOS, so network tests (PM
# downloads, channel reads) need the host's pointer to verify TLS.
# Keep this an explicit allowlist (no HERMES_TEST_* glob) so the "no
# credential can leak" property stays auditable at a glance.
TEST_ENV=()
for _test_var in HERMES_TEST_IMAGE HERMES_TEST_WORKERS HERMES_TEST_PATHS \
  HERMES_TEST_FILE_TIMEOUT HERMES_TEST_FILE_RETRIES HERMES_TEST_SLICE \
  SSL_CERT_FILE SSL_CERT_DIR HERMES_GATEWAY_LOCK_DIR HERMES_E2E_REQUIRE_TUI CI GITHUB_ACTIONS; do
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
  ${HERMES_RUN_E2E:+HERMES_RUN_E2E="$HERMES_RUN_E2E"} \
  ${EXTRA_PYTHONPATH:+PYTHONPATH="$EXTRA_PYTHONPATH"} \
  ${EXTRA_PYTEST_PLUGINS:+PYTEST_PLUGINS="$EXTRA_PYTEST_PLUGINS"} \
  "$PYTHON" "$SCRIPT_DIR/run_tests_parallel.py" "$@"
