#!/bin/bash
# repro.sh -- reproduce desktop-update paths against a sandboxed HERMES_HOME.
#
# Nothing here touches your real ~/.hermes or checkout. Each mode builds (or
# reuses) a disposable install under /tmp and drives the REAL code path --
# the actual installer, the actual orchestrator, the actual `hermes update`.
#
#   repro.sh shim          shim UI only: success event after 6s
#   repro.sh shim-fail     shim UI only: error event after 6s
#   repro.sh fresh         fresh install into a sandbox HERMES_HOME
#                          (scripts/install.sh, the literal user path)
#   repro.sh behind [N]    sandbox install rewound N commits (default 25),
#                          then the posix orchestrator drives it forward --
#                          the "user who hasn't updated in a while" path
#   repro.sh error         orchestrator against a broken install (missing
#                          venv) -- exercises abort + result-file + shim error
#
# The sandbox persists between runs (~/tmp is fine to nuke): fresh reuses
# nothing, behind/error reuse the last sandbox install when present because
# a from-scratch install is minutes.
#
# npm entry points (apps/desktop/package.json):
#   npm run update:shim / update:shim:fail / update:repro:fresh /
#   update:repro:behind [-- N] / update:repro:error

set -euo pipefail

MODE="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SANDBOX="${HERMES_UPDATE_REPRO_HOME:-/tmp/hermes-update-repro}"
SANDBOX_ROOT="$SANDBOX/hermes-agent"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

ensure_sandbox_install() {
  if [ -x "$SANDBOX_ROOT/venv/bin/hermes" ]; then
    say "reusing sandbox install at $SANDBOX_ROOT"
    return
  fi
  say "fresh sandbox install into $SANDBOX (this takes a while)"
  rm -rf "$SANDBOX"
  mkdir -p "$SANDBOX"
  # The literal user path: install.sh against a clone of THIS checkout, so
  # the repro reproduces what you're about to ship, not origin/main.
  git clone --quiet "$REPO_ROOT" "$SANDBOX_ROOT"
  HERMES_HOME="$SANDBOX" bash "$SANDBOX_ROOT/scripts/install.sh" --no-interactive
}

case "$MODE" in
  shim)
    HERMES_SELFTEST_HOLD_SECONDS="${HERMES_SELFTEST_HOLD_SECONDS:-6}" \
      bash "$SCRIPT_DIR/posix.sh" --self-test-ui
    ;;
  shim-fail)
    HERMES_SELFTEST_FAIL=1 HERMES_SELFTEST_HOLD_SECONDS="${HERMES_SELFTEST_HOLD_SECONDS:-6}" \
      bash "$SCRIPT_DIR/posix.sh" --self-test-ui
    ;;
  fresh)
    rm -rf "$SANDBOX"
    ensure_sandbox_install
    say "fresh install OK: $("$SANDBOX_ROOT/venv/bin/hermes" --version 2>/dev/null || echo '?')"
    ;;
  behind)
    N="${2:-25}"
    ensure_sandbox_install
    say "rewinding sandbox checkout $N commits"
    git -C "$SANDBOX_ROOT" fetch --quiet origin main || true
    git -C "$SANDBOX_ROOT" checkout --quiet main
    git -C "$SANDBOX_ROOT" reset --hard --quiet "HEAD~$N"
    say "sandbox now at: $(git -C "$SANDBOX_ROOT" log --oneline -1)"
    say "driving the orchestrator (watch the shim; log: $SANDBOX/logs/desktop-update-handoff.log)"
    HERMES_HOME="$SANDBOX" bash "$SCRIPT_DIR/posix.sh" \
      --install-root "$SANDBOX_ROOT" --branch main --desktop-pid 0 || true
    say "result file:"
    cat "$SANDBOX/.hermes-update-result.json" 2>/dev/null || echo "(none written)"
    echo
    say "sandbox after update: $(git -C "$SANDBOX_ROOT" log --oneline -1)"
    ;;
  error)
    ensure_sandbox_install
    say "breaking the sandbox venv, then driving the orchestrator"
    mv "$SANDBOX_ROOT/venv" "$SANDBOX_ROOT/venv.hidden"
    HERMES_HOME="$SANDBOX" bash "$SCRIPT_DIR/posix.sh" \
      --install-root "$SANDBOX_ROOT" --branch main --desktop-pid 0 || true
    mv "$SANDBOX_ROOT/venv.hidden" "$SANDBOX_ROOT/venv"
    say "result file (expect ok:false, exit 3):"
    cat "$SANDBOX/.hermes-update-result.json" 2>/dev/null || echo "(none written)"
    echo
    ;;
  *)
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
    exit 64
    ;;
esac
