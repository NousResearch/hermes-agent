#!/usr/bin/env bash
# ============================================================================
# deploy-test.sh — deploy the reconnect-optimization branch to the TEST Hermes.
#
#   repo checkout : $REPO_DIR            (default /home/claude/.hermes/hermes-agent-test)
#   branch        : $BRANCH              (default reconnect-optimization)
#   unit          : $UNIT                (default hermes-test-serve)
#   listener      : 127.0.0.1:10119      (fixed — this script deploys nothing else)
#
# What it does, in order, every run:
#   1. fetch the branch from the fork and hard-reset the checkout to it
#   2. make the checkout's uv environment satisfiable (see UV_LOCK below)
#   3. build the dashboard SPA  -> hermes_cli/web_dist      (the :10119 UI)
#   4. build the Webapp bundle  -> apps/desktop/dist-webapp (npm run build:webapp)
#   5. restart the test unit and wait for /api/health == 200
#   6. print the deployed commit sha
#
# Idempotent: every step is "make it so", not "do it once"; running it twice
# back-to-back is the acceptance test for this card.
#
# NEVER touches: hermes-serve.service (:9119, production), hermes-webui.service
# (:9120), the default profile, or any checkout other than $REPO_DIR. The
# guards below fail closed rather than trusting the caller.
#
# UV_LOCK: the pinned uv (0.13.5) cannot vouch for the branch base's uv.lock on
# this VM (`uv lock --check` reports it needs updating; the lock carries a
# git-pinned dependency the offline check cannot fetch). The step below re-locks
# LOCALLY when the check fails. The regenerated uv.lock is never committed or
# pushed — the checkout is a deploy target, not a PR branch.
#
# Env overrides: REPO_DIR BRANCH REMOTE UNIT PORT HEALTH_TIMEOUT SKIP_FRONTEND=1
# ============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/claude/.hermes/hermes-agent-test}"
BRANCH="${BRANCH:-reconnect-optimization}"
REMOTE="${REMOTE:-elight}"
UNIT="${UNIT:-hermes-test-serve}"
PORT="${PORT:-10119}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
SKIP_FRONTEND="${SKIP_FRONTEND:-0}"

log()  { printf 'deploy-test: %s\n' "$*"; }
fail() { printf 'deploy-test: ERROR: %s\n' "$*" >&2; exit 1; }

# --- fail-closed guards (production is not reachable from this script) ------
[ -d "$REPO_DIR/.git" ] || fail "no git checkout at REPO_DIR=$REPO_DIR"
REPO_REAL="$(cd "$REPO_DIR" && pwd -P)"
[ "$REPO_REAL" = "/home/claude/.hermes/hermes-agent" ] \
  && fail "REPO_DIR is production's checkout ($REPO_REAL); refusing"
case "$UNIT" in
  hermes-serve*|hermes-webui*|hermes-gateway*) fail "refusing to restart $UNIT (production unit)";;
esac
[ "$UNIT" = "hermes-test-serve" ] || fail "refusing: UNIT=$UNIT is not the test unit"
[ "$PORT" = "10119" ] || fail "refusing: this script deploys only the test listener (:10119)"

command -v uv >/dev/null || fail "uv not on PATH"
command -v systemctl >/dev/null || fail "systemctl not on PATH"
command -v curl >/dev/null || fail "curl not on PATH"

cd "$REPO_DIR"

# --- 1. fetch + reset to the branch ----------------------------------------
log "fetching $REMOTE/$BRANCH"
git fetch --prune "$REMOTE" "$BRANCH" >/dev/null || fail "git fetch $REMOTE $BRANCH failed"
git show-ref --verify --quiet "refs/remotes/$REMOTE/$BRANCH" || fail "$REMOTE/$BRANCH not found after fetch"
git checkout -q "$BRANCH" 2>/dev/null || git checkout -q -B "$BRANCH" "$REMOTE/$BRANCH"
git reset --hard -q "$REMOTE/$BRANCH"

# --- 2. keep the checkout's uv environment satisfiable ---------------------
if ! uv lock --check >/dev/null 2>&1; then
  log "uv.lock does not satisfy 'uv lock --check'; re-locking locally (never pushed)"
  uv lock >/dev/null || fail "uv lock failed"
fi

# --- 3+4. front-end artifacts ----------------------------------------------
if [ "$SKIP_FRONTEND" != "1" ]; then
  log "installing JS workspaces (pnpm install --frozen-lockfile)"
  scripts/run-in-hermes-env bash -c 'pnpm install --frozen-lockfile' >/dev/null \
    || fail "pnpm install failed"
  log "building the dashboard SPA -> hermes_cli/web_dist"
  scripts/run-in-hermes-env bash -c 'pnpm --dir web run build' >/dev/null \
    || fail "dashboard SPA build failed"
  log "building the Webapp bundle -> apps/desktop/dist-webapp (npm run build:webapp)"
  scripts/run-in-hermes-env bash -c 'npm --prefix apps/desktop run build:webapp' >/dev/null \
    || fail "Webapp bundle build failed"
  [ -f hermes_cli/web_dist/index.html ] || fail "no hermes_cli/web_dist/index.html after the build"
else
  log "SKIP_FRONTEND=1: serving the web_dist already on disk"
  [ -f hermes_cli/web_dist/index.html ] || fail "no hermes_cli/web_dist/index.html on disk"
fi

# --- 5. restart the test unit, wait for health -----------------------------
log "restarting $UNIT"
systemctl --user restart "$UNIT" || fail "systemctl --user restart $UNIT failed"

DEADLINE=$(( SECONDS + HEALTH_TIMEOUT ))
while :; do
  code="$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$PORT/api/health" || true)"
  [ "$code" = "200" ] && break
  [ "$SECONDS" -ge "$DEADLINE" ] && fail "/api/health never returned 200 (last: $code after ${HEALTH_TIMEOUT}s)"
  sleep 2
done
log "/api/health = 200"

# --- 6. report -------------------------------------------------------------
SHA="$(git rev-parse HEAD)"
SUBJECT="$(git log -1 --pretty=%s)"
log "unit: $(systemctl --user is-active "$UNIT")"
log "deployed sha: $SHA  ($SUBJECT)"
