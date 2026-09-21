#!/bin/bash
# Wiki daily sync - commits and pushes any changes to GitHub
# Silent on no changes, brief log on push
#
# Env overrides (P13 isolation / local disposable runs):
#   WIKI_SYNC_DRY_RUN=1 — print the git commands that would run, do not
#                          push, commit, or touch the working tree
#   WIKI_SYNC_DIR        — override the wiki checkout (default $HOME/wiki)
#   WIKI_SYNC_EXPECTED_REMOTE — exact permitted origin URL
set -uo pipefail
# NOTE: errexit (set -e) intentionally OFF so the dry-run path and the
# auth-failure path can print diagnostics without aborting early.

DRY_RUN=0
if [ "${WIKI_SYNC_DRY_RUN:-0}" = "1" ]; then DRY_RUN=1; fi

export HOME=${HOME:-/home/kensei}
export GIT_TERMINAL_PROMPT=0
export PATH="/usr/bin:${HOME}/.local/bin:${PATH:-}"

WIKI_DIR="${WIKI_SYNC_DIR:-/home/kensei/docs/wiki}"
EXPECTED_REMOTE="${WIKI_SYNC_EXPECTED_REMOTE:-https://github.com/Sahil-SS9/kensei-wiki.git}"

# Dry-run short-circuits before cd so the target dir need not exist.
if [ "${DRY_RUN}" = "1" ]; then
    echo "dry-run: cd ${WIKI_DIR}"
    echo "dry-run: would push unpushed commits, detect new work, commit + push"
    exit 0
fi

cd "${WIKI_DIR}" 2>/dev/null || {
    echo "ERROR: cannot cd to ${WIKI_DIR}"
    exit 1
}

# Serialize the index/commit/push sequence and fail closed if migration wiring
# points origin at a checked-out staging clone instead of the GitHub authority.
exec 9>"${WIKI_DIR}/.git/wiki-daily-sync.lock"
flock -n 9 || {
    echo "ERROR: wiki sync already running"
    exit 75
}
REMOTE_URL=$(git remote get-url origin 2>/dev/null) || {
    echo "ERROR: wiki origin is not configured"
    exit 1
}
if [[ "${REMOTE_URL}" != "${EXPECTED_REMOTE}" ]]; then
    echo "ERROR: refusing unsafe wiki origin: ${REMOTE_URL}"
    exit 1
fi

# --- Auth setup ---
# Fetch token from env (set by Hermes config) or gh CLI as fallback.
# Use GIT_ASKPASS to bypass the gh credential helper which can return stale cached tokens.
if [[ -n "${GITHUB_PERSONAL_ACCESS_TOKEN:-}" ]]; then
    GH_TOKEN="$GITHUB_PERSONAL_ACCESS_TOKEN"
elif [[ -n "${GH_TOKEN:-}" ]]; then
    :  # already set
else
    GH_TOKEN=$(gh auth token 2>/dev/null) || {
        echo "ERROR: gh auth token failed - is gh logged in?"
        gh auth status 2>&1
        exit 1
    }
fi
export GH_TOKEN
# Use GitHub CLI's credential helper so no token is written to disk or remote config.
unset GIT_ASKPASS
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=credential.helper
export GIT_CONFIG_VALUE_0="!gh auth git-credential"

# Refresh the checkpoint before deciding whether a push is safe. Automatic
# reconciliation is forbidden because generated wiki content can be duplicated.
git fetch --prune origin 2>&1 || exit 128
read -r BEHIND AHEAD < <(git rev-list --left-right --count origin/main...HEAD)
if (( BEHIND > 0 && AHEAD > 0 )); then
    echo "ERROR: wiki branch diverged from origin/main (behind=${BEHIND} ahead=${AHEAD})"
    exit 1
fi
if (( BEHIND > 0 )); then
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "ERROR: wiki branch is behind origin/main with local work; refusing automatic merge"
        exit 1
    fi
    git merge --ff-only origin/main 2>&1 || exit 1
fi

# 1. Push any unpushed commits from a previous failed run (idempotent recovery)
if [[ -n "$(git log origin/main..HEAD --oneline 2>/dev/null)" ]]; then
  git push origin main 2>&1 || exit 128
fi

# 2. Detect new work
if git diff --quiet && git diff --cached --quiet && [ -z "$(git status --porcelain)" ]; then
    exit 0
fi

# 3. Commit + push (with retry - GitHub has transient auth hiccups)
git add -A
git commit -m "wiki sync: $(date +%Y-%m-%d_%H:%M:%S)"
for attempt in 1 2 3; do
  git push origin main 2>&1 && exit 0
  sleep $((attempt * 5))
done
exit 128
