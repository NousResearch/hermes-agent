#!/bin/bash
# Wiki daily sync - commits and pushes any changes to GitHub
# Silent on no changes, brief log on push
#
# Env overrides (P13 isolation / local disposable runs):
#   WIKI_SYNC_DRY_RUN=1 — print the git commands that would run, do not
#                          push, commit, or touch the working tree
#   WIKI_SYNC_DIR        — override the wiki checkout (default $HOME/wiki)
set -uo pipefail
# NOTE: errexit (set -e) intentionally OFF so the dry-run path and the
# auth-failure path can print diagnostics without aborting early.

DRY_RUN=0
if [ "${WIKI_SYNC_DRY_RUN:-0}" = "1" ]; then DRY_RUN=1; fi

export HOME=${HOME:-/home/kensei}
export GIT_TERMINAL_PROMPT=0
export PATH="/usr/bin:${HOME}/.local/bin:${PATH:-}"

WIKI_DIR="${WIKI_SYNC_DIR:-${HOME}/wiki}"

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
