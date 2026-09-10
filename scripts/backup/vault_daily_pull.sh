#!/bin/bash
# Vault daily pull — fetches latest project docs from GitHub
# Silent on success, errors printed on failure
# Includes retry with exponential backoff for transient auth/network failures
#
# Env overrides (P13 isolation / local disposable runs):
#   VAULT_PULL_DRY_RUN=1 — print the git commands that would run, do not
#                          pull, probe auth, or touch the working tree
#   VAULT_PULL_DIR        — override the vault checkout
#                          (default $HOME/vaults/obsidian-master)
set -uo pipefail
# NOTE: errexit (set -e) intentionally OFF so the dry-run path and the
# auth-failure path can print diagnostics without aborting early.

DRY_RUN=0
if [ "${VAULT_PULL_DRY_RUN:-0}" = "1" ]; then DRY_RUN=1; fi

export HOME=${HOME:-/home/kensei}
export GIT_TERMINAL_PROMPT=0
export PATH="/usr/bin:${HOME}/.local/bin:${PATH:-}"

VAULT_DIR="${VAULT_PULL_DIR:-${HOME}/vaults/obsidian-master}"

if [ "${DRY_RUN}" = "1" ]; then
    echo "dry-run: would cd ${VAULT_DIR} and git pull --ff-only origin main"
    exit 0
fi

cd "${VAULT_DIR}" 2>/dev/null || {
    echo "ERROR: cannot cd to ${VAULT_DIR}"
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
        echo "ERROR: gh auth token failed — is gh logged in?"
        gh auth status 2>&1
        exit 1
    }
fi
export GH_TOKEN
# Use GitHub CLI's credential helper so no token is written to disk or remote
# config. (Older design used a GIT_ASKPASS helper script at
# ${HOME}/.hermes/scripts/git-askpass.sh — retired/archived 2026-07; wiki_daily_sync.sh
# already migrated to this credential-helper pattern.)
unset GIT_ASKPASS
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=credential.helper
export GIT_CONFIG_VALUE_0="!gh auth git-credential"

# --- Early auth probe ---
echo "INFO: probing GitHub auth..."
gh auth status 2>&1 | head -1
git ls-remote origin HEAD >/dev/null 2>&1 || {
    echo "ERROR: git ls-remote failed — auth may be broken"
    exit 1
}

# --- Retry loop: up to 3 attempts with 10s, 30s backoff ---
MAX_ATTEMPTS=3
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    GIT_OUTPUT=$(git pull --ff-only origin main 2>&1) && {
        # Success — silent exit
        exit 0
    }
    GIT_EXIT=$?

    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
        SLEEP=$(( attempt * 10 ))
        echo "WARN: git pull attempt $attempt/$MAX_ATTEMPTS failed (exit $GIT_EXIT), retrying in ${SLEEP}s..."
        sleep "$SLEEP"
    else
        echo "ERROR: git pull failed after $MAX_ATTEMPTS attempts (last exit $GIT_EXIT)"
        echo "$GIT_OUTPUT"
        exit 1
    fi
done
