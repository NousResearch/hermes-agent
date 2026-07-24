#!/usr/bin/env bash
# GitHub environment detection helper for Hermes Agent skills.
#
# Usage (via terminal tool):
#   source skills/github/github-auth/scripts/gh-env.sh
#
# After sourcing, these variables are set:
#   GH_AUTH_METHOD  - "gh", "curl", or "none"
#   GITHUB_TOKEN    - personal access token (set if method is "curl")
#   GH_USER         - GitHub username
#   GH_OWNER        - repo owner  (only if inside a git repo with a github remote)
#   GH_REPO         - repo name   (only if inside a git repo with a github remote)
#   GH_OWNER_REPO   - owner/repo  (only if inside a git repo with a github remote)

# --- Resolve the Hermes .env location ---
#
# HERMES_HOME is bridged into tool subprocesses. Under the official Docker
# layout HERMES_HOME=/opt/data while the subprocess HOME is redirected to
# /opt/data/home (profile isolation), so a bare ~/.hermes/.env expands to
# /opt/data/home/.hermes/.env — the WRONG file — while the real secrets live
# at /opt/data/.env. Resolve against HERMES_HOME first, and keep a home-dir
# fallback for bare installs / unexpected HERMES_HOME-unset subprocesses.
HERMES_ENV_FILE="${HERMES_HOME:-$HOME/.hermes}/.env"
HERMES_ENV_FALLBACK="$HOME/.hermes/.env"
GH_ENV_SOURCE=""

# --- Auth detection ---

GH_AUTH_METHOD="none"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
GH_USER=""

if command -v gh &>/dev/null && gh auth status &>/dev/null 2>&1; then
    GH_AUTH_METHOD="gh"
    GH_USER=$(gh api user --jq '.login' 2>/dev/null)
elif [ -n "$GITHUB_TOKEN" ]; then
    GH_AUTH_METHOD="curl"
    GH_ENV_SOURCE="environment"
elif [ -f "$HERMES_ENV_FILE" ] && grep -q "^GITHUB_TOKEN=" "$HERMES_ENV_FILE" 2>/dev/null; then
    GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" "$HERMES_ENV_FILE" | head -1 | cut -d= -f2 | tr -d '\n\r')
    if [ -n "$GITHUB_TOKEN" ]; then
        GH_AUTH_METHOD="curl"
        GH_ENV_SOURCE="$HERMES_ENV_FILE"
    fi
elif [ "$HERMES_ENV_FALLBACK" != "$HERMES_ENV_FILE" ] && [ -f "$HERMES_ENV_FALLBACK" ] && grep -q "^GITHUB_TOKEN=" "$HERMES_ENV_FALLBACK" 2>/dev/null; then
    GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" "$HERMES_ENV_FALLBACK" | head -1 | cut -d= -f2 | tr -d '\n\r')
    if [ -n "$GITHUB_TOKEN" ]; then
        GH_AUTH_METHOD="curl"
        GH_ENV_SOURCE="$HERMES_ENV_FALLBACK"
    fi
elif [ -f "$HOME/.git-credentials" ] && grep -q "github.com" "$HOME/.git-credentials" 2>/dev/null; then
    GITHUB_TOKEN=$(grep "github.com" "$HOME/.git-credentials" | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|')
    if [ -n "$GITHUB_TOKEN" ]; then
        GH_AUTH_METHOD="curl"
        GH_ENV_SOURCE="$HOME/.git-credentials"
    fi
fi

# Resolve username for curl method
if [ "$GH_AUTH_METHOD" = "curl" ] && [ -z "$GH_USER" ]; then
    GH_USER=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
        https://api.github.com/user 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('login',''))" 2>/dev/null)
fi

# --- Repo detection (if inside a git repo with a GitHub remote) ---

GH_OWNER=""
GH_REPO=""
GH_OWNER_REPO=""

_remote_url=$(git remote get-url origin 2>/dev/null)
if [ -n "$_remote_url" ] && echo "$_remote_url" | grep -q "github.com"; then
    GH_OWNER_REPO=$(echo "$_remote_url" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
    GH_OWNER=$(echo "$GH_OWNER_REPO" | cut -d/ -f1)
    GH_REPO=$(echo "$GH_OWNER_REPO" | cut -d/ -f2)
fi
unset _remote_url

# --- Summary ---

echo "GitHub Auth: $GH_AUTH_METHOD"
[ -n "$GH_USER" ]       && echo "User: $GH_USER"
[ -n "$GH_OWNER_REPO" ] && echo "Repo: $GH_OWNER_REPO"
[ -n "$GH_ENV_SOURCE" ] && echo "Token source: $GH_ENV_SOURCE"
if [ "$GH_AUTH_METHOD" = "none" ]; then
    # Make a "none" result self-diagnosing: show the exact path checked plus
    # HERMES_HOME/HOME so the common Docker mismatch (.env at $HERMES_HOME/.env,
    # but HOME redirected so ~/.hermes/.env points elsewhere) is obvious instead
    # of silent. Reported on Discord/GitHub against the docker-compose setup.
    echo "⚠ Not authenticated — see github-auth skill"
    echo "  Checked: $HERMES_ENV_FILE ($([ -f "$HERMES_ENV_FILE" ] && echo present || echo missing))"
    if [ "$HERMES_ENV_FALLBACK" != "$HERMES_ENV_FILE" ]; then
        echo "  Checked: $HERMES_ENV_FALLBACK ($([ -f "$HERMES_ENV_FALLBACK" ] && echo present || echo missing))"
    fi
    echo "  HERMES_HOME=${HERMES_HOME:-<unset>}  HOME=$HOME"
    echo "  (In Docker the .env lives at \$HERMES_HOME/.env, e.g. /opt/data/.env — not ~/.hermes/.env)"
fi

export GH_AUTH_METHOD GITHUB_TOKEN GH_USER GH_OWNER GH_REPO GH_OWNER_REPO
