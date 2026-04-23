#!/usr/bin/env bash
# GitLab environment setup for Hermes agent
# Source this script to configure GitLab access:
#   source "${HERMES_HOME:-$HOME/.hermes}/skills/gitlab/gitlab-auth/scripts/glab-env.sh"

set -euo pipefail

# Load from .env if not already set
if [ -z "${GITLAB_TOKEN:-}" ]; then
    if [ -f "${HERMES_HOME:-$HOME/.hermes}/.env" ]; then
        GITLAB_TOKEN=$(grep "^GITLAB_TOKEN=" "${HERMES_HOME:-$HOME/.hermes}/.env" 2>/dev/null | head -1 | cut -d= -f2 | tr -d '\n\r')
        export GITLAB_TOKEN
    fi
fi

if [ -z "${GITLAB_URL:-}" ]; then
    if [ -f "${HERMES_HOME:-$HOME/.hermes}/.env" ]; then
        GITLAB_URL=$(grep "^GITLAB_URL=" "${HERMES_HOME:-$HOME/.hermes}/.env" 2>/dev/null | head -1 | cut -d= -f2 | tr -d '\n\r')
        export GITLAB_URL
    fi
fi

# Defaults
GITLAB_URL="${GITLAB_URL:-https://gitlab.com}"
export GITLAB_URL

# Verify
if [ -n "${GITLAB_TOKEN:-}" ]; then
    echo "GitLab configured: ${GITLAB_URL} (token set)"
else
    echo "WARNING: GITLAB_TOKEN not set — GitLab tools will be unavailable"
    echo "Set GITLAB_TOKEN in ~/.hermes/.env or export it directly"
fi
