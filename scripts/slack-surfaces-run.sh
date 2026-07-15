#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="$REPO_ROOT/venv/bin/slack-surfaces"
REQUIRES_TOKEN=false

if [ ! -x "$CLI" ]; then
    echo "error: slack-surfaces is not installed in $REPO_ROOT/venv/bin" >&2
    echo "run ./setup-hermes.sh again to install it" >&2
    exit 1
fi

for arg in "$@"; do
    if [ "$arg" = "--apply" ]; then
        REQUIRES_TOKEN=true
        break
    fi
done

if [ "$REQUIRES_TOKEN" = true ] && [ -z "${SLACK_BOT_TOKEN:-}" ] && [ -n "${SLACK_BOT_TOKEN_OP_REF:-}" ]; then
    if ! command -v op >/dev/null 2>&1; then
        echo "error: SLACK_BOT_TOKEN_OP_REF is set but 1Password CLI 'op' is not installed" >&2
        exit 1
    fi
    export SLACK_BOT_TOKEN="$(op read "$SLACK_BOT_TOKEN_OP_REF")"
fi

if [ "$REQUIRES_TOKEN" = true ] && [ -z "${SLACK_BOT_TOKEN:-}" ]; then
    echo "error: SLACK_BOT_TOKEN is not set" >&2
    echo "set SLACK_BOT_TOKEN directly or set SLACK_BOT_TOKEN_OP_REF to an op:// reference" >&2
    exit 1
fi

exec "$CLI" "$@"
