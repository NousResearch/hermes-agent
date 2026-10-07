#!/usr/bin/env bash
# Kill all running Modal apps (sandboxes, deployments, etc.)
#
# Usage:
#   bash scripts/kill_modal.sh          # Stop rabbit-agent sandboxes
#   bash scripts/kill_modal.sh --all    # Stop ALL Modal apps

set -uo pipefail

echo "Fetching Modal app list..."
APP_LIST=$(modal app list 2>/dev/null)

if [[ "${1:-}" == "--all" ]]; then
    echo "Stopping ALL Modal apps..."
    echo "$APP_LIST" | grep -oE 'ap-[A-Za-z0-9]+' | sort -u | while read app_id; do
        echo "  Stopping $app_id"
        modal app stop "$app_id" 2>/dev/null || true
    done
else
    echo "Stopping rabbit-agent sandboxes..."
    APPS=$(echo "$APP_LIST" | grep 'rabbit-agent' | grep -oE 'ap-[A-Za-z0-9]+' || true)
    if [[ -z "$APPS" ]]; then
        echo "  No rabbit-agent apps found."
    else
        echo "$APPS" | while read app_id; do
            echo "  Stopping $app_id"
            modal app stop "$app_id" 2>/dev/null || true
        done
    fi
fi

echo ""
echo "Current rabbit-agent status:"
modal app list 2>/dev/null | grep -E 'State|rabbit-agent' || echo "  (none)"
