#!/usr/bin/env bash
# Pre-commit hook: reject branches with no common ancestor with main.
# Equivalent to .github/workflows/history-check.yml
set -euo pipefail

if ! git merge-base origin/main HEAD >/dev/null 2>&1; then
    echo "ERROR: This branch has no common ancestor with main."
    echo "Your branch history is disconnected from main."
    echo "Rebase onto main: git fetch origin main && git rebase origin/main"
    exit 1
fi
