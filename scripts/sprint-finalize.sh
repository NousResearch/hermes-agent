#!/usr/bin/env bash
# ============================================================
# sprint-finalize.sh — Consolidate sprint commits before review
# ============================================================
# Squashes all commits on the current sprint branch into a
# single clean commit, preserving correct authorship.
#
# Usage:
#   scripts/sprint-finalize.sh ["commit message"]
#
# The script:
#   1. Verifies we're on a sprint/ branch
#   2. Counts commits ahead of main
#   3. Squashes all into one commit (preserves author)
#   4. Force pushes the consolidated branch
#   5. PR automatically updates on GitHub
#
# After this, click "Approve" on the PR to merge.
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

log() { echo "  [sprint-finalize] $*"; }
die() { echo >&2 "  [sprint-finalize] FATAL: $*"; exit 1; }

# ---------------------------------------------------------
# Step 1 — Verify we're on a sprint branch
# ---------------------------------------------------------
BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "")

if [ -z "$BRANCH" ]; then
    die "Not on a branch (detached HEAD)."
fi

if [[ ! "$BRANCH" =~ ^sprint/ ]]; then
    die "Not on a sprint branch (current: $BRANCH). Run this on a sprint/* branch."
fi

log "Branch: $BRANCH"

# ---------------------------------------------------------
# Step 2 — Count commits ahead of main
# ---------------------------------------------------------
git fetch origin main --quiet 2>/dev/null || true

AHEAD=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo "0")
BEHIND=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo "0")

if [ "$AHEAD" -eq 0 ]; then
    die "No commits ahead of main. Nothing to consolidate."
fi

if [ "$BEHIND" -gt 0 ]; then
    die "Branch is $BEHIND commit(s) behind main. Rebase first."
fi

log "Commits ahead of main: $AHEAD"

# ---------------------------------------------------------
# Step 3 — Build consolidated commit message
# ---------------------------------------------------------
# Collect all commit messages
MESSAGES=$(git log --format='%s' origin/main..HEAD | tac)

# Use custom message if provided, otherwise auto-generate
if [ -n "${1:-}" ]; then
    COMMIT_MSG="$1"
else
    # Auto-generate from commit messages
    COMMIT_MSG="feat: sprint ${BRANCH##sprint/}

Consolidated from $AHEAD commits:

$(echo "$MESSAGES" | sed 's/^/- /')"
fi

log "Consolidated commit message:"
echo "$COMMIT_MSG" | sed 's/^/    /'

# ---------------------------------------------------------
# Step 4 — Squash all commits into one
# ---------------------------------------------------------
log "Squashing $AHEAD commits..."

# Get the merge base with main
MERGE_BASE=$(git merge-base origin/main HEAD)

# Soft reset to merge base (keeps all changes staged)
git reset --soft "$MERGE_BASE"

# Create new commit with correct authorship
# Author: John P. Alencar (the human)
# Committer: johnalencar-agent (the agent)
GIT_AUTHOR_NAME="John P. Alencar"
GIT_AUTHOR_EMAIL="johnpalencar@hotmail.com"
export GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL

git commit -m "$COMMIT_MSG"

log "Consolidated into 1 commit."

# ---------------------------------------------------------
# Step 5 — Force push
# ---------------------------------------------------------
log "Force pushing consolidated branch..."
git push --force origin "$BRANCH" 2>&1

log "Done! Branch $BRANCH consolidated and pushed."
log ""
log "Next step: Approve the PR on GitHub to merge."
log "  https://github.com/NousResearch/hermes-agent/pulls"
