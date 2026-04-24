#!/usr/bin/env bash
# update-prod-branch.sh — Sync fork with upstream and integrate into prod
#
# Branch model:
#   origin/main  → upstream (NousResearch/hermes-agent)
#   fork/main    → clean upstream mirror
#   fork/prod    → integration branch = upstream + our custom patches
#
# Workflow:
#   1. Fetch all remotes
#   2. Rebase local main onto origin/main (upstream mirror)
#   3. Push fork/main
#   4. Rescue any prod-only commits (merged directly to fork/prod)
#   5. Merge main into prod (prod = upstream + our patches, always)
#   6. Push fork/prod
#
# Usage:
#   cd ~/.hermes/hermes-agent && ./scripts/update-prod-branch.sh
#   DRY_RUN=1 cd ~/.hermes/hermes-agent/scripts/update-prod-branch.sh  # preview only

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

DRY_RUN="${DRY_RUN:-0}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[sync]${NC} $*"; }
warn() { echo -e "${YELLOW}[sync]${NC} $*" >&2; }
err()  { echo -e "${RED}[sync]${NC} $*" >&2; }

run() {
  if [ "$DRY_RUN" = "1" ]; then
    echo -e "  ${CYAN}(dry-run)${NC} $*"
  else
    "$@"
  fi
}

# ── Pre-flight ──────────────────────────────────────────────
if [ -n "$(git status --porcelain)" ]; then
  err "Working tree is dirty. Commit or stash first."
  git status --short
  exit 1
fi

CURRENT_BRANCH="$(git branch --show-current)"
log "Current branch: ${CURRENT_BRANCH}"

# ── Step 1: Fetch ───────────────────────────────────────────
log "Fetching remotes..."
run git fetch origin --tags --prune
run git fetch fork --prune

UPSTREAM_HEAD="$(git rev-parse origin/main)"
LOCAL_MAIN="$(git rev-parse main 2>/dev/null || echo 'none')"
FORK_PROD="$(git rev-parse fork/prod 2>/dev/null || echo 'none')"

log "origin/main:      ${UPSTREAM_HEAD:0:8}"
log "local main:       ${LOCAL_MAIN:0:8}"
log "fork/prod:        ${FORK_PROD:0:8}"

# ── Step 2: Check if there's anything to do ─────────────────
BEHIND="$(git rev-list --count main..origin/main 2>/dev/null || echo '0')"
if [ "$BEHIND" = "0" ] && [ "$DRY_RUN" = "0" ]; then
  # Still need to check for prod-only commits
  PROD_ONLY="$(git rev-list --count main..fork/prod 2>/dev/null || echo '0')"
  if [ "$PROD_ONLY" = "0" ]; then
    log "Already up to date. Nothing to do."
    exit 0
  else
    log "Upstream is current but fork/prod has ${PROD_ONLY} custom commit(s). Integrating..."
  fi
fi

# ── Step 3: Update main (upstream mirror) ──────────────────
if [ "$BEHIND" != "0" ]; then
  log "Rebasing main onto origin/main (${BEHIND} upstream commits)..."
  run git checkout main
  run git rebase origin/main

  if [ "$?" != "0" ]; then
    err "Rebase failed! Resolve conflicts then run:"
    err "  git rebase --continue"
    err "  Then re-run this script."
    exit 1
  fi

  log "Pushing fork/main..."
  run git push fork main:main
else
  log "main is already at origin/main."
fi

# ── Step 4: Rescue prod-only commits ───────────────────────
PROD_ONLY_COMMITS=$(git rev-list --reverse main..fork/prod 2>/dev/null || true)

if [ -n "$PROD_ONLY_COMMITS" ]; then
  PROD_ONLY_COUNT=$(echo "$PROD_ONLY_COMMITS" | wc -l | tr -d ' ')
  warn "Found ${PROD_ONLY_COUNT} commit(s) on fork/prod not on main:"
  echo "$PROD_ONLY_COMMITS" | while read -r hash; do
    SUBJECT="$(git log -1 --format='%s (%h)' "$hash")"
    warn "  - ${SUBJECT}"
  done

  # Cherry-pick each onto main so they're part of our base
  log "Cherry-picking prod-only commits onto main..."
  echo "$PROD_ONLY_COMMITS" | while read -r hash; do
    SUBJECT="$(git log -1 --format='%s' "$hash")"
    log "  → ${SUBJECT}"
    run git cherry-pick "$hash" || {
      err "Cherry-pick of ${hash:0:8} failed! Resolve conflicts:"
      err "  git cherry-pick --continue  (or --abort)"
      exit 1
    }
  done

  log "Pushing updated main with rescued commits..."
  run git push fork main:main
else
  log "No prod-only commits to rescue."
fi

# ── Step 5: Integrate into prod ────────────────────────────
log "Merging main into prod..."
run git checkout prod

# Use --ff-only when possible, otherwise merge
if git merge-base --is-ancestor main prod 2>/dev/null; then
  log "prod is already ahead of main (fast-forward). No merge needed."
elif git merge-base --is-ancestor prod main 2>/dev/null; then
  log "Fast-forwarding prod to main..."
  run git merge --ff-only main
else
  log " histories diverged. Merging main into prod..."
  run git merge main -m "merge: sync upstream $(git log -1 --format='%h %s' main)"
fi

# ── Step 6: Push prod ──────────────────────────────────────
log "Pushing fork/prod..."
run git push fork prod

# ── Done ────────────────────────────────────────────────────
echo ""
log "✅ Sync complete."
log "  main  → $(git log -1 --format='%h %s' main)"
log "  prod  → $(git log -1 --format='%h %s' prod)"

# Show what's unique on prod
AHEAD_COUNT="$(git rev-list --count main..prod 2>/dev/null || echo '0')"
if [ "$AHEAD_COUNT" != "0" ]; then
  log "Custom patches on prod (${AHEAD_COUNT} commits):"
  git log --oneline main..prod | while read -r line; do
    log "  ${line}"
  done
fi

# ── Step 7: Install repo-bundled skills ───────────────────
if [ -f "${REPO_DIR}/scripts/install-repo-skills.sh" ]; then
  log "Installing repo-bundled skills..."
  run "${REPO_DIR}/scripts/install-repo-skills.sh"
fi

# Return to original branch if needed
if [ "$CURRENT_BRANCH" != "prod" ] && [ "$CURRENT_BRANCH" != "main" ]; then
  run git checkout "$CURRENT_BRANCH"
fi
