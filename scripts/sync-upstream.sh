#!/usr/bin/env bash
# Sync this fork from NousResearch/hermes-agent upstream.
#
# Idempotent: safe to run multiple times. Reports state and stops on conflict
# rather than silently producing a half-merged tree.
#
# See MAINTAINING.md for what to do after a successful merge (smoke test +
# push) and how to handle conflicts.

set -e

# Sanity: make sure we're in the fork repo.
if ! git remote get-url upstream >/dev/null 2>&1; then
  echo "ERROR: no 'upstream' remote configured. Run:"
  echo "  git remote add upstream https://github.com/NousResearch/hermes-agent.git"
  echo "  git remote set-url --push upstream no_push"
  exit 1
fi

# Sanity: working tree must be clean — we don't want to merge on top of
# uncommitted changes.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean. Commit or stash first."
  git status --short
  exit 1
fi

# Sanity: must be on main (or whatever default branch we configured).
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
  echo "ERROR: not on main (current: $CURRENT_BRANCH). Switch to main first."
  exit 1
fi

echo "Fetching upstream..."
git fetch upstream

AHEAD=$(git rev-list --count main..upstream/main)
BEHIND=$(git rev-list --count upstream/main..main)

echo ""
echo "Status:"
echo "  upstream/main is $AHEAD commits ahead of main"
echo "  main is $BEHIND commits ahead of upstream/main (our patches)"

if [[ "$AHEAD" -eq 0 ]]; then
  echo ""
  echo "Already up to date with upstream/main. Nothing to do."
  exit 0
fi

# Show a preview of what's about to be merged.
echo ""
echo "Upstream commits since last sync:"
git log --oneline main..upstream/main | head -20
TOTAL_LINES=$(git log --oneline main..upstream/main | wc -l | tr -d ' ')
if [[ "$TOTAL_LINES" -gt 20 ]]; then
  echo "... and $((TOTAL_LINES - 20)) more"
fi

echo ""
read -p "Attempt 'git merge upstream/main'? [y/N] " ANSWER
if [[ "$ANSWER" != "y" && "$ANSWER" != "Y" ]]; then
  echo "Aborted."
  exit 0
fi

echo ""
if git merge upstream/main; then
  echo ""
  echo "Merge clean. NEXT STEPS:"
  echo "  1. Run smoke test (see MAINTAINING.md → 'Smoke test')"
  echo "  2. If smoke test passes: git push origin main"
  echo "  3. Close the upstream-watch GitHub issue (if any)"
else
  echo ""
  echo "MERGE CONFLICT. Resolve manually, then commit. See MAINTAINING.md → 'Conflict scenarios'."
  echo ""
  echo "Conflicted files:"
  git diff --name-only --diff-filter=U
  exit 2
fi
