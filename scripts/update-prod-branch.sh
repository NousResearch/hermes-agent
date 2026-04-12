#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

printf '→ Fetching origin (with tags) and fork...\n'
git fetch origin --prune --tags
git fetch fork --prune

printf '→ Syncing local main to origin/main...\n'
git checkout main
git reset --hard origin/main

printf '→ Pushing fork/main mirror...\n'
git push fork main:main

# Determine the latest upstream release tag (strict tag-only policy)
latest_tag=$(git tag --list 'v*' --sort=-version:refname | head -1)
if [ -z "$latest_tag" ]; then
  printf '✗ No v* tags found upstream. Aborting.\n'
  exit 1
fi
printf '→ Latest upstream release: %s\n' "$latest_tag"

# Check if prod already contains this tag
if git merge-base --is-ancestor "$latest_tag" fork/prod 2>/dev/null; then
  prod_patches=$(git log --oneline "$latest_tag"..fork/prod 2>/dev/null | wc -l | tr -d ' ')
  printf '✓ fork/prod already contains %s (%s prod patches on top). Nothing to do.\n' "$latest_tag" "$prod_patches"
  exit 0
fi

printf '→ Checking prod patch count...\n'
prod_patches=$(git log --oneline "$latest_tag"..fork/prod 2>/dev/null | wc -l | tr -d ' ')
behind_count=$(git rev-list --count fork/prod.."$latest_tag" 2>/dev/null || echo "0")
printf '  prod patches: %s, upstream release commits behind: %s\n' "$prod_patches" "$behind_count"

if [ "$behind_count" -gt 100 ] && [ "$prod_patches" -gt 20 ]; then
  printf '⚠ Large drift detected (%s upstream commits, %s prod patches).\n' "$behind_count" "$prod_patches"
  printf '  Rebase may produce many conflicts. Consider a clean reset approach.\n'
  printf '  See hermes-fork-prod skill for instructions.\n'
  printf '  Continue with rebase anyway? [y/N] '
  read -r answer
  if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
    printf 'Aborted.\n'
    exit 1
  fi
fi

printf '→ Rebasing prod onto %s...\n' "$latest_tag"
git checkout prod
git rebase "$latest_tag" || {
  printf '\n⚠ Rebase paused with conflicts. Resolve them, then:\n'
  printf '  git add <resolved-files>'
  printf '  git rebase --continue\n'
  printf '  (or: git rebase --abort to give up)\n'
  exit 1
}

printf '→ Force-pushing fork/prod...\n'
git push fork prod --force-with-lease

printf '✓ Prod branch sync complete (target: %s).\n' "$latest_tag"
printf '  main: %s\n' "$(git rev-parse main)"
printf '  prod: %s\n' "$(git rev-parse prod)"
printf '  prod patches on top of %s: %s\n' "$latest_tag" "$(git log --oneline "$latest_tag"..HEAD | wc -l | tr -d ' ')"
