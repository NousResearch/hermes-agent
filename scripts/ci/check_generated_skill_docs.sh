#!/usr/bin/env bash
# Fail when website/scripts/generate-skill-docs.py output differs from the committed docs.
#
# GitHub renders the committed copies, so a regeneration that changes anything
# means the published catalogs/pages are stale. Fix: run the generator and commit.
# `git add -N` makes newly generated pages visible to `git diff`.
set -euo pipefail

git add -N website/docs website/sidebars.ts
if ! git diff --exit-code --stat -- website/docs website/sidebars.ts website/i18n; then
  echo "::error::website/scripts/generate-skill-docs.py output differs from the committed docs; run it and commit the result"
  exit 1
fi
