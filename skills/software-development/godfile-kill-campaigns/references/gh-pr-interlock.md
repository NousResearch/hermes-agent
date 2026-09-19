# Native interlock + gh api patterns (campaign PR machinery)

## GitHub keyword parsing rules (learned the hard way)
- **Plain-text `Related #…` blocks ONLY.** Bold-wrapped `**Related #…**` breaks
  GitHub's keyword parser — the links never register.
- **Explicit full number lists, NEVER ranges.** `#78689-#78790` is not parsed;
  write every number out (73+ links per PR is normal).
- The linked-PRs panel + issue timelines are populated from these blocks
  (native cross-refs). Issue-side edges (issue body → PR) register fast;
  PR→epic cross-refs are async and can lag — verify after settling, not immediately.

## REST PR creation (gh pr create broken by Projects deprecation)
- Fork PR head MUST be `owner:branch` — bare `gfg/run-extract-s1-w1b` → HTTP 422
  "Validation Failed".
- `gh api` needs a **per-field `-f`**: `-f title=… -f head=… -f base=main
  -f body=…`. Missing the repeated `-f` → "accepts 1 arg(s), received 4".
- `-F field=@file` (CAPITAL F) reads the value from a file; lowercase `-f`
  treats `@path` literally.
- Commit-status queries need the PR head SHA:
  `gh api repos/O/R/pulls/N --jq '.head.sha'` then
  `gh api repos/O/R/commits/<sha>/status --jq '.state'`. Branch-name guesses 404
  for fork PRs.
- Idempotent PR-creation script: skip commit/push when `git status --porcelain`
  is empty (worktree already committed), only fire the API call.

## Issue-side interlock (both-ways requirement)
- Patch each campaign issue body with a plain-text `Related #…` line + an
  idempotency marker: `<!-- issue-edges:v1 -->`.
- **Bump the marker version to force re-patches**: adding PRs to the
  EXTRACTION list without bumping `v1`→`v2` leaves all issues "skipped" (the
  old marker blocks the rewrite).
- The interlock web covers ALL related PRs/issues by any author — parse the
  epic's per-issue table for edge PRs, don't just link your own.

## Epic kill-ledger
- Append per-PR rows (PR, shard, −lines from godfile, +lines in mixins/tests,
  module count, shipped-test count, source) + a cumulative total, via
  `-F body=@file` PATCH. Idempotent marker per append so re-runs don't duplicate.

## PR CI watch
- All checks pending for ~an hour after open is normal; verify via head-SHA
  status, not PR state. Attribution check needs BOTH email identities mapped
  (noreply AND real gmail) in AUTHOR_MAP via `scripts/add_contributor.py`.
