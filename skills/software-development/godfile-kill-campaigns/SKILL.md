---
name: godfile-kill-campaigns
description: "Run godfile kill campaigns: shard, interlink, and lock."
version: 0.1.0
author: Andrex Ibiza (andrexibiza), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [godfile, sharding, refactor, campaign, epic, interlock]
    category: software-development
    related_skills: [github, hermes-agent-skill-authoring]
---

# Godfile Kill Campaigns

Ship a god-file sharding campaign (epic #78647 family): per-file tracking
issues, byte-verbatim extraction PRs, and the PR↔issue interlock that makes
each kill auditable. Shard, never revert; the goal is owners at or under
2,000 physical lines with no façade regrowth.

## When to Use

- A sharding epic (#78647 and its per-file trackers) is active or being set up.
- You are cutting one extraction PR of a file already over the 2K ceiling.
- The maintainer asks for a kill campaign, scoreboard, or interlock audit.

Don't use for: single bug-fix PRs (plain `Fixes #N` suffices); refactors with
no decomposition ledger.

## Procedure

1. **Confirm the target on current main.** `wc -l` the file; read its per-file
   tracker issue; check no other open PR already owns the domain you are about
   to cut (`gh pr list --state open --search "<file>"`). Done when the tracker
   is live, the file is still over the bar, and the domain is unclaimed.
2. **Cut one coherent extraction PR per domain.** Branch from current
   `origin/main`, never from a stale wave. Pure move: byte-verbatim lines,
   the same exports, no behavior change; keep rationale comments and
   monkeypatch/test surfaces intact, updating patch-target tests in the same
   PR. Done when `tsc`/`eslint`/`ruff` and the narrowest tests are green.
3. **Interlink both directions.** The PR body carries `Part of #<epic>` AND
   `Part of #<file-tracker>` as separate lines (a combined line or
   `Progress on #N` binds nothing); the tracker thread carries the literal
   `#<PR>` token. Done when
   `gh api repos/O/R/issues/<tracker>/timeline --jq '.[] | select(.event=="cross-referenced") | .source.issue.number'`
   lists your PR.
4. **Post the scoreboard.** On the tracker, after each merged slice:
   `| slice | PR | module | line delta | test evidence |`. Update the epic's
   status row in the same pass. Done when the tracker and epic reflect the
   exact merged head.
5. **Audit before claiming a kill.** Every tracker must have binding PRs,
   every PR its trackers, no duplicate PRs for the same slice, and the file
   at or under 2,000 lines. Done when the epic's acceptance contract (all
   owners under the bar, CI green at the final head) is met and stated with
   receipts.

## Pitfalls

- **Stale-base extractions.** A PR cut from an old `main` conflicts on every
  hunk; rebase check the file's line count against `main` before opening.
- **Invented rosters.** Pull real issues/PRs via `gh`; a fabricated mess
  record corrupts the ledger and costs trust.
- **`closingIssuesReferences` on refactor PRs.** It stays empty for `Part of`
  links by design — the timeline cross-reference API is the registry to check.
- **Façade regrowth.** The ceiling is not a target: an extraction that freezes
  a second oversized coordination point fails the epic's contract.

## Verification

- [ ] Tracker issue live; domain unclaimed on current main
- [ ] Extraction is a pure move; narrow tests green at the exact head
- [ ] `Part of` lines separate; timeline cross-reference lists the PR
- [ ] Scoreboard + epic row updated with the merged head
- [ ] Kill claim states line count and CI receipts, no self-certification
