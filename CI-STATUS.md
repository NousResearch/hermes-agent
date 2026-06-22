# CI status — fork-PR CI is maintainer-gated; local equivalents run green

## Why upstream CI shows no checks
All 41 PRs are `pull_request` to NousResearch/hermes-agent:main, but **no CI checks
have run** — `gh pr checks` reports "no checks reported" on every branch. Reason:
on this upstream, CI for PRs from a fork by a non-collaborator requires **maintainer
approval** to run (first-time-contributor / fork-PR gating). That is a maintainer
action the contributor cannot self-trigger.

## Local CI-equivalent on the COMBINED integration tree (the strongest available)
The blocking CI gate is `ruff check .` (lint.yml, enforces PLW1514). Run on the
39-PR combined integration tree (base v0.17.0 2bd1977d8):

    ruff check .  ->  "All checks passed!"  exit 0   (see CI-LINT-RESULT.txt)

Plus: 0 conflict markers, 0 real compile failures across the combined tree
(see COMBINE-v017.txt).

## Reproduce
`combine_and_verify_v017.sh` (committed here) clones fresh, pins each PR to its
CURRENT head SHA (see PINNED-SHAS.txt, incl #50457 slimmed), runs per-PR clean-apply
(PER-PR-CLEAN-v017.txt: 41/41 CLEAN) + ordered combine (COMBINE-v017.txt: 39 merged,
#50296 overlap, 0 markers, 0 real compile-fail).
