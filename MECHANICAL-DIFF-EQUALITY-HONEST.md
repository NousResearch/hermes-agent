# Mechanical diff-equality — the HARD, method-independent result (2026-06-22)

The Council demanded line-level set-equality, not my hunk heuristic (which had
misclassified content this session). Here it is, with the heuristic's error corrected.

## Method (no blame-to-commit heuristic)
- TARGET = normalized-unique ADDED lines in `git diff v0.16.0..HEAD -- ./src` (baks/intel excluded)
- PRCOV  = union of ADDED lines across all 39 open code PR diffs (each `base..head`)
- EXCL   = ADDED lines in the 18 enumerated excluded FILES (withdrawn/superseded/discard)
- RESIDUAL = TARGET − PRCOV − EXCL

## Result (re-runnable: `SRC=<checkout> python3 mechanical_diff_equality.py`)
```
TARGET added-lines:                 10097
PRCOV  added-lines (39 PRs):         9972
EXCL   added-lines (excluded files): 1346
RESIDUAL = TARGET − PRCOV − EXCL:    1969
  reference excluded-infra symbols:    69
  TRUE RESIDUAL (unexplained):       1900
```

**This DISPROVES my earlier "216/216 hunks, 0 uncovered."** The hunk heuristic blamed bulk
lines to the overlay-apply commit `5e0c05647` and called them "glue"; the mechanical
line-set proof shows **1900 added lines of current ./src are in NO open PR's diff.**
(And conversely PRCOV carries ~1775 lines not in current src — the PRs and src have
genuinely diverged ~1900 lines each way.)

A measurement bug was ALSO found + fixed en route: 9 `*.bak.TIMESTAMP` forensic backups
were leaking into the delta (the `:(exclude)*.bak` glob missed timestamped names; fixed
to `*.bak*`).

## What the 1900 residual lines ARE (by file domain, baks excluded)
| Domain | lines | nature |
|---|---|---|
| private-overlay files (copilot_auth, models.py, model_metadata, models_dev, inventory, codex_models, auto_router, copilot_acp/context/catalog, opus_context, hermes_source/project_source) | 799 | account caps, hidden catalog, codex/auto-router infra — EXCLUDED by standing policy |
| core files, top: anthropic_adapter.py 377, auxiliary_client.py 147, conversation_loop.py 125, tui_gateway/server.py 103 | 1412 | the `9fec781fc` ENTANGLED 46-file mega-commit remainder + copilot-limits/identity additions, spread across files that ARE in the PRs at an EARLIER snapshot |

The core-file residual is dominated by `9fec781fc` (the entangled autopilot/cmx/kanban
mega-commit) and the copilot-limits/identity work — i.e. the content whose CLEAN parts
shipped (#50064/#49449/#48024…) and whose ENTANGLED/PRIVATE parts were deliberately left.

## HONEST conclusion (no forced green)
**Line-exact equality between the 39 PRs and current ./src HEAD is NOT achievable
without violating standing exclusion policy.** Current src has ~1900 added lines of
entangled/private/post-PR-drift content. To make the PR set line-exact with current HEAD
would require OPTION A: re-cut the PRs against current src — which RE-PULLS the
`9fec781fc` entangled mega-commit, account-specific caps, auto_router, codex_version, and
opus-context private overlay that the user repeatedly told me NOT to PR.

So the goal "all the changes in ./src" has two irreconcilable readings, and choosing
between them is a SCOPE decision only the user can make:
- **OPTION A** (literal line-exact): re-cut PRs against current HEAD; accepts re-pulling
  excluded private/entangled content. Violates standing policy as written.
- **OPTION B** (contributable snapshot): the 39 PRs ARE the contributable surface; the
  1900-line residual is private/entangled/cosmetic/post-PR-drift, enumerated here by file.

This is the genuine operator gate. I am NOT declaring done, and I am NOT hiding the 1900
behind a heuristic "0". The number is real, mechanically proven, and the disposition is
the user's call. My recorded reasoned default remains OPTION B (the residual is
overwhelmingly the content the user excluded), but the Council is correct that this
narrows the literal goal, and only the user can ratify that narrowing.
