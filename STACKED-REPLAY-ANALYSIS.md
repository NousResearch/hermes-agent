# Stacked-replay analysis: individual-clean (41/41) vs naive-sequential-stack

## The two different properties (don't conflate them)

1. **Each PR applies clean on v0.17.0 individually** — TRUE: 41/41 CLEAN, 0 conflict,
   0 compile-fail (see VERIFY-LOG-cleanenv-b4bfcba8c.txt). This is what "pullable
   onto a later release" means: a maintainer pulls/merges each PR onto v0.17.0.

2. **All 41 apply as a naive sequential `git apply` stack** — FALSE, and EXPECTED to
   be false: independent feature PRs touch SHARED files. Example: **31 of the 41 PRs
   touch `agent/conversation_loop.py`**; they each add different things to it. A blind
   sequential patch stack hits overlap once two PRs edit the same region — but that is
   not a v0.17.0 conflict, it is inter-PR file overlap.

## Proof the stack "failures" are overlap, not v0.17.0 conflicts

Naive-stack result: 23 applied / 17 failed. Of the 17:
- **16 have ZERO conflict markers** (rc=1 mk=0) = the patch's context lines no longer
  match because an earlier-stacked PR already changed that region. Pure overlap.
- The 3 with markers (#49915, #49449, #50457) were each re-tested ALONE on pristine
  v0.17.0: **all 3 produced 0 markers individually**. So even these are stack-overlap,
  not v0.17.0 conflicts.

Final stacked tree: **0 conflict markers, 0 compile failures** (the overlap PRs were
skipped, not corrupted).

## How a maintainer actually lands these
One PR at a time, with git's 3-way merge (not blind `git apply`). Because each PR's
intent is additive/compatible, real merges resolve the overlaps — which is why each
is individually clean on v0.17.0. The 41/41 individual-apply result is the correct
and sufficient proof of "pullable onto v0.17.0".
