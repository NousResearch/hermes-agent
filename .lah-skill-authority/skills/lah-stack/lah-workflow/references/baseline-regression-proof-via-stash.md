# Baseline Regression Proof via Stash (prove failures pre-exist before merge)

Use when a mission demands proof that full-suite failures on a modified tree
are PRE-EXISTING (e.g. "do not hide pre-existing failures", "baseline proof
before commits"). Proven on CLOE_LAHB_DUAL_FUNNEL_OBSERVABILITY_REPAIR_
CONTINUATION_V1 (26 failures proven identical on clean origin/main).

## Recipe (fully reversible — NEVER `git reset --hard` the mission worktree)

1. Record the exact pre-baseline state:
   `git status --short > pre-baseline-status.txt`

2. Stash ONLY the tracked mission files — do NOT use `-u`:
   ```
   git stash push -m "mission-work-v1" -- <file1> <file2> ...
   ```
   Rationale: plain `git stash` leaves untracked files in place (they would
   pollute the suite run and shift failure numbering); `git stash -u` would
   try to stash untracked `node_modules` (huge, often not gitignored).

3. Move untracked NEW test files OUT of the suite glob (don't rely on stash):
   `mv test/dual-funnel-*.test.js /tmp/name.bak`

4. Checkout the clean authority:
   - If the worktree HEAD tree == origin/main tree (verify with
     `git diff <head> <origin-main-sha>` — empty diff means equivalent),
     staying at HEAD is fine; otherwise detached-checkout the merge tip:
     `git checkout <origin-main-sha>` (detached, safe).
5. Run the FULL suite capturing output to a file:
   `timeout 1500 node --test test/*.test.js > baseline-full-suite.txt 2>&1`
   Then extract failure names: `grep "^not ok" baseline-full-suite.txt`.
6. Restore EXACTLY: `git checkout <mission-branch>` → `git stash pop` →
   `mv /tmp/name.bak test/name.test.js` → verify
   `git status --short` matches the pre-baseline file.

## Compare correctly — same NAMES, not just same count

A modified tree with N new tests shifts EVERY failure number by +N (node:test
numbers sequentially per file). Identical names + uniform shift = same set.
Extract names on both runs and diff them; do not compare counts alone.

Baseline run may also be much faster than expected: `node --test` runs test
FILES in parallel — a 3k-test suite can finish in seconds. A fast exit is not
a failure; check the captured file.

## Classification rule

- Clean origin/main has exactly the same failures → PREEXISTING_CONFIRMED.
- Clean has FEWER → isolate ONLY the delta, TDD-repair, re-run targeted +
  full before any commit.
- Billing/CI failures ("recent account payments have failed", 0ms runs) are
  environment, not code — classify separately from code correctness; local
  test runs are the code truth.

## Related

- Post-merge deploy verification: LAH-Brain auto-deploys on merge — confirm
  via production `GET /version` == merge SHA before declaring deployment.
- Admin merge with billing-dead CI: `gh pr merge <PR> --admin --merge
  --match-head-commit <head-sha>` (verify `--json state,closed,mergedAt,
  mergeCommit` after; `merged` is not a valid gh field).
