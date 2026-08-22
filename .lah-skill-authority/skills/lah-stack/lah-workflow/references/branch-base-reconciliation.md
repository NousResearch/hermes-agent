# Gate 0.5 — Branch-base reconciliation (MANDATORY before commit/merge)

Origin: operator mandate after a CLOE mission branch was built on `0458509`
while `main` had advanced to `302ed75` (6 evolution commits: grounded
KnowledgeCollector, FTS5 fixes, strict freshness, JSON multiprocess reload,
GIT_COMMIT snapshot, metrics/admin route). Committing/merging on the stale
base would have re-introduced or clobbered already-merged changes.

## When this applies

- The mission branch's `merge-base HEAD origin/main` == the branch's own HEAD
  (i.e. the branch was created from an old main).
- The operator states a base SHA and a "current main" SHA that differ.
- Any mission whose files overlap files that main has since evolved.

## Procedure

1. **Check base:** `git merge-base HEAD origin/main`. If it equals HEAD, the
   branch is stale. Always `git fetch origin` first.
2. **Verify main:** `git rev-parse origin/main`; confirm it contains the
   expected evolutions with `git log --oneline <old>..origin/main`.
3. **Produce compares:**
   - `git diff --stat <old-base>..origin/main`
   - `git diff --name-only <old-base>..origin/main` intersected with mission
     files → conflict set (files touched by BOTH main and mission).
   - Files main did NOT touch → clean port (copy verbatim).
4. **Backup non-destructively FIRST** — never `reset --hard`, never `clean`,
   no destructive global checkout:
   - `git diff -- <mission files> > tracked.patch`
   - copy new/untracked mission files to a backup dir
   - `sha256sum` all artifacts; write `MANIFEST.json` (branch, HEAD,
     merge-base, per-file hashes, excluded pre-existing dirty files).
   - Example: `/home/deploy/mission-backup-cloe-rf-v1/`.
5. **Port cleanly onto current main:**
   ```
   git worktree add <path> -b <branch>-main-reconciled origin/main
   ```
   For each mission file:
   - CLEAN (main didn't touch) → copy from old worktree, verify sha256 matches
     backup.
   - CONFLICT (main touched it) → apply the mission delta SURGICALLY onto the
     main version (base = main, delta = mission). Preserve main's corrections.
     Prove coexistence by grepping for both main's marker and the mission
     marker in the same file (e.g. `createRetrievalMetrics` +
     `getKnowledgeStack`, `commit déployé` + `TARGETED_VERIFICATION`).
   - New files → copy from backup, verify sha256.
6. **Re-verify on the reconciled worktree:**
   - mission test suites (e.g. 9/9 acceptance)
   - full suite `node --test --test-concurrency=1 "test/*.test.js" "test/*.test.mjs"`
     vs a clean-main baseline worktree → zero net regression (reconciled fails
     ≤ main fails; list the delta and prove each extra/removed failure is
     pre-existing/flaky, not caused by the mission).
   - `git diff --check` clean, secret scan clean.
7. **Then** commit, push, PR, merge per normal gates. Keep the old-base branch
   as an untouched backup.

## Pitfalls

- `git worktree add` fails if the source checkout is dirty — worktree add from
  a clean location, or commit/stash first; never force-checkout over dirty work.
- The reconciled worktree has no `node_modules` — symlink to the canonical
  checkout's `node_modules` (`ln -s`) for tests.
- `node --test test/` (bare dir) fails with "Cannot find module .../test" —
  use the glob form `"test/*.test.js" "test/*.test.mjs"`.
- Full-suite runs take >5 min on this VPS — run in background with
  `notify_on_complete=true`.
- Pre-existing failures must be proven against a CLEAN main worktree
  (`git worktree add /tmp/main-baseline-check <sha>`), not assumed.
- CI (`ci-governance`) hangs on main pre-existing — merges use the documented
  dead-CI-check protocol (`--admin`), see commit message of 0458509/302ed75.
- Behavioral receipt must be generated AFTER the code commit and left
  uncommitted (self-referential commit_after/diff_hash) — see
  behavioral-operator-simulation skill.
