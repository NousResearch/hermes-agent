# Divergent Local Main vs True origin/main — Pre-commit Check

## When it bites

The canonical checkout's LOCAL `main` can be a stale/divergent history while the
REAL remote `main` has moved. Committing on local main then produces a bogus PR
diff (every commit between the divergence point and remote main shows up as a
change).

Observed 2026-08-03 (CLOE_DASHBOARD_REPLY_SESSION_CONFLICT_REPAIR_V1 PHASE 2):
- Local canonical main: `f76c845` (contained the phase-1 fix as a local commit,
  plus an OLD history).
- True `origin/main`: `101526b8` (the actual merged PR #667).
- Committing phase-2 on local main → PR diff would have been 109 files /
  15902 insertions. Detected at PR creation, rebuilt branch on `origin/main`.

## Why it happens

The canonical checkout and the workspace clone are separate clones with
different remotes (`github.com-lah-stack` vs `github.com`). Neither is
automatically synced to remote `main`. After a PR merges remotely, local
branches keep whatever history they had; if a prior mission committed locally
without pushing (or pushed then the remote got a squash-merge with a different
SHA), local main and origin/main diverge.

## Mandatory pre-commit check (before creating the mission branch)

```bash
cd <canonical_checkout>
git fetch origin main
git merge-base --is-ancestor HEAD origin/main \
  && echo "LOCAL IS ANCESTOR — OK" \
  || echo "DIVERGED — REBASE ONTO origin/main FIRST"
```

Content-parity check when divergence is suspected (files identical, history
differs):

```bash
git diff HEAD origin/main -- <mission-files> | wc -l   # 0 = identical content
```

## Fix when diverged

1. Create the mission branch FROM `origin/main`, never from local main:
   ```bash
   git checkout -b <branch> origin/main
   ```
2. If the branch name already exists on local main, `git reset --hard origin/main`
   (on the branch you're about to rebuild) instead of trying to delete a branch
   that the current worktree has checked out.
3. Bring the new phase's files from the old commit: `git checkout <old-sha> -- <files>`.
4. Verify the phase-N-1 files are content-identical between local and remote
   BEFORE assuming parity: `git diff <old-local-main> origin/main -- <phase-1-files> | wc -l`.

## Same trap on the workspace clone

The workspace clone (`/home/deploy/openclaw-runtime`) often sits on old feature
branches (`feat/cloe-...`) well behind `origin/main`. Never build a mission
branch on the workspace's current branch either — use `origin/main` (or the
canonical checkout's fetched `origin/main`) as the base.

## Post-merge verification

After merging, verify merged files are byte-identical to the PR head:

```bash
for f in <files>; do
  a=$(git rev-parse <pr-head-sha>:$f); b=$(git rev-parse origin/main:$f)
  [ "$a" = "$b" ] && echo "IDENTICAL: $f" || echo "DIFFERS: $f"
done
```

Squash-merge creates a NEW SHA on main whose tree must equal the PR head's tree;
blob-level comparison is the authoritative check.

---

## FastSafe batch-check exit-code pitfall (Gate 4)

When batching the 15 FastSafe checks with grep in a shell loop, do NOT write:

```bash
grep -rn 'provider' file.mjs | head -3; echo EXIT:$?
```

`$?` captures `head`'s exit status (always 0), so "No provider calls" /
"No LLM generation calls" checks falsely report FAIL even when grep found
nothing (observed 2026-08-03). Correct patterns:

- `grep -c 'pattern' file` → compare the count to 0 in Python;
- `grep -q 'pattern' file && echo FOUND || echo CLEAN` (no pipe);
- if a pipe is required, read `PIPESTATUS[0]`.

Always re-run the raw grep to verify a FAIL before declaring FASTSAFE_FAILED.
