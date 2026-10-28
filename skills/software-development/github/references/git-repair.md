# Git Repair: conflicts, bisect, history rewrite, stacked branches

Load before resolving a conflict, bisecting a regression, rewriting history, or restacking
branches after their base moved. Run every command through `terminal` and report what its output
showed; keep resolved, verified and pushed as separate states.

## 1. Inventory what is pushed, first

`git fetch` first; a stale remote-tracking ref makes every later step wrong. Then, per branch:

| Branch | Local head | Remote head | Match | Others may have fetched |
| --- | --- | --- | --- | --- |
| `feature/x` | `git rev-parse feature/x` | `git rev-parse origin/feature/x` | same / ahead / behind / diverged | yes if shared, has an open PR, or a CI run |

A commit on a remote someone else may have fetched is shared history: rewriting it is a decision
for everyone who has it, not a local cleanup.

## 2. Recovery point before every rewrite

- `git branch backup/<name>-<date> <branch>` before a rebase, reset, amend or filter, or note the
  reflog entry to return to: `git reflog show <branch> -n 5`.
- A backup branch or reflog entry holds commits only. Before any `git reset --hard`, check
  `git status --porcelain`; save uncommitted edits (including conflict resolutions in progress)
  with `git stash push --include-untracked -m <why>` or a WIP commit on the backup branch.
- Recovery is `git reset --hard backup/<name>-<date>` or `git reset --hard <branch>@{N}` once the
  tree is saved or clean, and it comes before trying anything else when a rewrite goes wrong.

## 3. Conflicts

- Read both sides before choosing: `git diff --merge`, or `git log --merge -p -- <file>` for what
  each side intended. Decide each hunk by what both changes were for, not by which is newer.
- **Never pick a side in a generated or counted file.** Take either side, finish the merge, then
  re-run the producer and commit its output. A count pinned in two places conflicts whenever both
  sides added a case, and neither side is right.
- Lockfiles: resolve the manifest, then regenerate the lockfile with the package manager.
- A rebase replays commits one at a time, so the same hunk can conflict at every step.
  `git config rerere.enabled true` records resolutions so repeats resolve themselves.
- Verify after resolving: the build and the tests covering the conflicted files, observed green.

## 4. Bisect

1. Confirm one known-good and one known-bad commit, each by an observed run.
2. Write the exact command that tells them apart: exit 0 on good, 1–127 (not 125) on bad, 125 to
   skip an untestable commit. Exit ≥128 aborts the bisect.
3. `git bisect start <bad> <good>` then `git bisect run <command>`.
4. Record the first bad commit and the command's output at it and at its parent.
5. `git bisect reset` to return.

A flaky command bisects to noise. Measure its failure rate on the known-good and known-bad
commits first. If it ever fails on the good commit, make the bisect command run it a fixed number
of times (N, chosen from that rate) and decide by rule: all N pass → exit 0, all N fail → exit 1,
mixed → exit 125 (skip). Stop and report instead of bisecting when good and bad cannot be told
apart within N runs.

## 5. History rewrite

| Goal | Command | Rewrites |
| --- | --- | --- |
| Squash or reorder before review | `git rebase -i <base>` (or `--autosquash` with `fixup!` commits) | every commit after `<base>` |
| Fix the last commit | `git commit --amend` | the last commit |
| Drop local commits | `git reset --hard <ref>` | the branch pointer |
| Undo a pushed commit without rewriting | `git revert <sha>` | nothing |

Push a rewrite with `git push --force-with-lease=<branch>:<expected-sha> origin <branch>`, where
`<expected-sha>` is the remote head from the inventory. The lease refuses when the remote moved
since then, which is exactly the case a bare `--force` silently destroys. Never force-push a
shared default branch; revert there instead.

## 6. Stacked branches

When the base of a stack is rebased or squash-merged, every branch above it still carries the old
base commits.

- Record each branch's old base **before** the base moves: `git rev-parse <base-branch>`.
- Restack each branch onto the new head of the one below it:
  `git rebase --onto <new-base> <old-base> <branch>`.
- Never compute the old base with `git merge-base` after the base was rewritten; it finds a commit
  the rewrite no longer contains and replays the base's commits onto itself.
- Git 2.38+: `git rebase --update-refs --onto <new-base> <old-base> <top-branch>` restacks the
  whole stack and moves every intermediate branch ref in one pass. Keep `--onto <old-base>`:
  `--update-refs` only moves pointers, and without the recorded old base a squash-merged base's
  original commits are replayed again.
- Push each restacked branch with `--force-with-lease`, bottom-up.

## 7. Shared checkouts and worktrees

- Two sessions in one checkout share `HEAD`: a `checkout` by one moves the other's next commit
  onto the wrong branch. Use one `git worktree` per session.
- Removing a worktree mid-rebase loses the rebase state; finish or `git rebase --abort` first.
- A commit that landed on the wrong branch: move it with a cherry-pick onto the right branch, then
  `git branch -f` (or `git update-ref`) to repair the wrong one. Never `reset --hard` inside
  another session's checkout.

Ported from rlaope/oh-my-hermes (MIT), `skills/omh-git-workflow/references/git-repair-method.md`.
