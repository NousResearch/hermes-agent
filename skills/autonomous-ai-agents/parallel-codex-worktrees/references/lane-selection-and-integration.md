# Lane selection and integration checklist

Use this as the fast decision aid before launching Codex worktree lanes.

## Default choice

Start with **one isolated Codex worktree lane**.

Escalate to **2-4 parallel worktree lanes** only when all of these are true:

- the task is genuinely complex enough that orchestration overhead is worth it
- the slices are low-conflict and have clear ownership boundaries
- each lane can run a narrow verification step of its own
- integration back to the current branch is expected to be simpler than one larger lane

If you cannot explain the split in one sentence per lane, the task probably is not ready for parallelization.

## Worktree rule

For implementation work, keep Codex out of the shared checkout.

- single lane: create one isolated worktree
- parallel lanes: create one isolated worktree per lane
- do not let multiple Codex writers touch the same checkout

Suggested path pattern:

- `~/.worktrees/<repo>/<task-slug>/<lane-name>`

## Minimum lane contract

Every lane should leave behind verifiable git state, not just stdout:

- branch name is known
- `HEAD` changed from base
- `git status --short` is clean
- there is at least one real commit to inspect
- narrow tests were run or are runnable by Hermes immediately after

## Integration rule

Hermes accepts or rejects lanes from git state, not agent self-report.

Before telling the user a lane succeeded:

1. inspect the changed files
2. confirm the commit exists
3. confirm the worktree is clean
4. run the narrow verification yourself
5. integrate with cherry-pick or controlled reconciliation
6. run final integrated verification on the current branch
7. preserve unrelated pre-existing user changes on the destination branch — integrate only the accepted lane commit(s), not whatever else happens to be dirty on `main`
8. after integration, clean up the temporary worktree/branch unless there is a reason to keep it around

A good default for single-lane Codex work is: `worktree -> Codex commit -> Hermes re-run tests -> cherry-pick to current branch -> Hermes re-run tests again -> remove worktree`.

## Reporting shape

Keep the user-facing report brief and concrete:

- why this was single-lane vs parallel
- worktree path(s)
- branch name(s)
- commit(s) accepted
- tests rerun by Hermes
- leftovers intentionally excluded
