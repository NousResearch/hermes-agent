# hermes_parity operator guide

`hermes_parity` is an internal stdlib-only CLI for upstream/fork parity merges.
Run it from a Hermes checkout with remotes named `fork` and `origin`.

## Fork-delta acknowledgements

When stage 4 hard-fails on a fork-only file that upstream intentionally
deleted/renamed (reviewed, not an accident), clear it first-class instead of
reaching for `finish --force`:

```
python -m hermes_parity ack --reason "upstream removed X; fork copy superseded" path/to/file.py
```

Acks persist in `.parity-state.json` (`forkdelta_acks`: path + reason +
timestamp) and show in the gate detail as "acknowledged".

## Lifecycle

1. `python -m hermes_parity status --fail-behind 25`
2. `python -m hermes_parity start`
3. Resolve conflicts in the created worktree, preserving fork features listed in
   `docs/sync/fork-features.json`.
4. `python -m hermes_parity gates --worktree <worktree> --resume`
5. Classify reds with `python -m hermes_parity bisect --from-file failures.txt`
6. `python -m hermes_parity finish --worktree <worktree>`
7. A human lands the PR with a merge commit.

## status

Fetches `origin` and `fork`, then prints branch, fork/upstream ahead-behind
counts, merge-base, unmerged entry counts, marker counts, and conflict buckets.
`--fail-behind N` exits 2 when `fork/main` is behind `origin/main` by at least
`N` commits.

## start

Creates a new worktree under `~/.hermes/worktrees` by default from `fork/main`,
freezes the upstream target SHA, runs `git merge --no-commit --no-ff`, writes the
spec-shaped `.parity-state.json`, and emits
`docs/sync/review/conflict-buckets.md`. Rollback SHAs live only in the state
file.

## gates

Runs the stage ladder and appends JSONL records to `<worktree>/gates.jsonl`:
markers, imports, traps, manifest+forkdelta, tests, linuxonly. Each stage also
persists `{ok, at, tree_sha}` into `.parity-state.json`.

Use `--stage NAME` to run one stage, `--fast` for stages 1-4, `--resume` to skip
stages already green at the current tree SHA, and `--strict` to make trap
findings a hard failure. Every stage table includes its repro command. CI-owned
reminders list gitleaks, contributor-check, config-migration dry-run, and
desktop `tsc`; gitleaks shows the pinned workflow version when it can be parsed.

## bisect

Runs each pytest node against the merge worktree and an auto-managed cached
baseline worktree at `~/.hermes/worktrees/parity-baseline`, pinned to
`fork_main_at_start` from `.parity-state.json`. `--baseline` and `--merge`
override the defaults.

Accepts node ids as arguments or from `--from-file PATH`; `--from-file -` reads
stdin. `--jobs N` is capped at half the local CPU count. Classifications are:
`MERGE REGRESSION`, `INHERITED/FLAKY`, `UPSTREAM TEST`, `ORDER-POLLUTION`, and
`FLAKY` for regressions that pass on the required re-run.

## finish

Refuses if `fork/main` moved since `start`, unless the operator first merges
`fork/main` into the sync branch. Re-verifies gates, requires an in-progress
merge or staged merge changes, creates the merge commit, pushes
`git push <remote> <branch>`, writes a PR body file into
`docs/sync/review/parity-pr-body.md`, and prints a `gh pr create --body-file`
command. `--force` requires `--force-reason`, still runs gates, and still prints
reds.

## clean

Removes a parity worktree under the configured worktree root. The command prints
the rollback shape before removal.

## Landing Rules

Use `git merge --merge`; never squash parity merges. If the branch is behind,
merge `fork/main`; never rebase. Preserve contributor authorship and merge
history.

## Manifest Rule

Every postmortem that identifies a fork-only production behavior gets a
`docs/sync/fork-features.json` entry with feature, collecting pytest node IDs,
repo-relative path globs, and why the behavior matters.
