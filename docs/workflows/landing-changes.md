# Landing Changes

Stay on the branch already checked out for routine work. Do not rename it or
replace other work in the checkout. Treat existing changes as intentional.
Follow the user's instructions for commits, pushes, and pull requests.

Before landing, review the complete diff and run the applicable checks. Resolve
valid findings at their cause, then repeat affected checks. Use the repository
PR template when opening a PR. Keep upstream merges and unrelated work separate
from feature changes.

## Commits, Merges, PRs

- **Squash merges from stale branches silently revert recent fixes.** Before squash-merging,
  reconcile the branch with current `main` while preserving local work and contributor
  commits. Do not reset a shared working tree or discard uncommitted changes.
  Verify with `git diff HEAD~1..HEAD` after merging — unexpected deletions are a red flag.
- Salvage by cherry-pick so contributor authorship survives (see [contribution rubric](../contributing.md)).
- Tests per fix: 1–2 INVARIANT tests (behaviour contract, proven red on base), never
  change-detectors; ≤ 2 tests is the salvage bar too. Reject/rewrite in salvaged diffs:
  appendages to facades, new god helpers, compat aliases, wrappers.
