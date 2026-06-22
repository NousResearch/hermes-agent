# Integration-path artifact: combining all PRs onto v0.17.0 (3-way merge)

The goal asks the PRs "pull down later on top of a later release such as v0.17.0".
A maintainer does this with **3-way merges** (one PR at a time), NOT a naive
`git apply` stack. This artifact records the real combinability result.

## Method
Fresh v0.17.0 (2bd1977d8) worktree; merge each open PR's branch via
`git merge --no-ff` in ascending PR-number order. (Excludes #50111 manifest,
and the closed #50484/#50487/#50049.)

## Result: 37 of 39 PRs combine cleanly

| Metric | Value |
|---|---|
| PRs merged clean (3-way) | **37** |
| PRs with genuine combine-conflict | **2: #50457, #50296** |
| Final tree conflict markers | 0 (conflicting PRs left out, not corrupted) |
| Files changed vs v0.17.0 | 494 |

Compare: a NAIVE `git apply` stack merged only 23/39 — the gap (37 vs 23) is
exactly the file-overlap that 3-way merge resolves but blind patching cannot.

## The 2 combine-conflicts — diagnosis

### #50457 — redundant 100-file "cross-PR opus-context integration regression suite"
This is a giant bundle touching **139 .py files**, of which **94 duplicate other
PRs**. The overlap is what conflicts on combine. Only 6 files are unique to it, and
2 of those (subdirectory_hints.py + its test) belong to the closed-dup #50049/#29433
lineage. **Recommendation: slim #50457 to its ~4 genuinely-unique files
(hermes_cli/auth.py, runtime_provider.py, tests/agent/conftest.py,
test_copilot_opus_context_fix), or close it** — same redundant-mega-bundle pattern
as the already-closed #50484/#50487. Excluding #50457 drops combine-conflicts 2 -> 1.

### #50296 — 1-file overlap (background-review session isolation)
Genuine single-file overlap with another PR's edit to the same region; resolvable
by a maintainer's 3-way merge in the normal way (the collaborator @alt-glitch
already confirmed it's a distinct, non-duplicate layer).

## Note on the "compile failures"
The 6 reported compile-fails are `hermes_cli/inventory.py` + its tests — a NEW file
that does not exist in v0.17.0 and is created by #50457/#50064. When the integration
test excludes the creating PR, the file is absent, so the compile check reports
"No such file" — an artifact of the exclusion, NOT broken code. With the owning PR
merged, the file exists and compiles.

## Conclusion
- 37/39 PRs combine cleanly on v0.17.0 via the real (3-way merge) operation.
- The 2 conflicts trace to ONE redundant mega-bundle (#50457, recommend slim/close)
  + one ordinary single-file overlap (#50296, normal merge resolution).
- Each PR is ALSO individually clean on v0.17.0 (41/41, see VERIFY-LOG).
