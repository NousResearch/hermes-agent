---
name: merge-coverage-verification
description: "Use when recovering, cherry-picking, manually porting, merging, or rebasing changes and you must prove source patches and policy tests survive into a local or remote target. Runs read-only Git coverage checks, focused policy tests, and evidence recording before declaring a merge complete."
license: MIT
metadata:
  hermes:
    tags: [git, merge, recovery, cherry-pick, coverage, verification]
    related_skills: [requesting-code-review, test-driven-development, plan]
---

# Merge Coverage Verification

## Overview

Prove that named recovery changes reached a target, rather than relying on a
green merge, a common ancestor, or a commit message. Use the read-only
`scripts/ci/history_coverage_check.py` before and after a merge. It never
applies patches, edits refs, pops stashes, prunes worktrees, or runs arbitrary
manifest commands.

Use this skill for recovery branches, cherry-picks, rebase ports, and any PR
whose correctness depends on a known source change reaching `main`.

Do not use it for an ordinary one-commit feature with no named source to track.

## Required Evidence

Record these values before changing the target:

```text
PRE_TARGET   target commit before integration
POST_TARGET  target commit after integration
SOURCES      source commit OIDs and/or saved patch paths
POLICIES     fixed policy-test IDs, if behavior contracts are affected
```

Use full resolved commit OIDs. Do not claim coverage from a branch name alone.
For local-only source commits, retain a stable patch file or patch-id evidence;
remote CI cannot resolve an object that was never pushed.

## Workflow

1. Establish the target relation.

   ```powershell
   git merge-base --is-ancestor $PRE_TARGET $POST_TARGET
   python scripts/ci/history_coverage_check.py `
     --pre-target $PRE_TARGET `
     --post-target $POST_TARGET `
     --candidate-commit $SOURCE `
     --format json --strict
   ```

   Completion: the report has a passed ancestry check and every source is
   either `introduced_exact` or explicitly `preexisting_ancestry`.

2. Treat non-exact ports as review gates.

   `missing`, `unverifiable`, `manual_review_required`, and
   `reverted_in_target` are fail-closed. For commit sources, the checker also
   compares the source parent to POST on the source paths, so a fully reverted
   source cannot pass merely because its patch-id appeared earlier in range.
   If an exact target commit's source path changes again before POST, the
   checker also stops at `manual_review_required`; it cannot infer that a
   transformed implementation preserves the original semantics.
   Do not relabel a manual port as exact coverage. Map the source-to-target
   hunks, run focused tests, and obtain independent review before recording it
   as a manual-port exception in the ULW ledger.

   Declare a deliberately changed source port, rather than disguising it as a
   missing cherry-pick:

   ```powershell
   python scripts/ci/history_coverage_check.py `
     --pre-target $PRE_TARGET --post-target $POST_TARGET `
     --manual-port-commit $SOURCE --format json --strict
   ```

   The command intentionally exits nonzero until a reviewer records the
   separate manual-port evidence; it never turns the port into an automatic
   pass.

3. Run fixed policy tests when a covered change affects a contract.

   ```powershell
   python scripts/ci/history_coverage_check.py `
     --pre-target $PRE_TARGET --post-target $POST_TARGET `
     --candidate-commit $SOURCE `
     --policy-test wave1-verification `
     --run-policy-tests --format json --strict
   ```

   Allowed IDs are deliberately fixed:

   | ID | Focused contract |
   | --- | --- |
   | `wave1-verification` | verification bundle and ULW ledger scaffold |
   | `worktree-quarantine` | recovery-first worktree preservation |
   | `runtime-contract` | prompt-cache/role sequence and Codex runtime |

   Completion: every selected policy result is `passed`. Do not put shell
   commands in a manifest or substitute an arbitrary test command.

4. Scan the merge delta, not the whole repository.

   The checker runs `git diff --check PRE_TARGET POST_TARGET` and searches only
   *added* `<<<<<<< ` / `>>>>>>> ` markers. A standalone `=======` may be a
   legitimate separator and must not create a false positive.

   Completion: `diff_check` and `conflict_markers` are both passed.

5. Preserve evidence and make the decision.

   Save the JSON report, source OIDs/patch IDs, exact test commands and exit
   codes, and independent reviewer verdict in the existing ULW ledger. A
   merge is not complete while any source is missing, unverifiable, or a manual
   port without documented review evidence.

6. Enforce narrow path scope when the recovery contract names every changed
   path.

   ```powershell
   python scripts/ci/history_coverage_check.py `
     --pre-target $PRE_TARGET --post-target $POST_TARGET `
     --candidate-commit $SOURCE --strict-paths `
     --allow-target-path docs/recovery-ledger.md --format json --strict
   ```

   Without `--strict-paths`, the report still records target and unexpected
   paths for review. With it, every target path must come from a named source
   or an exact allowlist entry; do not use globs or broad directory allowances.

## Manifest and CI

For a versioned recovery PR, add an explicit `.github/merge-coverage.json`:

```json
{
  "version": 1,
  "candidate_commits": ["full-source-commit-oid"],
  "manual_port_commits": [],
  "candidate_patches": [],
  "policy_tests": ["worktree-quarantine"],
  "allowed_target_paths": []
}
```

`candidate_patches` are read relative to the manifest and must remain in that
directory; a contract may not traverse to another path. The optional
`merge-coverage.yml` workflow runs only when this file exists: PRs use
`base.sha → GitHub's synthetic merge sha`, while `main` pushes use
`before → sha`. This verifies the result that would land even when a PR branch
is behind its base; it is not a blanket gate for unrelated PRs.

## Common Pitfalls

1. **History-only success.** A common ancestor proves repository lineage, not
   that a source patch reached the target. Always pass source OIDs or patches.
2. **Preexisting success.** `preexisting_ancestry` is safe but does not prove
   this merge introduced the change. Keep it separate from `introduced_exact`.
3. **Patch-id overclaim.** A changed manual port may be correct, but it is not
   exact. Leave it review-required until tests and reviewer evidence exist.
   The final-path check catches a complete revert. A later edit to any source
   path is review-required instead of an automatic exact claim.
4. **Global marker scanning.** Do not fail on every `=======`; inspect only
   added opening and closing conflict markers in the merge delta.
5. **Unsafe manifest execution.** Policy IDs map to fixed argv. Never execute
   arbitrary commands supplied in a recovery manifest.
6. **Remote-object assumption.** A local-only commit cannot be verified by CI
   unless its patch evidence is included in the recovery contract.

## Verification Checklist

- [ ] `PRE_TARGET` is an ancestor of `POST_TARGET`.
- [ ] Every source reports `introduced_exact` or is separately recorded as
  preexisting/manual-review evidence.
- [ ] When exact recovery scope is known, `--strict-paths` records no
  unexplained target path.
- [ ] `git diff --check` and added-marker checks pass.
- [ ] Every relevant fixed policy-test ID passes.
- [ ] JSON report, commands, exit codes, and reviewer verdict are in the ULW
  ledger.
- [ ] For a recovery PR, the manifest is versioned and the optional CI workflow
  verifies both pre-merge and post-main target ranges.
