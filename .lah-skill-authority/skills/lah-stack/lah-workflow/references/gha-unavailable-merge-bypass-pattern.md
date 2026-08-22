# GHA-Unavailable Merge Bypass Pattern

Procedure for merging a PR when GitHub Actions is unavailable and a required CI check
blocks the merge. Established during `CLOE_FEATURE_BRANCH_TO_MAIN_CONTROLLED_MERGE_V1`
(PR #632, openclaw-runtime, 2026-07-30).

## Context

- `openclaw-runtime` requires `ci-governance` check via branch protection (`strict: true`)
- GitHub Actions (`openclaw-ci` workflow) had been **failing on `main` for 7+ days** —
  a pre-existing infrastructure failure, not caused by the feature branch
- `enforce_admins: false` — admins CAN bypass (branch protection allows `--admin` flag)
- Repository uses SSH remote alias (`github.com-lah-stack`) rather than standard `github.com`
  for git operations; `gh` CLI uses HTTPS with a token

## Prerequisites

Before attempting this pattern, the following must be true:

1. **Feature branch certified clean**: 218/218 bounded tests pass, 0 merge conflicts,
   no dependency changes, no unwanted files in diff
2. **CI check verified pre-existing**: `gh run list --branch main --workflow <name> --limit 3`
   confirms all recent runs on `main` also fail
3. **Operator authorization obtained**: use `clarify` with three options (admin bypass,
   auto-merge, block)
4. **Head SHA verified**: matches expected before merge attempt

## Protocol

### Step 1 — Verify pre-existing failure

```bash
gh run list --repo leanframeworklab/openclaw-runtime --branch main \
  --workflow openclaw-ci --limit 3 --json conclusion,headSha,createdAt
```

Expected: all three show `conclusion: "failure"`, with the `headSha` matching
commits on `main` (not on the feature branch).

### Step 2 — Document in PR body

Include in the PR body or a PR comment:

- Which check is dead (`ci-governance`)
- Since when it's been failing (check run timestamps)
- Evidence it fails on main (references to failed runs on main)
- Local certified test count (e.g. 218/218 passing)
- Statement: "GitHub Actions unavailable — local certified tests used as validation"

### Step 3 — Get operator authorization

```javascript
clarify({
  question: "The merge is blocked by a required CI check that fails pre-existing on main. Options:",
  choices: [
    "gh pr merge --admin (bypass dead check)",
    "gh pr merge --auto (wait — won't merge until CI fixed on main)",
    "Do not merge — report BLOCKED"
  ]
})
```

### Step 4 — Merge with admin bypass

```bash
gh pr merge <PR> --repo <owner>/<repo> --admin --merge \
  --subject "<merge-commit-title>" \
  --body "<merge-commit-body>"
```

The `--admin` flag bypasses required CI checks when `enforce_admins: false`.

### Step 5 — Post-merge verification

```bash
git fetch --prune origin
git rev-parse origin/main
git merge-base --is-ancestor <feature-head> origin/main  # must return YES
gh pr view <PR> --repo <owner>/<repo> --json state,mergedAt,mergeCommit
```

### Step 6 — Verify container unchanged

```bash
docker inspect <container> --format '{{.State.StartedAt}} {{.State.Status}} {{.RestartCount}}'
```

Confirm started-at is before merge time, restart count = 0.

## SSH remote workaround

When git remote uses a custom SSH hostname:

```text
origin  git@github.com-lah-stack:leanframeworklab/openclaw-runtime.git (fetch)
```

But `gh` authenticates via HTTPS to `github.com`:

```text
gh auth status → Logged in to github.com account leanframeworklab
  - Git operations protocol: https
```

The branch must be pushed to the SSH remote FIRST, THEN `gh pr create` works:

```bash
git push origin feat/my-branch
gh pr create --repo leanframeworklab/openclaw-runtime --base main --head feat/my-branch
```

Verify the branch exists on the remote before creating the PR:

```bash
git ls-remote --heads origin feat/my-branch
```

## Safety counters

When using this pattern, track:

```text
pull_requests_created  = 1
merges                 = 1
pushes                 = 1
npm installs in temp   = 1 (dotenv for fresh worktree)
deployments            = 0
runtime restarts       = 0
```

## Known pitfalls

| Trap | Symptom | Fix |
|------|---------|-----|
| **gh pr merge --admin without operator consent** | Bypassing security without authorization | Always use `clarify` first. Document the decision in the safety statement. |
| **Fresh worktree missing node_modules** | HTTP route proof tests fail with `ERR_MODULE_NOT_FOUND` (dotenv) | This is environmental, not a regression. Run unit tests (no dotenv) to confirm. Install dotenv for full suite. |
| **git worktree remove fails after merge simulation** | "contains modified or untracked files" | Use `git worktree remove --force <path>` |
| **mergeStateStatus=BLOCKED but mergeable=MERGEABLE** | This is the pre-existing CI failure pattern | Follow the four-step protocol above. |
