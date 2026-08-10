---
name: ci-fix
description: 'Diagnose and fix GitHub Actions CI failures. Inspects workflow runs and logs, identifies root causes, implements minimal fixes, and pushes to a fix branch.'
version: 1.0.0
author: Hermes Agent (adapted from Warp/ci-fix)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [CI, GitHub-Actions, debugging, DevOps, workflow]
    related_skills: [systematic-debugging, github-pr-workflow, verification-loop]
---

# CI Fix

Diagnose CI failures and implement fixes with minimal, targeted diffs. Pushes fixes to a dedicated branch without creating PRs.

## Prerequisites

Verify GitHub CLI authentication before proceeding:

```bash
gh auth status
```

If not authenticated, instruct the user to run `gh auth login` first.

## Workflow

### 1. Locate the Failing Run

Determine the failing workflow run. If working on a PR branch:

```bash
gh pr view --json statusCheckRollup --jq '.statusCheckRollup[] | select(.conclusion == "FAILURE")'
```

If working from a branch or run ID:

```bash
gh run list --branch <branch> --status failure --limit 5
gh run view <run-id> --verbose
```

### 2. Extract Failure Logs

Pull logs from failed steps to identify the root cause:

```bash
gh run view <run-id> --log-failed
```

For deeper inspection:

```bash
gh run view <run-id> --log --job <job-id>
gh run download <run-id> -D .artifacts/<run-id>
```

### 3. Identify Root Cause

Analyze logs for common failure patterns:

- **Build/compilation errors**: Missing dependencies, type errors, syntax issues
- **Test failures**: Assertion failures, timeouts, flaky tests
- **Linting/formatting**: Style violations, unused imports
- **Environment issues**: Missing secrets, permissions, resource limits

Prefer the smallest fix that resolves the issue. Deterministic code fixes are better than workflow plumbing changes.

### 4. Implement the Fix

Make minimal, scoped changes matching the repository's existing style:

- Fix only what's broken — avoid unrelated refactoring
- Keep changes to the failing job/step when possible
- If modifying workflow files, preserve existing permissions and avoid expanding token access

### 5. Push to Fix Branch

Create or update a dedicated fix branch:

```bash
git checkout -b ci-fix/<original-branch>
git add -A
git commit -m "fix: resolve CI failure in <job-name>"
git push -u origin ci-fix/<original-branch>
```

If the fix branch already exists, update it:

```bash
git checkout ci-fix/<original-branch>
git pull origin <original-branch>
# make fixes
git commit -m "fix: <description>"
git push
```

### 6. Verify the Fix

Trigger CI on the fix branch and monitor:

```bash
gh run list --branch ci-fix/<original-branch> --limit 1
gh run watch <new-run-id> --exit-status
```

To rerun only failed jobs:

```bash
gh run rerun <run-id> --failed
```

## Safety Notes

- Avoid `pull_request_target` unless explicitly requested — it can expose secrets to untrusted code
- Keep workflow `permissions:` minimal; don't broaden access to make tests pass
- For flaky tests, prefer deterministic fixes over blind reruns

## Deliverable

After fixing, provide a brief summary:

- **Failing run**: Link or ID
- **Root cause**: What broke and why
- **Fix**: What changed
- **Verification**: New run link showing green status

## Hermes Integration

- Use `terminal` tool for all `gh` CLI commands
- Use `read_file` to inspect workflow files and source code
- Use `write_file` / `replace_string_in_file` for minimal fixes
- Use `search_files` to find patterns across workflow files
- Combine with `systematic-debugging` for root cause analysis
- Combine with `verification-loop` for pre-push quality checks
- For Hermes Agent CI, check `.github/workflows/` for workflow definitions
