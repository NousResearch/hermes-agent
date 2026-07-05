---
name: github-flow-automation
description: "Autonomous GitHub branch, commit, PR, CI, and review-gate workflow for Hermes repo work."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [github, pull-request, ci, review-gates, automation]
    related_skills: [github-pr-workflow, requesting-code-review]
---

# GitHub Flow Automation

Use this when the user asks Hermes to autonomously ship repo-backed work.

## Safe default flow

1. Inspect repo, branch, remote, and dirty files.
2. Create or reuse a focused feature branch.
3. Implement one logical change set.
4. Run targeted tests.
5. Run review gates.
6. Commit only intentional files.
7. Push the branch.
8. Open a PR with summary, test plan, risk, and rollback.
9. Watch CI and fix failures up to three cycles.
10. Do not merge unless the user asked for merge or auto-merge and checks pass.

## Review gate checklist

Before push/PR:

```bash
git diff --stat HEAD
git diff --check
python -m pytest <targeted-tests> -q -o 'addopts='
```

Also verify:

- No unrelated files staged.
- No literal secrets or tokens in the diff.
- The PR body contains real test output.
- Risk and rollback are documented.

## PR body template

```markdown
## Summary
- ...

## Test plan
- `command` → result

## Risk / rollback
- Risk: ...
- Rollback: revert this PR

## Scope notes
- Unrelated dirty files intentionally excluded: ...
```

## CI fix loop

If CI fails:

1. Fetch failed job logs.
2. Find the exact failing command.
3. Reproduce locally if possible.
4. Fix only the failure.
5. Commit and push.
6. Re-check CI.

Stop after three failed fix cycles and report the remaining root-cause hypothesis.
