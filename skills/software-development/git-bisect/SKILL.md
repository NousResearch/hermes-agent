---
name: git-bisect
description: Find the first bad commit with a safe git bisect workflow for regressions and reproducible bugs.
version: 1.0.0
author: sonoyuncu
license: MIT
metadata:
  hermes:
    tags: [git, debugging, regression, bisect, troubleshooting]
    related_skills: [systematic-debugging, test-driven-development, github-pr-workflow]
---

# Git Bisect

Use `git bisect` when a bug exists now, did not exist in an earlier revision, and you can classify each tested commit as good or bad.

## When to Use

- Reproducible regressions with a known good range
- Failing tests that started after some recent change
- Build or runtime breakages caused by one specific commit
- Flaky behavior only if you can turn it into a deterministic pass/fail check

Do not use this when the repository is dirty, history is too shallow, or the result cannot be judged consistently.

## Quick Reference

```bash
# Start from a clean worktree
git status --short

# Mark the current broken state and an older known-good commit
git bisect start
git bisect bad
git bisect good <good-commit-or-tag>

# At each step, run the check and classify the commit
<test-command>
git bisect good
git bisect bad

# Or automate the loop with an exit-code-based script
git bisect run <script-or-command>

# Always restore your original branch when done
git bisect reset
```

## Procedure

1. Confirm the bug is real and reproducible now. Prefer one command that exits `0` for good and non-zero for bad.
2. Verify the worktree is clean with `git status --short`. If there are local edits, stop and preserve them before bisecting.
3. Make sure history is deep enough to contain a known good point. Full clones are ideal; shallow history weakens bisect.
4. Identify one known bad revision, usually `HEAD`, and one known good commit, tag, or merge base from before the regression.
5. Start the search:

```bash
git bisect start
git bisect bad
git bisect good <good-commit-or-tag>
```

6. For each checkout produced by Git, run the same validation command. Mark `good` only when the bug is absent, and `bad` only when it is present.
7. If the test can be scripted, prefer automation:

```bash
git bisect run scripts/run_tests.sh
```

Use a wrapper script if the project needs setup, environment variables, or a narrower test target.

8. When Git reports the first bad commit, inspect it with `git show --stat --patch <commit>` and confirm the result makes sense.
9. Exit bisect mode immediately after recording the result:

```bash
git bisect reset
```

## Pitfalls

- Do not bisect on a dirty worktree. Local edits contaminate results.
- Do not change the test command mid-run. Bisect only works with a stable pass/fail rule.
- Do not guess on ambiguous commits. If setup is broken for unrelated reasons, skip the commit with `git bisect skip`.
- Do not forget `git bisect reset`; otherwise you remain detached on an intermediate revision.
- Do not treat flaky tests as trustworthy. Stabilize the reproduction first or the result will be noise.

## Verification

- `git status --short` is clean before starting and after resetting.
- The chosen good commit is genuinely known-good, not assumed.
- The same command was used to judge every commit in the search.
- `git bisect reset` returned the repo to the original branch.
- The reported first bad commit was inspected and matches the symptom timeline.
