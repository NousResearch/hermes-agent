---
name: review
description: >
  AI-powered code review via /review. Analyzes staged changes (default),
  unstaged changes, specific files, or the current PR branch. Produces
  structured feedback with severity levels using existing tools.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [code-review, quality, git, review, feedback]
    related_skills: [requesting-code-review, github-code-review, test-driven-development]
---

# Code Review

On-demand code review. Analyzes diffs or files and returns structured feedback
organized by severity: critical issues, warnings, suggestions, and positives.

## Invocation

```
/review              — review staged changes (git diff --cached)
/review unstaged     — review working-tree changes (git diff)
/review <file>       — review a specific file in full
/review pr           — review all commits on the current branch vs main
```

## Step 1 — Determine the review target

Parse the user instruction to select one of four modes:

| Instruction        | Mode     | Diff command                          |
|--------------------|----------|---------------------------------------|
| (empty)            | staged   | `git diff --cached`                   |
| `unstaged`         | unstaged | `git diff`                            |
| a file path        | file     | `read_file` on the path               |
| `pr`               | pr       | `git diff main...HEAD`                |

If the mode is `staged` and `git diff --cached` is empty, check `git diff`.
If that is also empty, tell the user there are no changes to review and stop.

For `pr` mode, also run:
```bash
git log main..HEAD --oneline
git diff main...HEAD --stat
```

## Step 2 — Gather context

### For diff-based modes (staged, unstaged, pr)

1. Get the stat summary first to understand scope:
```bash
git diff --cached --stat   # or git diff --stat / git diff main...HEAD --stat
```

2. If the diff exceeds 500 lines, review file-by-file:
```bash
git diff --cached --name-only
# then for each file:
git diff --cached -- path/to/file.py
```

3. For each changed file, use `read_file` to see surrounding context when the
   diff alone is ambiguous. A diff shows what changed; the full file shows
   whether the change is correct.

### For file mode

Use `read_file` on the target path. If the file is in a git repo, also run:
```bash
git log --oneline -5 -- path/to/file
```

## Step 3 — Scan for common problems

Run these checks on the diff output (skip for file mode):

```bash
# Debug artifacts left behind
git diff --cached | grep -n "print(\|console\.log\|debugger\|binding\.pry"

# Credentials or secrets
git diff --cached | grep -in "password\|secret\|api_key\|token.*=.*['\"]"

# Merge conflict markers
git diff --cached | grep -n "<<<<<<\|>>>>>>\|======="

# TODO/FIXME/HACK markers (informational, not blocking)
git diff --cached | grep -n "TODO\|FIXME\|HACK\|XXX"
```

Substitute the appropriate diff command for the active mode.

## Step 4 — Review the code

Evaluate the changes against these categories. Focus on what matters; skip
categories that do not apply.

### Correctness
- Does the code do what it claims?
- Edge cases: empty inputs, nulls, zero-length collections, boundary values
- Off-by-one errors in loops and slices
- Error paths handled or propagated correctly

### Security
- No hardcoded secrets, API keys, or credentials in the diff
- User-facing input validated before use
- No SQL injection, XSS, command injection, or path traversal
- Auth and authz checks present where required

### Logic and data flow
- Conditionals test the right thing
- No unreachable code or dead branches
- State mutations happen in the right order
- Async code awaits correctly; no fire-and-forget promises hiding errors

### Code quality
- Naming is clear and consistent with the surrounding codebase
- No unnecessary complexity or premature abstraction
- Duplicated logic that should be extracted
- Functions focused on a single responsibility

### Testing
- New behavior has corresponding tests
- Tests cover both happy path and error cases
- No tests that always pass (tautologies)

### Performance (flag only when impact is real)
- N+1 queries or unnecessary loops over large collections
- Blocking calls in async code paths
- Missing indexes for new query patterns

## Step 5 — Present findings

Use this structure for the review output:

```
## Review Summary

**Target:** [staged changes | unstaged changes | path/to/file | PR branch]
**Files:** [N] ([+additions] [-deletions])

### Critical
- **file.py:line** — [what is wrong]. Fix: [concrete suggestion].

### Warnings
- **file.py:line** — [what is wrong].

### Suggestions
- **file.py:line** — [observation or improvement].

### Looks good
- [specific positive observation about the code]
```

Rules for the output:
- Every finding references a file and line number.
- Critical and Warning items include a concrete fix or direction, not just
  "this is bad."
- The Looks Good section names something specific. If nothing stands out,
  omit it rather than writing filler.
- Omit empty severity sections entirely.
- If there are zero findings at any severity, say so directly: "No issues
  found. The changes look clean."

### Severity guide

| Level      | Blocks merge? | Examples                                        |
|------------|---------------|-------------------------------------------------|
| Critical   | Yes           | Security holes, data loss, crashes               |
| Warning    | Usually       | Bugs in secondary paths, missing error handling  |
| Suggestion | No            | Style, naming, minor refactors, missing docs     |
| Looks good | N/A           | Clean patterns, good test coverage, clear naming |

## Pitfalls

- **Empty diff:** Check `git status` and report. Do not fabricate findings.
- **Binary files in diff:** Skip them, note they were skipped.
- **Very large diffs (>1000 lines):** Split by file. Summarize scope before
  diving in. Prioritize files with the most additions.
- **File mode on a non-existent path:** Report the error, do not guess.
- **Not a git repo (for diff modes):** Tell the user and suggest file mode.

## Relationship to other skills

- **requesting-code-review:** A heavier pre-commit pipeline with static
  scans, subagent verification, and auto-fix loops. Use that when you want
  automated gate-keeping before a commit.
- **github-code-review:** Reviews other people's PRs on GitHub with inline
  comments via the GitHub API. Use that to post review comments on a remote PR.
- **This skill (/review):** Quick, conversational review of your own changes.
  No subagents, no API calls, no auto-fix. Just feedback.
