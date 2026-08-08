---
name: review
description: Inspect code diffs, files, or PR branches.
version: 1.0.0
author: Rishabh Bhandari (@RishabhKodes) + Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [code-review, quality, git, review, feedback]
    related_skills: [requesting-code-review, github-code-review, test-driven-development]
---

# Code Review Skill

Use this skill for quick, conversational review of local code changes. It reviews
staged diffs, unstaged diffs, a single file, or the current branch against
`main`, then returns findings grouped by severity. It does not post GitHub
comments, run subagents, or auto-fix code.

## When to Use

- The user invokes `/review` with no arguments to review staged changes.
- The user invokes `/review unstaged` to review working-tree changes.
- The user invokes `/review <path>` to review one file.
- The user invokes `/review pr` to review the current branch against `main`.
- The user asks for a lightweight review of their own local work.

Use `requesting-code-review` instead when the user wants a heavier pre-commit
pipeline with static checks, an independent reviewer subagent, and auto-fix
loops. Use `github-code-review` instead when the user wants comments posted on a
remote GitHub PR.

## Prerequisites

- Diff-based modes require a git repository and `git` available through
  `terminal`.
- File mode requires the target file to be readable through `read_file`.
- For broad or multi-file reviews, load `references/review-checklist.md` with
  `skill_view` if you need the full review checklist.

## How to Run

```text
/review
/review unstaged
/review path/to/file.py
/review pr
```

Treat text after `/review` as the target selector:

| User input | Target | Primary context |
|------------|--------|-----------------|
| empty | staged changes | `git diff --cached` |
| `unstaged` | working-tree changes | `git diff` |
| file path | one file | `read_file` |
| `pr` | branch vs main | `git diff main...HEAD` |

If staged mode has no staged diff, check `git diff`. If both diffs are empty,
report that there are no changes to review and stop.

## Quick Reference

- Use `terminal` for git commands.
- Use `read_file` when a diff lacks enough surrounding context.
- Use `search_files` to look for debug artifacts, conflict markers, or related
  call sites when the diff suggests a broader issue.
- Use `skill_view` to load `references/review-checklist.md` during larger
  reviews.
- Keep findings concrete: every issue needs a file path and line number when
  available.

## Procedure

1. Determine the target.
   - Empty instruction: staged changes.
   - `unstaged`: working-tree changes.
   - `pr`: current branch compared with `main`.
   - Anything else: treat it as a file path.

2. Gather a scope summary.
   - For staged mode, run `git diff --cached --stat`.
   - For unstaged mode, run `git diff --stat`.
   - For PR mode, run `git log main..HEAD --oneline` and
     `git diff main...HEAD --stat`.
   - For file mode, read the file with `read_file`.

3. Gather review context.
   - For diff modes, read the relevant diff with `terminal`.
   - If a diff is large, list changed files first and review one file at a time.
   - Use `read_file` for changed files when the diff alone is ambiguous.
   - Use `search_files` for related definitions, call sites, conflict markers,
     or suspicious debug artifacts.

4. Review for material issues.
   - Correctness: broken behavior, missed edge cases, bad error handling.
   - Security: secrets, unsafe input handling, injection, traversal, auth gaps.
   - Logic and data flow: wrong conditions, bad state ordering, async mistakes.
   - Code quality: unnecessary complexity, unclear naming, misplaced
     abstractions, duplicated logic.
   - Testing: missing tests for new behavior or tests that do not assert the
     behavior they claim to cover.
   - Performance: only flag issues with credible runtime or scale impact.

5. Produce the review.
   - Start with the reviewed target and file count when available.
   - Lead with findings, grouped by severity.
   - Omit empty severity groups.
   - Include a specific positive observation only when there is one.
   - If there are no findings, say so directly.

Use this output shape:

```text
## Review Summary

Target: staged changes | unstaged changes | path/to/file | PR branch
Files: N

### Critical
- file.py:line - What is wrong. Fix: concrete direction.

### Warnings
- file.py:line - What is wrong. Fix: concrete direction.

### Suggestions
- file.py:line - Improvement or observation.

### Looks good
- Specific positive observation.
```

Severity guide:

| Level | Blocks merge? | Examples |
|-------|---------------|----------|
| Critical | Yes | Security holes, data loss, crashes |
| Warning | Usually | Secondary-path bugs, missing error handling |
| Suggestion | No | Naming, small refactors, missing docs |
| Looks good | N/A | Clean patterns, useful tests, clear boundaries |

## Pitfalls

- Empty diffs: check `git status`, report the state, and do not invent
  findings.
- Binary files: note that they were skipped.
- Large diffs: split by file and prioritize files with the largest or riskiest
  changes.
- Missing file path: report that the file could not be read and stop file mode.
- Non-git directory: explain that diff modes need git and suggest file mode.
- Uncertain line numbers: cite the nearest stable location and explain the
  context.

## Verification

- Confirm the selected target matches the user's invocation.
- Confirm every finding is tied to observed code, not speculation.
- Confirm critical and warning findings include a concrete fix direction.
- Confirm empty severity sections are omitted.
- Confirm a no-issue review says "No issues found. The changes look clean."
