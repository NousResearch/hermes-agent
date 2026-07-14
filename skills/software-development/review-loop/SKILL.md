---
name: review-loop
description: "Iterate on non-trivial changes with reviewer feedback: validate, review, fix, and revalidate before shipping."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [software-development, review, quality, feedback-loop, agentic-engineering]
    related_skills: [requesting-code-review, code-structure, simplify-code, subagent-driven-development]
---

# Review Loop Skill

Use this skill to turn a one-shot implementation into an iterative quality loop. The loop can use another LLM, a Hermes subagent, a human reviewer, GitHub PR comments, CI output, or an external code-review tool.

Core loop:

```text
implement -> run checks -> request review -> fix feedback -> re-run checks -> re-review -> ship
```

This skill complements `requesting-code-review`: use `review-loop` while converging on the solution, then use `requesting-code-review` as the final pre-commit/pre-push gate.

## When to Use

Use when:

- a feature/fix is implemented but has not been independently reviewed
- code was generated quickly and may contain hidden edge-case bugs
- the change touches auth, secrets, payments, user data, migrations, deployment, gateway behavior, cron jobs, or agent prompt/tool behavior
- CI, tests, a human, or a reviewer model produced feedback
- the user asks for “review loop”, “iterate until clean”, “fix review feedback”, or “don’t ship the first draft”

Skip only for trivial typo/doc changes or when the user explicitly says to skip review.

## Prerequisites

- A clean understanding of the intended behavior.
- The current diff or PR available.
- A way to run relevant validation: tests, build, lint, smoke command, or manual reproduction.
- For Hermes core changes, read root `AGENTS.md` and use `scripts/run_tests.sh` for pytest targets when possible.

## How to Run

1. Capture the current diff and validation baseline.
2. Run targeted tests/build/lint.
3. Ask a reviewer to inspect the diff.
4. Convert feedback into concrete fixes.
5. Apply fixes in focused passes.
6. Re-run validation.
7. Repeat until no blockers/majors remain.
8. Use `requesting-code-review` before final commit/push for non-trivial work.

## Quick Reference

Reviewer prompt template:

```text
Review the current git diff for correctness, security, maintainability, missing tests, and edge cases.
Return blockers first, then major issues, then minor suggestions.
Do not modify files. Include file paths and line references when possible.
Assume the intended behavior is: <one-sentence behavior>.
```

Feedback classification:

- **Blocker**: correctness regression, data loss, security issue, broken build, failing required test.
- **Major**: missing validation, brittle edge case, hard-to-maintain structure, untested critical path.
- **Minor**: naming, comments, small readability improvements, optional polish.

Exit criteria:

- All blockers fixed or explicitly reported as blocked.
- Major issues fixed or consciously deferred with rationale.
- Relevant validation rerun after fixes.
- Diff reviewed for secrets and unrelated changes.

## Procedure

### 1. Establish baseline

Run:

```bash
git status --short
git diff --stat
git diff --name-only
```

Then run the narrowest relevant validation. For Hermes core:

```bash
scripts/run_tests.sh tests/path/test_file.py -q
```

If the repo has known baseline failures, record them before judging your changes.

### 2. Collect review feedback

Prefer one of:

- `delegate_task` reviewer subagent with file/diff context
- existing PR comments or CI logs
- external reviewer CLI/API if already configured
- human review comments
- local checklist if no reviewer is available

Do not invent review output. If review tooling is unavailable, say so and use the checklist honestly.

### 3. Fix in small loops

For each loop:

1. Pick one blocker/major cluster.
2. Patch only the relevant files.
3. Re-run the targeted validation.
4. Re-open the diff and confirm no unrelated changes slipped in.
5. Repeat.

### 4. Finalize

Before shipping, summarize:

- validation commands and results
- feedback received
- fixes applied
- remaining risks
- commit/push status

## Pitfalls

1. **Calling a first pass “reviewed.”** A review loop needs feedback plus fixes plus re-validation.
2. **Letting reviewer suggestions cause scope creep.** Fix blockers/majors; defer unrelated ideas.
3. **Ignoring CI because local tests pass.** CI is another reviewer; consume its feedback.
4. **Asking the same context to self-review.** Fresh context catches more defects.
5. **Fabricating external review.** Only report tools and checks that actually ran.

## Verification

Before finishing:

- [ ] Diff and intended behavior were inspected
- [ ] Relevant tests/build/lint/smoke checks ran
- [ ] Independent or checklist-based review was performed
- [ ] Blockers/majors were fixed or explicitly reported
- [ ] Validation reran after fixes
- [ ] Final diff is secret-clean and scoped
