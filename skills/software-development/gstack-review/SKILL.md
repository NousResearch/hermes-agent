---
name: gstack-review
description: 'Pre-landing PR review with structural-issue checklist + 8 specialist lenses. Use when asked to review a PR, code review, check diff, or pre-landing review.'
version: 1.0.0
author: Hermes Agent (adapted from garrytan/gstack, MIT)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [code-review, PR, quality, security, testing, pre-landing]
    related_skills: [github-code-review, systematic-debugging, test-driven-development]
---

# gstack-review — Pre-Landing PR Review

Adapted from [gstack](https://github.com/garrytan/gstack) (MIT, © 2026 Garry Tan).

## When to invoke

Analyze the current branch's diff against the base branch for **structural issues that tests don't catch**. Use when the user says "review this PR", "code review", "pre-landing review", or "check my diff". Proactively suggest when the user is about to merge or land code changes.

## Workflow

### Step 0: Detect base branch

Determine the base branch:

- **GitHub:** `gh pr view --json baseRefName -q .baseRefName` (if PR exists); else `gh repo view --json defaultBranchRef -q .defaultBranchRef.name`
- **Git-native fallback:** `git symbolic-ref refs/remotes/origin/HEAD` → strip prefix; else try `origin/main`, then `origin/master`; else default to `main`

Substitute the detected base branch in every subsequent `git diff`/`git log`/`git fetch` command.

### Step 1: Check branch

1. `git branch --show-current` — get current branch.
2. If on the base branch, output: **"Nothing to review — you're on the base branch or have no changes against it."** and stop.
3. `git fetch origin <base> --quiet` then `DIFF_BASE=$(git merge-base origin/<base> HEAD)` then `git diff "$DIFF_BASE" --stat` — if no diff, output the same message and stop.

### Step 2: Read the checklist

Read [`reference/checklist.md`](reference/checklist.md). **If the file cannot be read, STOP and report the error.** Do not proceed without the checklist.

### Step 3: Scope drift detection (before code quality)

1. Read `TODOS.md` if it exists. Read the PR description (`gh pr view --json body --jq .body 2>/dev/null || true`). Read commit messages (`git log origin/<base>..HEAD --oneline`).
2. Identify the **stated intent** — what was this branch supposed to accomplish?
3. `git diff "$DIFF_BASE" --stat` and compare files changed against stated intent.
4. Evaluate with skepticism:
   - **SCOPE CREEP:** files changed unrelated to stated intent; new features/refactors not in the plan; "while I was in there…" changes.
   - **MISSING REQUIREMENTS:** plan/PR items not addressed in the diff; test coverage gaps; partial implementations.
5. Output before the main review:
   ```
   Scope Check: [CLEAN / DRIFT DETECTED / REQUIREMENTS MISSING]
   Intent: <1-line summary of what was requested>
   Delivered: <1-line summary of what the diff actually does>
   [If drift: list each out-of-scope change]
   [If missing: list each unaddressed requirement]
   ```
6. This is **INFORMATIONAL** — does not block. Proceed.

### Step 4: Two-pass review against the diff

Apply the CRITICAL categories from [`reference/checklist.md`](reference/checklist.md) against the diff:

**Pass 1 (CRITICAL — highest severity):**

- SQL & Data Safety
- Race Conditions & Concurrency
- LLM Output Trust Boundary
- Shell Injection
- Enum & Value Completeness

**Pass 2 (INFORMATIONAL):**
Async/Sync Mixing · Column/Field Name Safety · Dead Code (version only) · LLM Prompt Issues · Completeness Gaps · Time Window Safety · Type Coercion at Boundaries · View/Frontend · Distribution & CI/CD Pipeline

Follow the output format and the "DO NOT flag" suppressions in the checklist — do NOT flag items in the suppression list, and do NOT re-flag anything already addressed in the full diff.

### Step 5: Specialist lenses (use delegate_task for parallel review)

Dispatch specialist subagents in parallel via Hermes's `delegate_task` tool. Each reads its checklist file and applies it to `git diff "$DIFF_BASE"`. Specialist scope gates (from the checklist):

- [`reference/security.md`](reference/security.md) — when SCOPE_AUTH=true OR (SCOPE_BACKEND=true AND diff > 100 lines)
- [`reference/testing.md`](reference/testing.md) — **always-on**
- [`reference/maintainability.md`](reference/maintainability.md) — always-on
- [`reference/performance.md`](reference/performance.md) — always-on
- [`reference/data-migration.md`](reference/data-migration.md) — when migrations present
- [`reference/api-contract.md`](reference/api-contract.md) — when API surface changes
- [`reference/red-team.md`](reference/red-team.md) — when diff > 200 lines OR security specialist found CRITICAL findings. **Runs after** the others — adversarial analysis looking for what they missed.

### Step 6: Fix-first heuristic

For each finding, decide AUTO-FIX vs ASK per the heuristic in [`reference/checklist.md`](reference/checklist.md). Rule of thumb: if the fix is mechanical and a senior engineer would apply it without discussion, AUTO-FIX; if reasonable engineers could disagree, ASK. Critical findings default toward ASK; informational findings default toward AUTO-FIX.

### Step 7: Output format

```
Pre-Landing Review: N issues (X critical, Y informational)

**AUTO-FIXED:**
- [file:line] Problem → fix applied

**NEEDS INPUT:**
- [file:line] Problem description
  Recommended fix: suggested fix
```

If no issues: `Pre-Landing Review: No issues found.`

Be terse. For each issue: one line describing the problem, one line with the fix. No preamble, no summaries, no "looks good overall."

## Voice

- Lead with the point. Say what it does, why it matters, what changes for the builder.
- Be concrete. Name files, functions, line numbers, commands, outputs, evals, real numbers.
- Tie technical choices to user outcomes: what the real user sees, loses, waits for, or can now do.
- Be direct about quality. Bugs matter. Edge cases matter. Fix the whole thing, not the demo path.
- Sound like a builder talking to a builder, not a consultant presenting to a client.
- No em dashes. No AI vocabulary: delve, crucial, robust, comprehensive, nuanced, multifaceted, furthermore, moreover, additionally, pivotal, landscape, tapestry, underscore, foster, showcase, intricate, vibrant, fundamental, significant.

## Completeness Principle

AI makes completeness cheap, so the complete thing is the goal. Recommend full coverage (tests, edge cases, error paths). When options differ in coverage, include `Completeness: X/10` (10 = all edge cases, 7 = happy path, 3 = shortcut). The only out-of-scope thing is genuinely unrelated work; flag that as separate scope, never as an excuse for a shortcut.

## Confusion Protocol

For high-stakes ambiguity (architecture, data model, destructive scope, missing context): STOP. Name it in one sentence, present 2-3 options with tradeoffs, and ask. Do not use for routine coding or obvious changes.

## Completion Status

Report status using one of: **DONE** (with evidence), **DONE_WITH_CONCERNS** (list concerns), **BLOCKED** (state blocker + what was tried), or **NEEDS_CONTEXT** (state exactly what's needed).

## Hermes Integration

- Use `terminal` tool for all git commands
- Use `delegate_task` for parallel specialist lens review
- Use `read_file` to read checklist files
- Use `search_files` to find patterns across the codebase
- Use `gh` CLI when available for PR metadata; fall back to `git` commands

## License

Methodology ported from [garrytan/gstack](https://github.com/garrytan/gstack) under the MIT License. Copyright (c) 2026 Garry Tan.
