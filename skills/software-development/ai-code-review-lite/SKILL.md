---
name: ai-code-review-lite
description: "Use when reviewing AI-generated code or unfamiliar-language diffs without running a heavy verification process. A simple confidence gate based on scope, tests, risk triggers, and rollback."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [code-review, ai-generated-code, confidence, lightweight-review]
    related_skills: [requesting-code-review, github-code-review, test-driven-development]
---

# AI Code Review Lite

## Overview

A lightweight review routine for AI-generated code. Use it when a full audit is too much, but blind trust is too little.

The goal is simple: decide whether the change is safe enough to merge, needs more evidence, or should move to a heavier review.

## When to Use

Use this when:

- An AI agent produced a small or medium code change.
- The diff is in a language you do not know deeply.
- You want a quick PR review with a confidence level.
- The change is not obviously high-risk.

Do not use this when the change touches auth, payments, permissions, secrets, database migrations, deployment, background jobs, or large refactors. Use `requesting-code-review` or a full review instead.

## The Five Checks

### 1. Scope

Ask:

- What was the task?
- Which files changed?
- Do the changed files match the task?

Red flags:

- unrelated files changed
- generated or lock files changed without explanation
- large diff for a small request

### 2. Tests

Require actual command output, not a claim.

Good:

```text
npm test
cargo test
python -m pytest
```

For bug fixes, prefer proof that a test failed before and passed after.

Red flags:

- “should work”
- “not tested”
- tests only cover the happy path

### 3. Risk Triggers

Escalate to heavier review if the diff includes:

- auth, roles, sessions, tokens
- database schema, migrations, raw SQL
- shell commands, filesystem writes/deletes
- secrets or credentials
- background jobs, queues, cron
- concurrency or shared state
- deploy, infra, CI/CD
- `eval`, `exec`, unsafe deserialization
- TypeScript `any` or broad `as SomeType` at API boundaries
- Rust `unsafe`, `unwrap()`, or `expect()` in runtime paths

### 4. Independent Pass

Ask another reviewer agent to check only for blockers:

```md
Review this task, evidence, and diff. Return only:
- approve / request changes / escalate
- missing evidence
- blocking correctness issues
- blocking security issues
- risky files to inspect manually
```

The implementer should not be the only reviewer.

### 5. Rollback

Ask:

- Can this be reverted cleanly?
- Is there a migration or irreversible side effect?
- If it breaks, what command or PR reverts it?

No rollback plan means no high confidence.

## Output Template

```md
## AI Code Review Lite

Verdict: Approve / Request changes / Escalate
Confidence: Low / Medium / High

### Evidence checked
- Scope matches task: yes/no
- Tests shown: yes/no, command:
- Risk triggers found: yes/no
- Independent pass: yes/no
- Rollback clear: yes/no

### Blocking issues
- none / list

### Manual spots checked
- file: reason

### Notes
- short reviewer note
```

## Confidence Rules

High confidence:

- scope is tight
- tests ran and match the change
- no risk triggers
- independent pass found no blockers
- rollback is clear

Medium confidence:

- one minor piece of evidence is missing, but the diff is small and low-risk

Low confidence:

- missing tests, vague scope, unfamiliar-language danger pattern, unexpected files, or unclear rollback

## Common Pitfalls

1. **Reading every line first.** Start with scope and evidence. Open code only where risk appears.
2. **Trusting summaries.** Summaries are useful, but command output is better.
3. **Overusing this skill.** If risk triggers appear, escalate. Lite review is not a parachute.
4. **Pretending language fluency.** In Rust or TypeScript, check the danger patterns before style.

## Verification Checklist

- [ ] Task and changed files are clear.
- [ ] Test output is present or missing tests are explicitly called out.
- [ ] Risk triggers were checked.
- [ ] Another reviewer or review pass checked for blockers.
- [ ] Rollback path is clear.
- [ ] Verdict and confidence are stated plainly.
