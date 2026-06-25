---
name: simplicio-review
description: Deep adversarial code review — runs dual-perspective analysis (security/correctness + code quality) in parallel, then deduplicates into a single verdict. Use before merging any non-trivial change.
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [review, code-quality, security, adversarial]
    related_skills: [bug-pipeline, simplicio-tasks]
---

# Simplicio Review — Adversarial Branch Review

> Two parallel review rubrics → deduped into one verdict. Scoped strictly to the diff.

## Trigger

Before merging non-trivial work, when the user says "review this branch/PR hard", or when simplicio-tasks needs the verify gate.

## Procedure

1. Read the full diff: `git diff main...HEAD` or `gh-axi pr diff <N>`
2. Run TWO parallel reviews (delegate to `think` or `debugger` profile):
   - **Review A** — Security + correctness (CRITICAL/HIGH findings)
   - **Review B** — Code quality + patterns (MEDIUM/LOW findings)
3. Deduplicate findings into one verdict table
4. Verdict: APPROVE / WARN / BLOCK (same severity rules as bug-pipeline)

Record: `lore-axi kb-add key="simplicio-review/verdict" value="APPROVE|WARN|BLOCK"`
