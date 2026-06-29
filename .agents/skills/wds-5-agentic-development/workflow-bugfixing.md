---
name: bugfixing
description: Fix bugs in existing code through structured investigation and verification
---

# [F] Bugfixing — Fix Bugs in Existing Code

**Goal:** Systematically investigate, fix, and verify bugs with minimal side effects.

**When to use:** A bug has been reported or discovered in existing code.

---

## CORE PRINCIPLES

- **Reproduce first** — Never fix what you cannot reproduce. A fix without reproduction is a guess.
- **Minimal fix** — Target the root cause with the smallest change possible. Do not refactor surrounding code during a bugfix.
- **Regression check** — Every fix must be verified against the original bug AND tested for side effects on related functionality.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.

### Essential Guides

- **[Execution Principles](data/guides/EXECUTION-PRINCIPLES.md)** — Document-first, plan-then-execute
- **[Session Protocol](data/guides/SESSION-PROTOCOL.md)** — Read dialog, verify plan, present status
- **[Inline Testing Guide](data/guides/INLINE-TESTING-GUIDE.md)** — Verify fixes with Puppeteer before presenting

---

## STEPS

Execute steps in `./steps-f/`:

| Step | File | Purpose |
|------|------|---------|
| 01 | step-01-reproduce.md | Reproduce and document the bug |
| 02 | step-02-investigate.md | Investigate root cause |
| 03 | step-03-fix.md | Implement the fix |
| 04 | step-04-verify.md | Verify fix, check regressions |
| 05 | step-05-document.md | Document fix, update tests |

**Flow:** 01 → 02 → 03 → 04 → 05

### Critical Rules

- **ALWAYS** reproduce the bug before investigating
- **ALWAYS** identify root cause before writing a fix
- **ALWAYS** create a test that catches the bug before fixing
- **ALWAYS** run regression checks after fixing
- **NEVER** refactor surrounding code in the same fix
- **NEVER** fix symptoms — fix the root cause

---

## AFTER COMPLETION

1. Append a progress entry to `{output_folder}/_progress/00-design-log.md` under `## Progress`:
   `### [date] — Bugfix: [what was fixed]`
2. Suggest re-running acceptance testing (Phase 5 [T]) to verify the fix
