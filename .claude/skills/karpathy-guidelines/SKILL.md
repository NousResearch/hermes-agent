---
name: karpathy-guidelines
description: Behavioural guidelines to reduce common LLM coding mistakes. Use when writing, reviewing, or refactoring code to avoid overcomplication, make surgical changes, surface assumptions, and define verifiable success criteria.
license: MIT
---

# Karpathy Guidelines

Behavioural guidelines to reduce common LLM coding mistakes, derived from
Andrej Karpathy's observations on LLM coding pitfalls. Source:
`forrestchang/andrej-karpathy-skills`.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial
tasks (typos, one-line CSS, obvious renames), use judgment — don't write a
test harness for a typo.

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- When your changes create orphans, remove the imports/variables/functions
  that YOUR changes made unused. Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the request.

## 4. Goal-Driven Execution

Define success criteria. Loop until verified.

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan with a verification per step. Strong
success criteria let you loop independently; weak criteria ("make it work")
require constant clarification.

## JJ-specific calibration

- **Stacked PRs (Merovingian / Morpheus / Morpheus LAGIC):** rule 3 is the brake.
  Cleaning up an orphan caused by a previous PR in the stack is in scope.
  "While we're here, let's tidy X" is drive-by — mention it, don't do it.
- **Carry-forward orphans:** valid cleanup items, but only when explicitly
  tasked. Surface them in Open Threads on wrap-up; don't sweep them mid-session.
- **Security backlog (XSS, formula injection):** exactly the shape for rule 4 —
  write a failing test first, then make it pass. Don't "fix the bug" by eyeballing.
- **Trivial fixes:** the tradeoff carve-out applies. Don't write a test harness
  for a typo.
