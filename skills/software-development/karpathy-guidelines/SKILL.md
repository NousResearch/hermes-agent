---
name: karpathy-guidelines
description: 'Behavioral guidelines to reduce common LLM coding mistakes: think before coding, simplicity first, surgical changes, goal-driven execution. Use when writing, reviewing, or refactoring code.'
version: 1.0.0
author: Hermes Agent (adapted from karpathy-guidelines)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [guidelines, quality, simplicity, coding-standards]
    related_skills: [ponytail, code-simplification, coding-standards]
---

# Karpathy Guidelines

Behavioral guidelines to reduce common LLM coding mistakes, derived from Andrej Karpathy's observations on LLM coding pitfalls.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan before executing. Verify each step before proceeding.

## 5. Surface Assumptions

**Explicit is better than implicit. Wrong assumptions are the root of most bugs.**

Before coding:

- "I'm assuming the API returns JSON."
- "I'm assuming this runs on Node 18+."
- "I'm assuming the database schema hasn't changed."

Surface these BEFORE the user discovers them through a bug.

## Hermes Integration

- Apply before every `write_file` or `replace_string_in_file`
- Pair with `ponytail` for maximum simplicity enforcement
- Pair with `systematic-debugging` for bug fix discipline
- Use `search_files` to verify assumptions about existing patterns
- Use `read_file` to check conventions before editing
