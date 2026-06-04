---
name: sparc-methodology
description: >
  SPARC development methodology. 5-phase structured approach:
  Specification → Pseudocode → Architecture → Refinement → Completion.
  Use for complex feature development or any task where "just start coding" would fail.
tags: [methodology, development, planning, sparc]
---

# SPARC Methodology

5-phase structured development. Use when a task is complex enough that jumping to code would miss important design decisions.

## When to Use
- New feature with multiple components
- System redesign or major refactor
- Anything touching 3+ files
- Tasks with unclear requirements

## When NOT to Use
- Simple bug fix (one file, clear cause)
- Small config change
- Anything a single `patch()` call handles

## The 5 Phases

### 1. SPECIFICATION — Understand WHAT before HOW
- Gather requirements from user request, existing code, docs
- Define constraints (tech stack, performance, compatibility)
- Write acceptance criteria (how do we know it's done?)
- Identify scope boundaries (IN vs OUT)
- Flag ambiguities — ask before assuming
- **NEVER hard-code env vars or secrets**

### 2. PSEUDOCODE — Think through logic without syntax
- Write high-level algorithm flow
- Identify data structures and interfaces
- Trace through edge cases
- Add TDD anchors (test points)

### 3. ARCHITECTURE — Decide structure before implementation
- Design module/file organization
- Define component boundaries and data flow
- Select libraries/tools if needed
- **ALL files must be < 500 lines** (decompose larger files)

### 4. REFINEMENT — Implement, test, iterate
- Implement core functionality first
- Write tests alongside code (TDD: red-green-refactor)
- Run tests after each change
- Refactor as patterns emerge
- Target 90%+ test coverage for core logic

### 5. COMPLETION — Verify, document, deliver
- Run full test suite
- Verify all acceptance criteria met
- Check for security issues
- Report what was done with evidence

## Quick SPARC (medium tasks)

1. SPECIFY: One sentence + 3 acceptance criteria
2. PLAN: List files to change and why
3. IMPLEMENT: Code + test in a loop
4. VERIFY: Run tests, check criteria
5. DONE: Report with evidence

## Rules (All Phases)
1. ALL files < 500 lines
2. NO hard-coded secrets or env vars
3. Auto-format on save
4. After writing code, run corresponding tests
5. Store learnings in memory for future tasks
