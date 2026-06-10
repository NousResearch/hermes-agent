---
name: code-simplifier
description: "Use when code works but feels unnecessarily complex, over-abstracted, hard to scan, or difficult to explain. Simplify while preserving behavior and validation."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [software-development, simplification, refactor, readability, agentic-engineering]
    related_skills: [code-structure, gpt-loop, requesting-code-review]
---

# Code Simplifier Skill

Use this skill when code works but the implementation is heavier than the problem. The target is boring, obvious code with the same behavior.

This is not a rewrite license. Simplify accidental complexity, preserve essential domain/security behavior, and verify with tests or smoke checks.

## When to Use

Use when:

- the user says “simplify”, “too complex”, “make this cleaner”, or “reduce slop”
- abstractions hide simple logic
- a file has too many layers, callbacks, flags, or clever helpers
- deeply nested conditionals obscure the happy path
- duplicated branches can be collapsed safely
- naming makes code harder to understand than necessary
- future agents would likely break the code because it is too clever

Do not use when:

- complexity is necessary domain logic, protocol handling, concurrency, security, or compatibility work
- there is no way to validate behavior and the code path is risky
- the user actually asked for a new feature or architecture change

## Prerequisites

- Identify the behavior that must not change.
- Find existing tests or define a minimal smoke check.
- Read any relevant local references in `.references/` only if API/library behavior is uncertain.

## How to Run

1. State the current behavior and public contract.
2. Identify accidental complexity.
3. Simplify one cluster at a time.
4. Preserve real boundaries and safety checks.
5. Run validation after simplification.
6. Use `gpt-loop` for non-trivial simplification passes.

## Quick Reference

Good simplifications:

- inline one-use wrappers
- replace clever expressions with explicit branches
- collapse duplicated code with a small helper
- rename variables/functions to describe intent
- turn boolean flag tangles into clearer branch functions
- remove dead code and stale comments
- reduce nested conditionals with guard clauses

Bad simplifications:

- removing validation because it “looks redundant”
- flattening real architecture boundaries
- changing public APIs for aesthetics
- deleting edge-case branches without proving they are unreachable
- hiding complexity behind a generic helper with vague names

## Procedure

### 1. Freeze behavior

Before editing, write down:

- public inputs/outputs
- expected side effects
- edge cases already handled
- relevant tests or smoke checks

If behavior is not clear, inspect or test first.

### 2. Find accidental complexity

Look for:

- helpers used exactly once
- config flags that always have one value
- duplicated branches differing only in constants
- nested `if`/`else` chains where guard clauses would clarify
- names like `process`, `handle`, `data`, `manager`, `thing`
- comments explaining code that could be clearer with better names

### 3. Simplify in focused patches

- Preserve public function names and signatures unless the task includes API cleanup.
- Keep safety checks: auth, validation, path checks, rate limits, retries, idempotency.
- Prefer explicit local code over “generic” helpers that hide intent.
- Keep formatting churn separate from logic simplification.

### 4. Validate and compare

Run the relevant checks. If possible, compare before/after behavior with the same input.

For Hermes core, use targeted tests first:

```bash
scripts/run_tests.sh tests/path/test_file.py -q
```

If no test exists, run a smoke command and state the limitation.

## Pitfalls

1. **Simplifying away edge cases.** Edge branches often exist for a reason; prove they are dead before deleting.
2. **Inlining everything.** Simpler is not always flatter; keep meaningful boundaries.
3. **Breaking public contracts.** API/CLI/schema changes are feature work, not simplification.
4. **Deleting useful comments.** Remove stale comments, not domain warnings.
5. **No validation.** Simpler-looking code that is untested is just a guess.

## Verification

Before finishing:

- [ ] Behavior was stated before edits
- [ ] Removed complexity was accidental, not essential
- [ ] Public API stayed stable or change was explicitly requested
- [ ] Tests/build/smoke checks ran
- [ ] Diff is easier to scan and has no unrelated changes
- [ ] Security/validation checks were preserved
