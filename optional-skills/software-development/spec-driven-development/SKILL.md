---
name: spec-driven-development
description: "Spec-driven dev with executable contracts, TDD, subagents."
version: 1.0.0
author: Rafael Zendron (rafaumeu)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [spec, planning, decomposition, context-engineering, tdd, contracts]
    related_skills: [plan, test-driven-development, subagent-driven-development]
    requires_toolsets: [terminal]
---

# Spec-Driven Development

Build features through executable specifications, not ad-hoc coding. Each feature starts as a decomposed spec with testable contracts, then gets implemented via TDD by subagents.

**What it does:** Forces a MAPEAR > SPEC > VALIDATE > IMPLEMENT pipeline for every feature.
**What it doesn't do:** Replace the `plan` skill. Plans are broader; specs are per-task contracts.

## When to Use

- Any feature that touches > 1 file
- Multi-step implementations requiring subagents
- When you need deterministic output from AI coding agents
- Before any non-trivial code change

## Prerequisites

- A project with a test framework configured (pytest, vitest, etc.)
- `delegate_task` available for subagent dispatch

## Procedure

### Step 1: MAPEAR — Understand existing code

Before writing any spec:

1. List files that will be touched
2. Check for existing implementations that the spec might duplicate:
   - Search for existing DB tables: `SELECT table_name FROM information_schema.tables`
   - Search for existing routes: `search_files` with the route pattern
   - Check existing types/interfaces
3. Record what exists so specs don't propose creating duplicates

### Step 2: Create spec directory

```
.planning/<feature>/
├── README.md              ← Index: problem, solution, order, rules
└── specs/
    ├── TASK-001.md        ← Spec 1
    └── TASK-002.md        ← Spec 2
```

### Step 3: Write each spec (TASK-XXX.md)

Every spec MUST contain these sections:

```markdown
# TASK-XXX: [name]

## WHAT
[One sentence describing what this task builds]

## DEPENDENCIES
- TASK-xxx: [what needs to be ready first]

## IMPORTS
[Exact imports this task needs]

## WHAT TO CREATE
File: `src/path/file.ts`
[Description + copy-pasteable code skeleton]

## DERIVED TESTS
given [pre-condition]
when [action]
then [expected result]
  and [expected side effect]

Mandatory edge cases:
- [ ] Happy path
- [ ] Invalid input / edge case
- [ ] External dependency failure (if applicable)

## OUTPUT CONTRACT
[What this task exports — Zod schema, TypeScript type, or N/A]

## PITFALLS
- [Known edge cases for this specific task]

## ACCEPTANCE CRITERIA
- [ ] [max 5 testable items — each MUST have at least 1 mechanically derivable test]
- [ ] No ambiguous terms (avoid: "adequate", "fast", "correct" without metric)

## CI VALIDATION
[Which CI gate validates this — lint, test, build, security]

## ESTIMATE
LOC: ~[100-200] | Subagent time: ~[5-15 min]
```

### Step 4: Spec Review Gate (blocking)

A spec is NOT ready for implementation if any item fails:

- [ ] Acceptance criteria has at least one mechanically derivable test
- [ ] `## DERIVED TESTS` section filled with pseudocode
- [ ] `## OUTPUT CONTRACT` filled (or explicitly marked N/A)
- [ ] Dependencies listed with output contracts referenced
- [ ] No ambiguous terms in acceptance criteria

If any item fails: spec goes back to refinement. Agent does NOT start implementation.

### Step 5: Spec Validation Loop

Before implementing, diff the spec against the real project state:

1. Does the spec propose creating something that already exists?
2. Do the imports reference modules that don't exist yet?
3. Does the output contract match what downstream tasks expect?

If the spec proposes creating a duplicate: spec goes back to refinement.

### Step 6: Execute via subagents

- Each TASK = 1 subagent via `delegate_task`, max 2 parallel
- Explicit dependency: `DEPENDENCIES: [TASK-xxx]`
- Subagent reads spec + dependencies, implements without context of other specs

### Step 7: Checkpoint per phase

After each phase completes:

- What was done (summary)
- Current state (branch, commit hash, tests passing?)
- Next phase
- Blockers

## Key Rules

1. **200 LOC max per spec.** One spec = one independent subagent task.
2. **No spec = no code.** Ever. Even when it seems simple.
3. **Spec Review Gate is blocking.** No implementation without passing the gate.
4. **Spec Validation Loop catches duplicates.** Diff against real state before implementing.
5. **Subagents never expand scope.** Work outside the TASK = propose new TASK + stop.

## Pitfalls

- **Spec proposes duplicate code.** Always search existing codebase before writing a spec. If it already exists, the spec should reference it, not recreate it.
- **Spec too broad (> 200 LOC).** Split into smaller specs. Each spec = 1 independent subagent task.
- **Skipping spec for "simple" features.** Simple features grow. Spec always. Even if it's 5 lines.
- **Subagent expanding scope.** If a subagent starts working outside its TASK, abort it, create a new TASK, restart.
- **Missing output contract.** Without a contract, downstream tasks can't validate their inputs. Always specify.
- **Ambiguous acceptance criteria.** Words like "adequate", "fast", "correct" without a metric = spec needs refinement.

## Rush Mode

For production bugs with clear symptoms or trivial 1-3 line fixes:

1. Quick trace: UI → API → DB, find root cause
2. Skip spec — for 1-line fixes with obvious root cause
3. TDD still mandatory — regression test FIRST, then fix
4. Commit + PR + merge fast cycle

This is NOT for features or refactors. Only for: production bugs with clear symptoms, 1-3 line fixes, obvious typos.

## Agent Guardrails

- **Abort rule:** 3 consecutive failures on the same step = STOP, document, escalate to human.
- **Scope boundary:** Subagent never expands beyond the delegated TASK.
- **Human confirmation required for:** `git push --force`, `DROP TABLE` without `WHERE`, production deploys outside window, secret rotation.

## Verification

After all tasks complete:

1. All specs have corresponding passing tests
2. `validate:pr` (lint + type-check + test) passes
3. No orphaned specs (every spec has a commit)
4. README.md updated with final state

Run: `pytest tests/ -q && npm run lint && npm run build`
