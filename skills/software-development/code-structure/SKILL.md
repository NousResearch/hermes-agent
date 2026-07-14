---
name: code-structure
description: "Restructure messy working code into clear boundaries without changing behavior."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [software-development, refactor, architecture, structure, agentic-engineering]
    related_skills: [review-loop, simplify-code, requesting-code-review]
---

# Code Structure Skill

Restructure messy working code into clean boundaries that humans and future agents can understand. This is behavior-preserving refactoring, not a redesign license.

The goal is to remove AI-generated slop: giant files, mixed concerns, hidden side effects, duplicated branches, and code that cannot be tested without booting the whole app.

## When to Use

Use when:

- a feature works but the implementation is tangled or hard to extend
- UI/API/database/external-service logic is mixed in the same file
- a file or function is too large for a future agent to safely edit
- business logic is embedded inside route handlers, UI components, or CLI parsing
- duplicate code appears in multiple places
- tests are difficult because side effects are everywhere
- the user says the implementation feels like “slop” or asks for “code structure”

Do not use for:

- speculative rewrites before behavior is understood
- replacing a simple clear function with layers of abstractions
- architecture astronaut work with no validation path

## Prerequisites

- Read the current implementation and understand the public behavior.
- Check existing tests/build commands from `AGENTS.md`, README, package config, or project docs.
- If the repo has local upstream snapshots in `.references/`, search only the relevant folder; do not dump the whole reference tree into context.

## How to Run

1. Inspect the current diff and relevant files.
2. State the behavior that must remain unchanged.
3. Extract clear boundaries in small, reviewable steps.
4. Run targeted validation after each risky move.
5. Use `review-loop` while iterating and `requesting-code-review` before shipping non-trivial refactors.

## Quick Reference

Common boundaries:

- **Route/controller**: request parsing, auth check handoff, response shaping.
- **Service/use case**: business workflow and orchestration.
- **Repository/data access**: persistence and query details.
- **Integration/client**: external APIs, retries, rate limits, response normalization.
- **Validation/schema**: input validation and normalization.
- **UI component/view**: rendering and direct user interaction only.
- **Pure utility**: deterministic transformations with no network, filesystem, DB, or time dependencies.

## Procedure

### 1. Map behavior before editing

Write down:

- inputs and outputs
- public API or CLI/UI contract
- side effects: database, filesystem, network, env vars, logs, cron, gateway messages
- existing tests or smoke checks
- likely edge cases

If you cannot explain what the code does, do not restructure it yet.

### 2. Identify the smallest useful boundary

Prefer one of these first moves:

- extract a pure helper from a route/component
- move external API calls into a client wrapper
- move persistence into a repository function
- move workflow orchestration into a service function
- split validation from execution

Avoid creating abstract base classes, plugin systems, factories, or dependency containers unless the code already needs them.

### 3. Move code mechanically

- Preserve public function names and call signatures unless changing them is part of the task.
- Keep import direction clean: routes/UI can depend on services; services can depend on repositories/clients; repositories/clients should not import routes/UI.
- Do not change formatting and structure everywhere in one patch.
- Keep commits/diffs grouped by behavior boundary.

### 4. Add seams for tests

- Inject external clients where practical.
- Keep pure functions free of global mutable state.
- Move time/random/env reads to the edge or make them injectable.
- Prefer testable return values over printing/logging-only behavior.

### 5. Validate

Run the smallest command that proves behavior still works. Examples:

```bash
scripts/run_tests.sh tests/path/test_file.py -q
python -m pytest tests/path/test_file.py -q
npm test -- --runInBand path/to/test
npm run build
```

If no tests exist, run a smoke command and document the gap.

## Pitfalls

1. **Refactoring while changing behavior.** Keep behavior changes separate unless the user explicitly asked for a feature/fix.
2. **Over-abstracting.** One clear function is often better than five “architecture” files.
3. **Breaking import boundaries.** Lower layers should not import upper layers.
4. **Moving secrets into code.** Runtime config belongs in `.env`, config files, or secret stores — never in tracked source.
5. **Skipping validation.** A prettier diff that does not run is a regression.

## Verification

Before finishing:

- [ ] Behavior to preserve was stated
- [ ] Boundaries are clearer than before
- [ ] No unrelated rewrite was introduced
- [ ] Targeted tests/build/smoke checks ran
- [ ] Diff was reviewed for accidental secrets and import cycles
- [ ] Any missing tests or unresolved risks were reported
