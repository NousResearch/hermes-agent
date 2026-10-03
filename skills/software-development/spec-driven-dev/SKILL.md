---
name: spec-driven-dev
description: "Spec-Driven Development: constitution, spec, plan, tasks."
version: 0.1.0
author: Juan Ramon Gros (@jrgros-ops); Hermes Agent (adapted from github/spec-kit pattern)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [planning, specification, spec-kit, design, constitution, acceptance-criteria, workflow]
    related_skills: [plan, spike, subagent-driven-development, test-driven-development, requesting-code-review]
---

# Spec-Driven Development Skill

A Hermes-native port of the GitHub Spec Kit flow. Use it to produce executable
artifacts that carry a feature from project invariants through validated work:

```
constitution  ->  spec  ->  plan  ->  tasks  ->  implement
   (why)        (what)    (how)    (slices)    (validated execution)
```

Each stage produces a file the next stage can read without re-deriving intent.
If a stage needs human judgment, it stops and asks; it does not guess.

## When to Use

Load this skill when the user wants to design a feature before coding it, or
when an existing plan is too thin to hand to `subagent-driven-development`.

Use when:

- User says "I want to design X before building it".
- User asks for a "spec", "specification", "design doc", or "PRD".
- A feature crosses 3+ files or spans multiple sessions.
- Multiple profiles (coder, reviewer, orchestrator) will touch the work.
- The work is ambiguous enough that plan-only would invent the answer.

Do not use when:

- The task is a single-file bug fix or trivial refactor; use `systematic-debugging`.
- The user wants raw feasibility exploration; use `spike`.
- The user already has a good plan; use `plan` or `subagent-driven-development`.
- The work is one-shot throwaway; use `sketch`.

## Prerequisites

- Access to the repository or workspace being designed.
- A clear feature name that can become the `.spec/<feature>/` directory.
- Permission to create or update files under `.spec/`.
- The project constitution at `.spec/constitution.md`, if it already exists.

The constitution is project-scoped. It captures why the project exists, its
principles, and non-negotiable invariants reused across features. Feature
artifacts live under `.spec/<feature>/`, but the constitution stays at
`.spec/constitution.md` and is frozen unless explicitly amended.

This skill uses only built-in Hermes tools:

- `read_file`, `write_file`, `patch`, `search_files`
- `terminal` for read-only inspection commands
- `todo` to track stage transitions
- `delegate_task` through `subagent-driven-development` at stage 5

No MCP server, new dependency, or shell pipeline is required.

## How to Run

1. Confirm scope in 1-3 sentences: what is being designed and why. Ask if
   unclear; completion means the user-facing feature boundary is explicit.
2. Load `references/constitution-template.md`. Reuse `.spec/constitution.md`
   when it exists; otherwise scaffold it as the project-scoped constitution.
   Completion means the project invariants are available to every later stage.
3. Load `references/spec-template.md`. Write `.spec/<feature>/spec.md`.
   Completion means the feature's user stories and acceptance criteria are
   frozen enough to plan against.
4. Load `references/implementation-flow.md`. Write `.spec/<feature>/plan.md`.
   Completion means the implementation approach, architecture, tech choices,
   and expected file layout are documented.
5. Load `references/tasks-template.md`. Write `.spec/<feature>/tasks.md`.
   Completion means the plan is split into executable 2-5 minute slices.
6. Hand `.spec/<feature>/tasks.md` to `subagent-driven-development`. Do not
   re-plan inline. Completion means tasks have been executed and validated.
7. After implementation, append an outcome section to `.spec/<feature>/spec.md`
   with the acceptance-criteria checklist marked done and any deviations noted.

See `references/implementation-flow.md` for the full gating logic between
stages, what counts as frozen, and the amend-vs-revise rules.

## Quick Reference

| Stage | Output file | Purpose |
|-------|-------------|---------|
| 1. Constitution | `.spec/constitution.md` | WHY the project exists; principles and invariants reused across features. |
| 2. Spec | `.spec/<feature>/spec.md` | WHAT the feature must achieve; user stories and acceptance criteria. |
| 3. Plan | `.spec/<feature>/plan.md` | HOW the feature will be implemented; architecture, choices, and file layout. |
| 4. Tasks | `.spec/<feature>/tasks.md` | Executable work breakdown; bite-sized TDD-ready slices. |
| 5. Implement | No new spec file | Validated execution of the tasks, updating task state as work lands. |

`.spec/` is the working tree of this skill. After `spec.md` freezes, later
stages should not mutate earlier stages unless the user explicitly chooses an
amendment path.

## Procedure

1. Constitution: capture the project's why, principles, and invariants in
   `.spec/constitution.md`. Reuse it between features. Amend only when an
   invariant itself is wrong or incomplete, not to make a difficult plan easier.
2. Spec: capture what the feature must accomplish in `.spec/<feature>/spec.md`.
   Keep implementation details out. Acceptance criteria must be checkable as
   yes/no outcomes a reviewer can evaluate quickly.
3. Plan: capture how the feature will be implemented in `.spec/<feature>/plan.md`.
   Include architecture, key files, data flow, testing strategy, and constraints
   inherited from `.spec/constitution.md`.
4. Tasks: split the plan into `.spec/<feature>/tasks.md`. Each task should be
   small enough for a focused implementation pass and should identify its
   expected validation.
5. Implement: execute the tasks, preferably with `subagent-driven-development`
   when delegation fits. Mark tasks as complete only after the relevant tests,
   checks, or manual verification have actually run.

## Pitfalls

- Skipping constitution. Without project invariants, spec debates reopen every
  stage.
- Treating constitution as feature-scoped. The constitution belongs at
  `.spec/constitution.md` and is reused across features; feature work gets
  `spec.md`, `plan.md`, and `tasks.md` under `.spec/<feature>/`.
- Vague acceptance criteria. "Works correctly" is not testable. Each criterion
  must be expressible as a yes/no check a reviewer can run in under 30 seconds.
- Tasks larger than 5 minutes. If a task is bigger, split it before handing off
  to `subagent-driven-development`; that skill assumes bite-size work.
- Mutating constitution to fix a plan. Amend only when the invariant itself was
  wrong; otherwise fix the plan to honor the invariant.
- Mixing stages. A spec is not a plan. A plan is not tasks. If implementation
  steps appear in `spec.md`, the scope leaked.

## Verification

- Confirm `.spec/constitution.md` exists or was intentionally scaffolded as the
  project-scoped source of principles and invariants.
- Confirm `.spec/<feature>/spec.md` states what the feature must accomplish and
  has checkable acceptance criteria.
- Confirm `.spec/<feature>/plan.md` explains how the feature will be built and
  honors the constitution.
- Confirm `.spec/<feature>/tasks.md` breaks work into executable slices with
  validation expectations.
- Confirm implementation tasks were executed and validated before marking them
  complete.
- Confirm any post-implementation deviations are recorded in the feature spec.