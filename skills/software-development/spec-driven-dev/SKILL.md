---
name: spec-driven-dev
description: "Spec-Driven Development: constitution, spec, plan, tasks."
version: 0.1.0
author: Hermes Agent (adapted from github/spec-kit pattern)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [planning, specification, spec-kit, design, constitution, acceptance-criteria, workflow]
    related_skills: [plan, spike, subagent-driven-development, test-driven-development, requesting-code-review]
---

# Spec-Driven Development

A Hermes-native port of the GitHub Spec Kit flow. Five-stage pipeline that
front-loads the *why* before the *what* and the *how*:

```
constitution  ->  spec  ->  plan  ->  tasks  ->  implement
   (why)        (what)    (how)    (slices)    (code)
```

Load this skill when the user wants to design a feature before coding it,
or when an existing plan is too thin to hand to `subagent-driven-development`.

## Core principle

Specs are **executable artifacts**, not documentation. Each stage produces a
file the next stage can read without re-deriving intent. If a stage needs
human judgment, it stops and asks — it does not guess.

## Pipeline at a glance

| Stage | Output file (under `.spec/<feature>/`) | Purpose |
|-------|----------------------------------------|---------|
| 1. Constitution | `constitution.md` | Non-negotiable project invariants. Frozen unless amended. |
| 2. Spec | `spec.md` | WHAT the feature does. User stories + acceptance criteria. |
| 3. Plan | `plan.md` | HOW we will build it. Architecture, tech choices, file layout. |
| 4. Tasks | `tasks.md` | Bite-sized 2-5 min slices with TDD cycles, ready for `subagent-driven-development`. |
| 5. Implement | (no file) | Hand tasks.md to `subagent-driven-development`. Updates each task inline as it lands. |

`.spec/` is the working tree of this skill. It is **read-mostly** after the
spec stage freezes — anything past spec.md should not mutate earlier stages.

## When to load this skill

**Load when:**
- User says "I want to design X before building it"
- User asks for a "spec", "specification", "design doc", or "PRD"
- A feature crosses 3+ files or spans multiple sessions
- Multiple profiles (coder, reviewer, orchestrator) will touch the work
- The work is ambiguous enough that plan-only would invent the answer

**Do NOT load when:**
- The task is a single-file bug fix or trivial refactor — use `systematic-debugging`
- The user wants raw feasibility exploration — use `spike`
- The user already has a good plan — use `plan` or `subagent-driven-development`
- The work is one-shot throwaway — use `sketch`

## How to run

1. Confirm scope: 1-3 sentences on what is being designed and why. Ask if unclear.
2. Load `references/constitution-template.md`. If `.spec/constitution.md`
   exists for the project, reuse it; otherwise scaffold one.
3. Load `references/spec-template.md`. Write `.spec/<slug>/spec.md`.
   Freeze, then move on.
4. Load `references/implementation-flow.md`. Write `.spec/<slug>/plan.md`.
5. Load `references/tasks-template.md`. Write `.spec/<slug>/tasks.md`.
6. Hand `tasks.md` to `subagent-driven-development`. Do not re-plan inline.
7. After implementation, append an outcome section to `spec.md`
   (acceptance-criteria checklist marked done + deviations).

See `references/implementation-flow.md` for the full gating logic between
stages, what counts as "frozen", and the amend-vs-revise rules.

## Tools referenced (Hermes-native)

This skill uses only built-in tools:

- `read_file`, `write_file`, `patch`, `search_files`
- `terminal` (read-only inspection commands)
- `todo` (to track stage transitions)
- `delegate_task` (via `subagent-driven-development` at stage 5)

No MCP server, no new dependency, no shell pipeline.

## Pitfalls

- **Skipping constitution.** Without invariants, spec debates reopen every stage.
- **Vague acceptance criteria.** "Works correctly" is not testable. Each criterion
  must be expressible as a yes/no check a reviewer can run in under 30 seconds.
- **Tasks larger than 5 minutes.** If a task is bigger, split it before handing
  off to `subagent-driven-development` — that skill assumes bite-size.
- **Mutating constitution to fix a plan.** Amend only when the invariant itself
  was wrong; otherwise fix the plan to honor the invariant.
- **Mixing stages.** A spec is not a plan. A plan is not tasks. If you find
  yourself writing implementation steps in `spec.md`, the scope leaked.
