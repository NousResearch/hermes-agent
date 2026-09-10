---
name: spec-driven-development
description: 'Creates specs before coding. Use when starting a new project, feature, or significant change and no specification exists yet. Use when requirements are unclear, ambiguous, or only exist as a vague idea.'
version: 1.0.0
author: Hermes Agent (adapted from agent-skills)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [specification, planning, requirements, design, architecture]
    related_skills: [writing-plans, executing-plans, test-driven-development, idea-to-design]
---

# Spec-Driven Development

## Overview

Write a structured specification before writing any code. The spec is the shared source of truth between you and the human engineer — it defines what we're building, why, and how we'll know it's done. Code without a spec is guessing.

## When to Use

- Starting a new project or feature
- Requirements are ambiguous or incomplete
- The change touches multiple files or modules
- You're about to make an architectural decision
- The task would take more than 30 minutes to implement

**When NOT to use:** Single-line fixes, typo corrections, or changes where requirements are unambiguous and self-contained.

## The Gated Workflow

Spec-driven development has four phases. Do not advance to the next phase until the current one is validated.

```
SPECIFY → PLAN → TASKS → IMPLEMENT
   │        │       │         │
   ▼        ▼       ▼         ▼
 Human    Human   Human     Human
 reviews  reviews reviews   reviews
```

### Phase 1: Specify

Start with a high-level vision. Ask the human clarifying questions until requirements are concrete.

**Surface assumptions immediately.** Before writing any spec content, list what you're assuming:

```
ASSUMPTIONS I'M MAKING:
1. This is a web application (not native mobile)
2. Authentication uses session-based cookies (not JWT)
3. The database is PostgreSQL (based on existing Prisma schema)
4. We're targeting modern browsers only (no IE11)
→ Correct me now or I'll proceed with these.
```

**Write a spec document covering these six core areas:**

1. **Objective** — What are we building and why? Who is the user? What does success look like?

2. **Commands** — Full executable commands with flags, not just tool names.

   ```
   Build: npm run build
   Test: npm test -- --coverage
   Lint: npm run lint --fix
   Dev: npm run dev
   ```

3. **Project Structure** — Where source code lives, where tests go, where docs belong.

4. **Code Style** — One real code snippet showing your style beats three paragraphs describing it.

5. **Testing Strategy** — What framework, where tests live, coverage expectations.

6. **Boundaries** — Three-tier system:
   - **Always do:** Run tests before commits, follow naming conventions, validate inputs
   - **Ask first:** Database schema changes, adding dependencies, changing CI config
   - **Never do:** Commit secrets, edit vendor directories, remove failing tests without approval

**Reframe instructions as success criteria.** When receiving vague requirements, translate them into concrete conditions:

```
REQUIREMENT: "Make the dashboard faster"

REFRAMED SUCCESS CRITERIA:
- Dashboard LCP < 2.5s on 4G connection
- Initial data load completes in < 500ms
- No layout shift during load (CLS < 0.1)
→ Are these the right targets?
```

### Phase 2: Plan

With the validated spec, generate a technical implementation plan:

1. Identify the major components and their dependencies
2. Determine the implementation order (what must be built first)
3. Note risks and mitigation strategies
4. Identify what can be built in parallel vs. what must be sequential
5. Define verification checkpoints between phases

Save the plan to `docs/plans/` and the task list to `docs/plans/tasks.md`.

### Phase 3: Tasks

Break the plan into discrete, implementable tasks:

- Each task should be completable in a single focused session
- Each task has explicit acceptance criteria
- Each task includes a verification step (test, build, manual check)
- Tasks are ordered by dependency, not by perceived importance
- No task should require changing more than ~5 files

### Phase 4: Implement

Execute tasks one at a time following `test-driven-development` and `incremental-implementation`. Use `context-engineering` to load the right spec sections and source files at each step.

## Keeping the Spec Alive

The spec is a living document, not a one-time artifact:

- **Update when decisions change** — If you discover the data model needs to change, update the spec first, then implement.
- **Update when scope changes** — Features added or cut should be reflected in the spec.
- **Commit the spec** — The spec belongs in version control alongside the code.
- **Reference the spec in PRs** — Link back to the spec section that each PR implements.

## Common Rationalizations

| Rationalization                       | Reality                                                                                                 |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| "This is simple, I don't need a spec" | Simple tasks don't need _long_ specs, but they still need acceptance criteria. A two-line spec is fine. |
| "I'll write the spec after I code it" | That's documentation, not specification. The spec's value is in forcing clarity _before_ code.          |
| "The spec will slow us down"          | A 15-minute spec prevents hours of rework.                                                              |
| "Requirements will change anyway"     | That's why the spec is a living document. An outdated spec is still better than no spec.                |

## Red Flags

- Starting to write code without any written requirements
- Asking "should I just start building?" before clarifying what "done" means
- Implementing features not mentioned in any spec or task list
- Making architectural decisions without documenting them
- Skipping the spec because "it's obvious what to build"

## Hermes Integration

- Use `write_file` to save specs to `docs/specs/`
- Use `read_file` to read existing specs and code
- Use `search_files` to find patterns across the codebase
- Use `terminal` for git and build commands
- Use `delegate_task` for parallel task execution
- Combine with `writing-plans` for the Plan phase
- Combine with `executing-plans` for the Implement phase
- Combine with `test-driven-development` for test-first implementation
- Combine with `idea-to-design` for brainstorming before specifying
