---
name: blueprint
description: 'Turn a one-line objective into a step-by-step construction plan for multi-session, multi-agent engineering projects. Each step has a self-contained context brief so a fresh agent can execute it cold.'
version: 1.0.0
author: Hermes Agent (adapted from community)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [planning, multi-session, architecture, delegation]
    related_skills: [writing-plans, executing-plans, idea-to-design]
---

# Blueprint — Construction Plan Generator

Turn a one-line objective into a step-by-step construction plan that any coding agent can execute cold.

## When to Use

- Breaking a large feature into multiple PRs with clear dependency order
- Planning a refactor or migration that spans multiple sessions
- Coordinating parallel workstreams across sub-agents
- Any task where context loss between sessions would cause rework

**Do not use** for tasks completable in a single PR, fewer than 3 tool calls, or when the user says "just do it."

## How It Works

Blueprint runs a 5-phase pipeline:

1. **Research** — Read project structure, existing plans, and memory files to gather context.
2. **Design** — Break the objective into one-PR-sized steps (3–12 typical). Assign dependency edges, parallel/serial ordering, and rollback strategy per step.
3. **Draft** — Write a self-contained Markdown plan file. Every step includes a context brief, task list, verification commands, and exit criteria — so a fresh agent can execute any step without reading prior steps.
4. **Review** — Adversarial review via `delegate_task` against a checklist and anti-pattern catalog. Fix all critical findings.
5. **Register** — Save the plan, update memory index, and present step count and parallelism summary.

## Plan Structure

Each step in the plan includes:

````markdown
### Step N: [Title]

**Context Brief:** [What a fresh agent needs to know — no prior steps required]

**Dependencies:** [Which steps must complete first]

**Tasks:**

- [ ] [Specific, verifiable task]
- [ ] [Specific, verifiable task]

**Verification:**

```bash
# Exact commands to verify this step is done
```
````

**Exit Criteria:** [How to know this step is complete]

```

## Examples

### Basic usage
```

/blueprint "migrate database to PostgreSQL"

```

Produces a plan with steps like:
- Step 1: Add PostgreSQL driver and connection config
- Step 2: Create migration scripts for each table
- Step 3: Update repository layer to use new driver
- Step 4: Add integration tests against PostgreSQL
- Step 5: Remove old database code and config

### Multi-agent project
```

/blueprint "extract LLM providers into a plugin system"

```

Produces a plan with dependency graph showing which steps can run in parallel.

## Hermes Integration

- Use `read_file` and `search_files` for research phase
- Use `write_file` to save plans to `docs/plans/`
- Use `delegate_task` for adversarial review phase
- Use `executing-plans` for step-by-step execution
- Use `delegate_task` with batch mode for parallel steps
- Use `memory` tool to track plan state across sessions
```
