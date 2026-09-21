---
name: context-routing
description: "Multi-agent context routing with all-context.md entry points. Router-first discipline: read the router, then the routed file, never skip."
version: 1.0.0
author: KENSEI (extracted from withkynam/vibecode-pro-max-kit)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [context, routing, multi-agent, handoff, documentation]
    related_skills: [kanban-orchestrator, kanban-worker, subagent-driven-development]
---

# Context Routing

## Purpose

Prevent context bloat and stale information by enforcing a router-first discipline. Large projects use `all-*.md` entrypoints as routers, not as the full knowledge. Agents MUST follow the routing tables to read the most relevant deeper file(s) before proposing or executing operational steps.

Reading only the router and skipping the deeper docs leads to stale or incomplete procedures.

## Core Principle

**Router → Routed → Action.**

1. Read the router file first (`all-context.md`, `all-development-protocols.md`, `all-tests.md`)
2. Follow its routing table to the smallest relevant deeper file
3. Only then act

## Router File Types

| Router | What it routes to | When to read |
|--------|-------------------|--------------|
| `all-context.md` | Project-specific coding preferences, architecture, aliases, env vars | Before any substantial planning or implementation |
| `all-development-protocols.md` | Orchestration, implementation standards, plan lifecycle, phase programs | Before delegating or following workflows |
| `all-tests.md` | Test routing, verification commands, harness expectations | Before testing, verification, or debugging |
| `all-plans.md` | Active plan inventory, feature folders, phase programs | Before resuming work or creating new plans |

## Routing Discipline

### When starting work on an existing project:

1. **Read `all-context.md`** first
2. **Choose the smallest relevant routed context file** — not the biggest, not all of them
3. If `Feature:` is present in the handoff, inspect `features/{feature}/active/`, `reports/`, and `references/` before falling back to general folders
4. When research touches testing, read `all-tests.md` before deeper test docs
5. When planning, read `all-development-protocols.md` before following workflows

### When receiving a handoff:

Treat `Work context`, `Feature`, `Reports`, or `Plans` as authoritative scope hints. Do not ignore them and fall back to general discovery.

### When observing ongoing work:

Treat direct `*_PLAN_*.md`, legacy `PLAN.md`, legacy `plan.md`, and `phase-*` files as valid active-plan compatibility shapes. Resume from the last known state rather than starting fresh.

## Context Validation

After routing, validate broad understanding against the router:

- Environment variables match documented requirements
- Import paths use documented aliases (e.g., `~/*` for Next.js app)
- Existing services follow documented conventions
- TypeScript export maps are current
- API procedures align with documented surface
- Product name and branding are consistent

If context files appear outdated, unindexed, or contradicted by codebase, flag the drift instead of silently overriding.

## Anti-Patterns

| Bad | Good |
|-----|------|
| Read `all-context.md` and act immediately | Read `all-context.md`, follow its routing table, read the deeper file, then act |
| Load every context file "to be safe" | Load only the smallest relevant routed file |
| Ignore `Feature:` hints and search the whole repo | Use `Feature:` to scope to the feature folder first |
| Assume the router is the full knowledge | The router is an index; the deeper file is the content |

## KENSEI Context Handoff Protocol Integration

KENSEI's `context-handoff-protocol.md` defines 5 handoff types (A through E) with structured payloads. Context routing complements this:

- **Type D (Worker handoff)** — The worker should read any `all-context.md` referenced in the task body, then route to the smallest relevant file
- **Type E (Completion handoff)** — The completing agent should note which context files were consulted so the receiver knows what's current

## Gap Analysis: vibecode vs KENSEI

| vibecode feature | KENSEI equivalent | Gap |
|------------------|-------------------|-----|
| `all-context.md` router | `CLAUDE.md` / `AGENTS.md` / task body | KENSEI tasks often embed context directly; no single router file exists per-project |
| `process/features/{feature}/` folders | Kanban task workspace | KENSEI workspaces are `scratch` or `dir`, not feature-scoped by default |
| `reports/` surface per feature | `kanban_comment` + `kanban_complete` metadata | KENSEI comments are durable but not file-based; no explicit `reports/` folder convention |
| `harness/` evidence pack | `kanban_complete` metadata + workspace artifacts | KENSEI metadata is JSON, not a file tree; no `risk-gate.json` convention exists |
| Context drift flagging (`vc-generate-context`, `vc-audit-context`) | Manual skill updates | No automated drift detection for Hermes skills |

**Recommendation**: Adopt the router-first discipline within KENSEI skill documentation. For project-specific work, encourage a project-local `AGENTS.md` or `CLAUDE.md` that acts as the router, with routed files under `docs/` or `.hermes/`. The Kanban system already provides scoped workspaces — extend this with a lightweight `context/` or `docs/` folder inside workspaces when tasks are long-running.

## Four-Bucket Strategy

From context engineering principles:

1. **Write** — Save context externally (scratchpads, files, kanban comments)
2. **Select** — Pull only relevant context (retrieval, filtering, routing)
3. **Compress** — Reduce tokens while preserving info (summarization)
4. **Isolate** — Split across sub-agents (partitioning, delegation)

Apply these buckets in order: write what you discover, select only what the next agent needs, compress if approaching context limits, isolate if parallel work is possible.
