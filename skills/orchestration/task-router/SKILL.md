---
name: task-router
description: Route a task to the right Hermes profile (engineer, homelab, debugger, mail, etc.) when it needs a specialized toolset — without standing up a Kanban board. Trigger when a task clearly belongs to a specialist and the current profile lacks the right tools.
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [orchestration, routing, profiles, delegation]
    related_skills: [model-selection-router, orchestration-patterns, cost-guard]
---

# Task Router

> Match the task to the profile whose toolset actually fits it. The right profile means the right tools loaded and no tokens wasted on irrelevant ones.

## Trigger

Apply when a task needs a **specialized agent** — i.e. the current profile doesn't have the tools the task requires (no terminal for a code task, no Proxmox for an infra task, no Fastmail toolset for email, etc.). This is lighter-weight than the Kanban orchestrator: no board, no dependency graph — just send the work to the right profile.

## Profile routing table

These are the profiles configured on this install. Match the task to a row:

| Task type | Profile | Why (toolset) |
|-----------|---------|---------------|
| Code / implementation | `engineer` | terminal + file + code_execution + skills |
| Infra / homelab | `homelab` or `ops` | proxmox + cluster-ops + prometheus |
| Email / calendar | `mail` or `calendar` | Fastmail toolsets |
| Research / web search | `search` or `web_researcher` | tavily + exa + brave |
| Pure reasoning | `think` | sequential-thinking MCP |
| GitHub / git ops | `version-control` | github MCP + file |
| KB lookup / update | `kb` | knowledge MCP |
| Debug a prod issue | `debugger` | terminal + prometheus + cluster-ops |
| Document handling | `documents` | document toolset |
| File operations | `files` | file toolset |
| Building an MCP server | `mcp-builder` | MCP scaffolding toolset |
| Persisting a learning | `memory` | memory toolset |

When two profiles fit (`homelab` vs `ops`, `search` vs `web_researcher`), pick by emphasis: `ops`/`debugger` for change/incident work, `homelab` for steady-state cluster queries; `web_researcher` for deep multi-source digs, `search` for a quick fact.

## How to invoke

- **Spawn a child on a profile:** `delegate_task` with `profile: <name>` — the child runs with that profile's toolset and the cheap `delegation.model` by default.
- **Switch the current session:** `/profile <name>` in-session to re-home yourself onto the specialist toolset.

If no row fits the task, ask the user which profile to use rather than guessing — an unknown profile name is dropped silently.

## When NOT to route

Stay where you are and just do the work when:

- It's a **single-turn lookup** the current profile can already answer.
- The task is **under ~5 tool calls** and your current toolset covers it.
- The **answer is already in context**.
- The task **belongs to the current profile** — e.g. you're already `engineer` and the task is code. Re-routing to yourself is pure overhead.

## Anti-patterns

- **Routing to a profile you also already are.** If you're `engineer` and the task is a small code edit, just do it.
- **Inventing profile names.** Only the names in the table exist on this install; an unknown assignee is silently dropped and the work never runs.
- **Spinning up a Kanban board for a one-lane task.** Routing ≠ orchestration. Use `delegate_task` to a single profile here; reach for the board only when multiple specialists, persistence, or parallel fan-out are needed (see `orchestration-patterns`).
- **Routing tiny work and paying the spawn tax.** A child agent costs more than a same-profile turn — see `cost-guard` for the multiplier.
