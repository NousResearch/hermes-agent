# Hermes Agent - Development Guide

Instructions for AI coding assistants and developers working on the hermes-agent codebase.

**Never give up on the right solution.**

## What Hermes Is

Hermes is a personal AI agent that runs the same agent core across a CLI, a
messaging gateway (Telegram, Discord, Slack, and ~20 other platforms), a TUI,
and an Electron desktop app. It learns across sessions (memory + skills),
delegates to subagents, runs scheduled jobs, and drives a real terminal and
browser. It is extended primarily through **plugins and skills**, not by
growing the core.

Two properties shape almost every design decision and are the lens for
reviewing any change:

- **Per-conversation prompt caching is sacred.** A long-lived conversation
  reuses a cached prefix every turn. Anything that mutates past context,
  swaps toolsets, or rebuilds the system prompt mid-conversation invalidates
  that cache and multiplies the user's cost. We do not do it (the one
  exception is context compression).
- **The core is a narrow waist; capability lives at the edges.** Every model
  tool we add is sent on every API call, so the bar for a new *core* tool is
  high. Most new capability should arrive as a CLI command + skill, a
  service-gated tool, or a plugin — not as core surface.

## Instruction Loading Contract

This root file is the concise entry point. Detailed guidance is under
[`references/agent-guide/`](references/agent-guide/README.md) and remains normative.

Before editing:

1. Read [Contribution and footprint](references/agent-guide/contribution-and-footprint.md)
   for every contribution, bug premise check, design review, or capability addition.
2. Identify the subsystems and files the change touches.
3. Read every matching reference from the map below before changing those files.
4. Read any nearer `AGENTS.md` in a subdirectory; narrower instructions add to this file.

Do not treat linked references as optional background. If guidance conflicts, the
universal invariants in this root file win; otherwise all applicable guidance applies.

## Reference Map

| When you touch... | Read first |
| --- | --- |
| Contribution scope, bug rationale, new capability, automated triage | [Contribution and footprint](references/agent-guide/contribution-and-footprint.md) |
| Environment, repository layout, TypeScript organization, dependency chain | [Development and project structure](references/agent-guide/development-and-project-structure.md) |
| `run_agent.py`, CLI, slash commands, TUI, dashboard chat, desktop chat | [Agent, CLI, TUI, and desktop architecture](references/agent-guide/agent-cli-and-tui-architecture.md) |
| Tools, dependencies, configuration, cwd behavior, skins | [Tools, configuration, dependencies, and themes](references/agent-guide/tools-config-and-themes.md) |
| Plugins, memory/model providers, skills, skill review | [Plugins and skills](references/agent-guide/plugins-and-skills.md) |
| Toolsets, delegation, curator, cron, kanban | [Toolsets and durable systems](references/agent-guide/toolsets-and-durable-systems.md) |
| Prompt caching, background notifications, profiles, known traps | [Policies, profiles, and pitfalls](references/agent-guide/policies-profiles-and-pitfalls.md) |
| Any test or verification work | [Testing](references/agent-guide/testing.md) |

## Core Contribution Lens

- Be expansive at product edges and conservative at the core-agent/model-tool waist.
- Verify a reported premise on current `main`, trace the exact runtime path, and read the
  original design intent before calling intentional isolation or omission a bug.
- Prefer the smallest permanent footprint: extend existing code, then CLI + skill, then
  service-gated tool, plugin, MCP server, and only lastly a new core tool.
- Extend shared infrastructure instead of duplicating managers, hooks, or category-specific
  integrations. Three competing integrations usually call for an interface and orchestrator.
- Preserve contributor authorship when salvaging work; build on it rather than silently
  reimplementing it.
- Test behavior contracts and relationships, not snapshots of changing catalogs, versions,
  counts, source text, or implementation formatting.
- Use real imports and realistic paths against a temporary `HERMES_HOME` for resolution,
  propagation, security, backend, filesystem, and network boundaries. Mocks alone are not E2E proof.

## Non-Negotiable Invariants

- **Prompt caching is sacred.** Do not mutate past context, swap toolsets, reload memory, or
  rebuild the system prompt during a conversation. Context compression is the sole normal exception.
- **Message-role alternation is strict.** Never inject two consecutive messages of the same
  role or a synthetic user message mid-loop.
- **The model tool schema is costly.** New core tools require a high bar; local or niche
  capability belongs in skills, plugins, service-gated tools, or MCP.
- **`config.yaml` owns behavior; `.env` owns secrets only.** Do not add user-facing
  `HERMES_*` variables for timeouts, thresholds, flags, paths, or display preferences.
- **Profiles are isolated.** Use `get_hermes_home()` for state paths and
  `display_hermes_home()` for user-facing paths; never hardcode `~/.hermes` in code.
- **Plugins do not patch core files.** Expand a generic hook/ABC/context surface instead.
  New third-party-product integrations and new memory backends ship as standalone plugins.
- **Tool schemas must not promise unavailable cross-tool calls.** Add conditional guidance
  only while assembling the effective tool definitions.
- **Dependencies are bounded.** PyPI dependencies need ceilings; Git URLs and GitHub Actions
  use commit SHAs; regenerate the lock file when dependency metadata changes.
- **Tests never touch a user's real Hermes home.** Use the repository test wrapper and its
  isolated environment.

## Working Method

1. Inspect the current branch, worktree, nearby instructions, and relevant history.
2. Reproduce or trace the real behavior before editing.
3. Choose the narrowest solution that preserves the original feature intent.
4. Keep route roots and god-files thin; extract focused modules when a coherent cluster has grown.
5. Add behavior-focused tests through public or dependency-injected seams.
6. Run the smallest targeted verification first, then the broader relevant suite.
7. Inspect the final diff for unrelated changes, stale-branch reversions, secrets, and cache hazards.

## Quick Paths and Commands

- User settings: `~/.hermes/config.yaml`
- Secrets: `~/.hermes/.env`
- Logs: `~/.hermes/logs/`
- Core test entry point: `scripts/run_tests.sh`
- TUI checks: run the relevant `npm` commands from `ui-tui/`
- Desktop-specific rules: `apps/desktop/AGENTS.md`

Use the filesystem as the canonical project map; the detailed map is in
[Development and project structure](references/agent-guide/development-and-project-structure.md).

## Document Size and Reference Policy

Keep `AGENTS.md`, `SOUL.md`, `USER.md`, and other core instruction documents at
**250 lines or fewer**. When a document would exceed that limit:

1. Create a focused document in a `references/` directory near the source document.
2. Move topic-specific detail there instead of deleting or over-compressing it.
3. Leave an explicit link and read-when routing rule in the original document.
4. Keep each new reference focused and at 250 lines or fewer where practical; split again
   instead of creating another monolith.
5. Avoid duplicated normative text. Keep universal invariants here and detail in one reference.

The reference index is [`references/agent-guide/README.md`](references/agent-guide/README.md).
