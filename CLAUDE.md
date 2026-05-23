# Hermes Agent — Claude Code Guide

This file is the Claude Code entry point for this repository. Path-scoped rule
files in `.cursor/rules/` provide domain-specific context automatically based on
which files you're editing. The canonical agent/developer guide is `AGENTS.md`.

## MANDATORY: Fetch Hermes Context First

**Before doing ANY work in this repo, load fleet context via Hermes MCP.**

Canonical reference: `website/docs/user-guide/features/cursor-hermes.md`
Cursor rule: `.cursor/rules/hermes-fleet.mdc`

### Session Start Checklist

Prefer `fleet_context_snapshot()` when available; otherwise run in order:

1. `skills_list()` — discover agents and their SOUL.md files
2. `agents_list(include_heartbeat=true)` — registry + live heartbeat
3. `learnings_read()` — HOT-tier persistent memory (.learnings/memory.md)
4. `knowledge_read(artifact="latest_state")` — current fleet state
5. `knowledge_read(artifact="held_spec_ledger")` — active constraints

### Before Modifying a Specific Agent

1. `skills_read(name="<agent>")` — read the SOUL.md behavioral contract
2. `agents_get(name="<agent>")` — full detail (registry, heartbeat, files)
3. `knowledge_read(artifact="contradiction_ledger")` — known conflicts

### Before Modifying Pipeline or Infrastructure

1. `artifacts_list()` — browse operational outputs
2. `knowledge_read(artifact="operator_brief")` — current production status
3. `learnings_read(file="projects/")` — project-specific memory

**Read first. Then code.**

---

## Environment

```bash
source .venv/bin/activate  # or: source venv/bin/activate
```

## Test Commands (always use the wrapper)

```bash
scripts/run_tests.sh                              # Full suite
scripts/run_tests.sh tests/gateway/               # Subsystem
scripts/run_tests.sh tests/tools/test_delegate.py::TestBlockedTools  # Specific
.venv/bin/ruff check .                            # Lint
```

Do not call `pytest` directly — the wrapper normalizes credentials, HOME,
timezone, locale, and worker count.

---

## Project Invariants

- Profile-aware state paths must use `get_hermes_home()` from `hermes_constants`;
  user-facing path text should use `display_hermes_home()`.
- Tests must not write to a real `~/.hermes/` — use fixtures with `HERMES_HOME`.
- Prompt caching must not be broken mid-conversation. Slash commands that alter
  tools, skills, memory, or system prompt should defer invalidation unless an
  explicit `--now` flow exists.
- Built-in tools require both registration in `tools/*.py` and exposure through
  `toolsets.py`.
- Plugin capabilities should use generic plugin hooks; do not hardcode
  plugin-specific logic into core files.

---

## Governance Constraints (ACTIVE)

1. **Read-only MCP access.** Skills MCP tools are read-only. Do not write to
   `.learnings/`, `artifacts/`, `agents/AGENT_REGISTRY.json`, or knowledge
   layer files through MCP.

2. **No Town-to-Hermes feedback automation.** The Town-Hermes Feedback Protocol
   is FROZEN until after h20d. Do not implement automated memory sync,
   contradiction-ledger routing, or `.learnings/` write paths.

3. **Held specifications.** Check `held_spec_ledger` before making changes. If a
   spec is held, do not modify the constrained area without operator approval.

4. **Authority model.** Respect the 5-tier model. Most agents are `observe_only`
   or `observe_and_propose`. Only `crt_resolution_watcher` has `mutate_data`.
   No agent has `mutate_config` — that is operator-only.

5. **Lane constraints.** Lane A (deterministic): no LLM. Lane B: LLM on anomaly
   only. Lane C: manual-only, no cron.

---

## High-Value Files

| File | Purpose |
|------|---------|
| `run_agent.py` | AIAgent, conversation loop, interrupts, compression |
| `model_tools.py` | Tool discovery, schema filtering, function dispatch |
| `toolsets.py` | Toolset definitions and platform bundles |
| `cli.py` | Classic CLI and slash-command dispatch |
| `gateway/run.py` | Messaging gateway runner |
| `hermes_cli/config.py` | Default config and config migration |
| `mcp_serve.py` | MCP server entry point (stdio) |
| `hermes_skills_mcp.py` | 7 read-only skills/knowledge MCP tools |
| `agents/AGENT_REGISTRY.json` | Authoritative agent fleet manifest |
| `AGENTS.md` | Full agent fleet documentation |

---

## Anti-Patterns (Do NOT)

1. Do not hardcode agent counts — source from `AGENT_REGISTRY.json`
2. Do not call `pytest` directly — use `scripts/run_tests.sh`
3. Do not write to `.learnings/` via automation (feedback protocol FROZEN)
4. Do not assign `mutate_config` authority to any agent
5. Do not create Lane A agents that import LLM modules
6. Do not reduce API timeout below 2400s (Together cold start spikes)
7. Do not modify production hashes without a `governance/HASH_ROTATIONS.md` entry
8. Do not bypass held specifications regardless of change size
9. Do not commit host-specific paths (use `${workspaceFolder}` in configs)
10. Do not revert unrelated user changes in the working tree

---

## Path-Scoped Rules (auto-loaded by Cursor)

Domain-specific context loads automatically when you edit matching files:

| Rule File | Domain | Loads When Editing |
|-----------|--------|-------------------|
| `hermes-fleet.mdc` | MCP bootstrap, source-of-truth hierarchy | Always |
| `agent-lifecycle.mdc` | Agent SOUL.md, registry, heartbeats | `agents/**` |
| `pipeline-production.mdc` | Daily pipeline, CI, cron | `tools/run_daily*.py`, `cron/**`, `.github/**` |
| `mcp-development.mdc` | Adding/testing MCP tools | `mcp_serve.py`, `hermes_skills_mcp.py` |
| `governance-specs.mdc` | Held specs, promotion, architecture | `governance/**`, `production_data/**` |
| `signal-agents.mdc` | Herald, Bellringer, Intraday, CRT | `agents/herald/**`, signal agent dirs |
| `llm-config.mdc` | Provider routing, inference params | `hermes_cli/config.py`, `run_agent.py` |
| `knowledge-learnings.mdc` | Knowledge layer, .learnings/ memory | `artifacts/**`, `.learnings/**` |
| `testing-quality.mdc` | Test conventions, fixtures | `tests/**`, `scripts/run_tests.sh` |

For operational state (13F cycle, spec status, freeze dates), always fetch live
via MCP (`knowledge_read`) rather than relying on stale file content.

---

## Start Here

- Work from the current git branch unless the user asks you to switch.
- Prefer the repo's existing patterns and helper APIs over new abstractions.
- Do not revert unrelated user changes in the working tree.
- Keep edits scoped to the request and the affected subsystem.
