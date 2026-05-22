# Hermes Agent - Claude Code Guide

This file is the Claude Code entry point for this repository. The canonical,
more complete agent/developer guide is `AGENTS.md`; use it as the source of
truth when details differ.

## MANDATORY: Fetch Hermes Context First

**Before doing ANY work in this repo, load fleet context via Hermes MCP.** Do not
rely on cached or stale context from previous sessions.

**Canonical reference:** [Cursor & Hermes](website/docs/user-guide/features/cursor-hermes.md)
(source-of-truth hierarchy, skills-only vs gateway mode, `HERMES_AGENTS_DIR`).

**Cursor rule:** `.cursor/rules/hermes-fleet.mdc` mirrors this checklist.

### Session Start Checklist

Prefer `fleet_context_snapshot()` when available; otherwise run:

1. **`skills_list()`** — discover all custom agents and their SOUL.md files.
   This tells you what agents exist, which have skills documents, and where
   they live on disk.

2. **`agents_list(include_heartbeat=true)`** — load the full agent registry
   with lane assignments, authority levels, status, and live heartbeat data.
   This is how you know which agents are healthy vs. stale.

3. **`learnings_read()`** — read the HOT-tier persistent memory
   (`.learnings/memory.md`). This contains corrections, patterns, and
   operational knowledge the fleet has accumulated. Respect the 100-line cap.

4. **`knowledge_read(artifact="latest_state")`** — load the current knowledge
   layer state. This is the fleet's shared understanding of production status,
   anomalies, and operational context.

5. **`knowledge_read(artifact="held_spec_ledger")`** — check for any held
   specifications that constrain what changes are allowed.

### Before Modifying a Specific Agent

When the task involves a specific agent (e.g. "fix herald", "update bellringer"):

1. **`skills_read(name="<agent_name>")`** — read the agent's SOUL.md. This
   defines the agent's identity, purpose, constraints, and behavioral rules.
   You must understand the SOUL.md before changing anything about the agent.

2. **`agents_get(name="<agent_name>")`** — get the full agent detail including
   registry entry, heartbeat status, and file listing.

3. **`knowledge_read(artifact="contradiction_ledger")`** — check for known
   contradictions or discrepancies that may affect this agent.

### Before Modifying Pipeline or Infrastructure

When the task involves pipeline code, cron, CI, or infrastructure:

1. **`artifacts_list()`** — browse the artifacts directory to understand what
   operational outputs exist and their structure.

2. **`knowledge_read(artifact="operator_brief")`** — read the latest daily
   operator brief for current production status and known issues.

3. **`learnings_read(file="projects/")`** — list project-specific memory files
   for relevant namespace context.

### Why This Matters

The Hermes fleet has 30 agents with custom SOUL.md files, a 5-tier authority
model, 3 execution lanes, and a 4-layer monitoring stack. Changes that ignore
this context risk:

- Breaking agent authority boundaries
- Violating held specifications
- Contradicting fleet learnings
- Introducing regressions into a production pipeline that runs daily

**Read first. Then code.**

---

## Start Here

- Work from the current git branch unless the user asks you to switch.
- Prefer the repo's existing patterns and helper APIs over new abstractions.
- Do not revert unrelated user changes in the working tree.
- Keep edits scoped to the request and the affected subsystem.

## Environment

```bash
source .venv/bin/activate  # or: source venv/bin/activate
```

`scripts/run_tests.sh` is the required test wrapper. It probes `.venv`, `venv`,
and the shared Hermes checkout venv, then runs pytest with CI-like environment
settings.

## Test Commands

```bash
scripts/run_tests.sh
scripts/run_tests.sh tests/gateway/
scripts/run_tests.sh tests/tools/test_delegate.py::TestBlockedTools
.venv/bin/ruff check .
```

Do not call `pytest` directly unless there is no alternative; the wrapper
normalizes credentials, HOME, timezone, locale, and worker count.

## Important Project Invariants

- Profile-aware state paths must use `get_hermes_home()` from
  `hermes_constants`; user-facing path text should use `display_hermes_home()`.
- Tests must not write to a real `~/.hermes/`; use the existing fixtures and set
  `HERMES_HOME` when mocking home directories.
- Prompt caching must not be broken mid-conversation. Slash commands that alter
  tools, skills, memory, or system prompt state should defer invalidation unless
  an explicit `--now` flow exists.
- Built-in tools require both registration in `tools/*.py` and exposure through
  `toolsets.py`.
- Plugin capabilities should use generic plugin hooks/surfaces; do not hardcode
  plugin-specific logic into core files.

## High-Value Files

- `run_agent.py` - `AIAgent`, conversation loop, interrupts, compression.
- `model_tools.py` - tool discovery, schema filtering, function dispatch.
- `toolsets.py` - toolset definitions and platform bundles.
- `cli.py` - classic CLI and slash-command dispatch.
- `gateway/run.py` - messaging gateway runner.
- `hermes_cli/config.py` - default config and config migration.
- `tools/` - built-in tool implementations.
- `plugins/` - plugin systems and bundled plugins.
- `tests/` - pytest suite.

## MCP Server & Skills Integration

The Hermes MCP server (`mcp_serve.py`) runs as a stdio MCP server that Cursor
and Claude Code connect to automatically via `.cursor/mcp.json`. It provides
two tool surfaces:

### Messaging Tools (10 tools)

Conversations, messages, events, approvals across connected platforms:
`conversations_list`, `conversation_get`, `messages_read`,
`attachments_fetch`, `events_poll`, `events_wait`, `messages_send`,
`channels_list`, `permissions_list_open`, `permissions_respond`

### Skills & Knowledge Tools (hermes_skills_mcp.py, 7 tools)

Read-only access to the custom Hermes agent fleet, skills, knowledge layer,
and persistent memory.

| Tool | Purpose |
|------|---------|
| `skills_list` | List all agent SOUL.md files and repo skills |
| `skills_read` | Read a specific SOUL.md or skill document |
| `agents_list` | List agents with registry data and optional heartbeat |
| `agents_get` | Full agent detail: registry, SOUL.md, heartbeat, files |
| `knowledge_read` | Read knowledge layer artifacts (latest_state, ledgers) |
| `learnings_read` | Read .learnings/ memory files (HOT/WARM/COLD tiers) |
| `artifacts_list` | Browse the artifacts/ directory tree |

**Key paths** (resolved via HERMES_HOME and HERMES_REPO):
- `agents/` - Custom agent directories, each with SOUL.md, HEARTBEAT.md
- `agents/AGENT_REGISTRY.json` - Authoritative agent fleet manifest
- `artifacts/ops/knowledge_layer/` - Knowledge layer state files
- `artifacts/ops/held_spec_ledger/` - Held specification tracking
- `.learnings/memory.md` - HOT-tier persistent memory (100-line cap)
- `skills/` - Upstream OpenClaw skill categories

### Architecture Notes

- `hermes_skills_mcp.py` is a standalone module imported by `mcp_serve.py`
- All tools are **read-only** — no mutation of skills, registry, or artifacts
- Gracefully degrades: if `hermes_skills_mcp` import fails, the messaging
  tools still work (logged at DEBUG level)
- Path resolution uses HERMES_HOME/HERMES_REPO env vars, same as the rest
  of the codebase

## Governance Constraints

**These constraints are active and must not be violated:**

1. **Read-only MCP access.** The skills MCP tools expose Hermes data for
   reading only. Do not attempt to write to `.learnings/`, `artifacts/`,
   `agents/AGENT_REGISTRY.json`, or any knowledge layer file through MCP
   or by circumventing the read-only surface.

2. **No Town-to-Hermes feedback automation.** The Town-Hermes Feedback
   Protocol is FROZEN until after h20d (May 26, 2026). Do not implement
   automated memory sync, contradiction-ledger routing, or `.learnings/`
   write paths. This is a governance decision, not a technical limitation.

3. **Held specifications.** Check the held_spec_ledger before making changes.
   If a specification is held, do not modify the constrained area without
   explicit operator approval.

4. **Authority model.** Respect the 5-tier authority model when modifying
   agent configurations. Most agents are `observe_only` or
   `observe_and_propose`. Only `crt_resolution_watcher` has `mutate_data`.
   No agent has `mutate_config` — that is operator-only.

5. **Lane constraints.** Lane A agents (deterministic) must not depend on
   LLM gateway tokens. Lane B agents use LLM on anomaly only. Lane C
   agents are manual-only, no cron.

## Recent CI/PR Notes

This branch contains audit fixes around:

- subagent blocked-tool enforcement,
- `AIAgent.close()` cleanup of shared terminal/background resources,
- Google Chat plugin platform registration and Pub/Sub handoff,
- setup-provider config resync,
- gateway runtime env reload authority,
- concurrent interrupt test scaffolding.

When touching these areas, rerun the focused tests listed in the PR body before
committing.
