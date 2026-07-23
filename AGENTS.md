# Hermes Agent Development Guide

Instructions for AI coding assistants and developers working on Hermes Agent.

**Never give up on the right solution.**

## What Hermes is

Hermes runs one agent core across the CLI, messaging gateway, TUI, dashboard,
desktop app, and editor integrations. It learns through memory and skills,
delegates to subagents, runs scheduled work, and operates real tools.

Two principles govern the codebase:

1. **Prompt caching is sacred.** A conversation reuses a stable prefix. Do not
   mutate past context, swap toolsets, reload memories, or rebuild the system
   prompt mid-conversation. Context compression is the only routine exception.
2. **The core is a narrow waist.** Every permanent model tool is sent on every
   API call. Product capability should grow at the edges through existing
   surfaces, commands, skills, gated tools, plugins, providers, or MCP.

## How to work

- Reproduce reported behavior on current code before calling it a bug.
- Trace the exact runtime path and original design intent before changing it.
- Fix the whole bug class, including sibling call paths.
- Preserve prompt caching, message-role alternation, profile isolation, and
  stable system prompts.
- Prefer real end-to-end validation over mocks when configuration, discovery,
  serialization, security, files, or networks are involved.
- Keep feature changes focused. Large mechanical refactors are fine when the
  declared task is the refactor.
- Preserve contributor authorship when salvaging external work.
- Ask when intent is genuinely ambiguous; do not “fix” deliberate isolation or
  resurrect a superseded design.

## Contribution boundary

Hermes expands aggressively at its edges: platform adapters, providers,
desktop/TUI/dashboard features, models, and channels are welcome. Restraint is
primarily about the core agent and permanent tool schema.

### Preferred capability ladder

Choose the highest applicable rung:

1. extend existing code;
2. CLI command plus skill;
3. prerequisite-gated tool;
4. plugin;
5. MCP server and catalog entry;
6. new core tool only when broadly fundamental and not achievable above.

When several implementations share a category, design an abstract interface
and orchestrator rather than merging unrelated special cases.

### Reject or redesign

- speculative hooks without a concrete consumer;
- non-secret behavior configured through new `HERMES_*` environment variables;
- core tools that duplicate terminal, file, command, skill, plugin, or MCP
  capabilities;
- lazy/paginated readers for mandatory instructional content;
- security mitigations that destroy the protected feature;
- unapproved telemetry or attribution;
- cache-breaking mid-conversation mutation;
- dead code wired into production without real-path validation;
- plugin-specific branches in core;
- in-tree third-party product integrations that should be standalone plugins;
- change-detector tests and source-text tests.

Before closing or rejecting external work, distinguish a wrong premise from a
valid contribution that simply needs human product judgment. Automated review
must not convert “we do not want this” into “cannot reproduce.”

## Repository routing

The filesystem is canonical; counts and exact lists change. Start here:

| Area | Primary paths | Read before editing |
|---|---|---|
| Agent loop and tools | `run_agent.py`, `model_tools.py`, `toolsets.py`, `agent/`, `tools/` | `docs/development/component-guide.md` |
| CLI and configuration | `cli.py`, `hermes_cli/` | `hermes_cli/AGENTS.md` |
| Gateway and platforms | `gateway/`, `plugins/platforms/` | this guide plus `docs/development/component-guide.md` |
| TUI and dashboard chat | `ui-tui/`, `tui_gateway/`, `web/` | `ui-tui/AGENTS.md` |
| Desktop app | `apps/desktop/` | `apps/desktop/AGENTS.md` |
| Plugins/providers | `plugins/` | `plugins/AGENTS.md` |
| Default skills | `skills/` | `skills/AGENTS.md` |
| Optional skills | `optional-skills/` | `optional-skills/AGENTS.md`, then `skills/AGENTS.md` |
| Tests | `tests/` | `tests/AGENTS.md` |
| Cron and Kanban | `cron/`, `plugins/kanban/` | component guide and user feature docs |

Nested `AGENTS.md` files contain scoped implementation detail. Do not copy
their full contents back into this root file.

## Development environment

Prefer `.venv`; fall back to `venv` or the managed Hermes environment:

```bash
source .venv/bin/activate
scripts/run_tests.sh tests/path/to/test_file.py
```

Always run Python tests through `scripts/run_tests.sh`, never direct `pytest`.
Read `tests/AGENTS.md` for isolation and test-design rules.

## Non-negotiable engineering rules

### Prompt and conversation invariants

- The system prompt must remain byte-stable for a conversation.
- Never inject consecutive same-role messages.
- Commands that change prompt state default to deferred activation on the next
  session. An explicit `--now` path may invalidate immediately.
- Compression is the only normal operation allowed to rewrite context.
- Background or cron results must enter through their supported queues/frames,
  not by corrupting the active transcript.

### Configuration

- `config.yaml` owns timeouts, thresholds, flags, paths, display preferences,
  and all other non-secret behavior.
- `.env` owns credentials only: API keys, tokens, and passwords.
- Add defaults in `hermes_cli/config.py`.
- Bump `_config_version` only for an active migration or structural
  transformation, not a simple deep-merged key addition.
- Verify every consumer: classic CLI, subcommands/setup, and gateway have
  distinct loading paths.

### Profile-safe state

- Use `get_hermes_home()` for files, caches, logs, checkpoints, and state.
- Use `display_hermes_home()` for user-visible paths.
- Never hardcode `~/.hermes` or `Path.home() / ".hermes"` for active-profile
  state.
- Module-level constants may cache `get_hermes_home()` because profile override
  happens before imports.
- Profile discovery is intentionally HOME-anchored so every active profile can
  enumerate its siblings.
- Adapters using unique credentials should acquire and release scoped token
  locks to prevent two profiles from using one credential simultaneously.

### Tools and extensions

- Tool registration occurs through `tools/registry.py`.
- Discovery imports a tool but does not expose it; built-in tools must also
  belong to a toolset in `toolsets.py`.
- Handlers return JSON strings.
- Optional tools use requirement checks and disappear when unavailable.
- Tool schema descriptions cannot assume another toolset is present.
- Plugins stay within plugin directories and generic extension surfaces.
- State owned by a tool or plugin is profile-local.

### Dependencies

All dependencies have upper bounds:

| Source | Required form |
|---|---|
| PyPI package | `>=floor,<next-major` |
| Pre-1.0 PyPI package | bounded minor range |
| Git URL | full commit SHA |
| GitHub Action | commit SHA plus version comment |
| CI-only pip dependency | exact version |

After dependency changes, regenerate the lockfile and run the relevant supply
chain checks. Never commit a bare unbounded `>=` requirement.

## TypeScript conventions

These apply across the TUI, desktop, dashboard, and future TypeScript packages:

- Shared or distant state belongs in small, feature-owned stores.
- Rendering components subscribe; non-rendering actions read stores directly.
- Keep route roots thin and hooks single-purpose.
- Avoid prop-drilling through multiple layers.
- Keep persistence beside the state owner.
- Prefer interfaces for public object shapes.
- Extend primitive/component props with `React.ComponentProps`, `Pick`, or
  `Omit`.
- Prefer table-driven maps over long condition ladders.
- Mark intentionally ignored promises with `void`.

More specific UI rules live in nested guides.

## Critical pitfalls

### Gateway control messages pass two guards

Messages encounter both the base adapter's active-session queue and the gateway
runner's active-agent interception. Commands that must work while an agent is
blocked—stop, approval, denial, queue, status, or similar controls—must bypass
both and dispatch inline.

### Do not add `simple_term_menu`

It has rendering defects in common terminals. New interactive menus use
`hermes_cli/curses_ui.py`.

### Avoid ANSI erase-to-EOL in live output

Do not use `\033[K` under `prompt_toolkit` patching. Pad the remainder of the
line with spaces.

### Process-global resolved tool names are temporarily mutable

`model_tools.py` keeps `_last_resolved_tool_names` process-wide. Delegation
saves/restores it around child execution. New readers must tolerate that
temporary scope.

### Schema descriptions cannot hardcode cross-tool advice

The referenced tool may be disabled. Add conditional cross-tool guidance while
assembling definitions only when both surfaces are available.

### Stale branch squash merges can revert unrelated fixes

Update the branch before squashing and inspect the final merge diff for
unexpected deletions or old versions of unrelated files.

### Dead code needs end-to-end proof

Unused modules may encode abandoned assumptions. Before connecting one to a
live path, validate the real resolution/import/configuration chain against a
temporary home.

## Background systems

- `delegate_task` children are isolated and concurrency-limited. Background
  delegation is process-local and does not survive restart.
- Cron is the durable scheduler. It uses its own sessions and delivery frames;
  do not mirror cron output into a gateway transcript.
- Kanban is the durable multi-agent work queue. Board is the hard isolation
  boundary; tenant is only a namespace within a board.
- Curator may maintain only agent-created skills, never bundled or
  hub-installed skills. It archives rather than deletes and never transitions
  pinned skills automatically.
- Gateway background-process notifications are controlled through
  `display.background_process_notifications`.

Detailed component ownership lives in `docs/development/component-guide.md`;
user-facing behavior belongs in the website feature documentation.

## Testing requirements

The short version:

- use `scripts/run_tests.sh`;
- isolate `HERMES_HOME`;
- make no live network calls in unit tests;
- test behavior, not source text;
- assert invariants, not current catalog snapshots or counts;
- put JavaScript/TypeScript behavior in the JS test suite;
- exercise real integration paths for discovery, configuration, security,
  serialization, and I/O.

`tests/AGENTS.md` is authoritative for test work.

## Documentation maintenance

Keep this root guide limited to rules relevant across most changes. Put:

- subsystem rules in the nearest nested `AGENTS.md`;
- architecture and routing detail in `docs/development/`;
- user procedures in `website/docs/`;
- skill procedures in `SKILL.md` plus `references/`;
- exhaustive command or field catalogs next to the owning subsystem.

When changing behavior, update the canonical scoped document instead of adding a
second explanation here.
