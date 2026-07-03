# Hermes Agent - Runtime Agent Guide

Instructions for AI coding assistants working inside the `hermes-agent` codebase.

**Never give up on the right solution.** Keep changes narrow, verified, and cache-safe.

This file is auto-discovered and injected into agent context. Keep it focused on runtime-essential guidance. Contributor policy, PR review rubric, dependency policy, skill-authoring standards, and deep subsystem notes live in [`CONTRIBUTING.md`](CONTRIBUTING.md) and should be opened only when needed for contribution/review work.

## What Hermes Is

Hermes is a personal AI agent that runs the same agent core across a CLI, messaging gateway (Telegram, Discord, Slack, and ~20 other platforms), a TUI, and an Electron desktop app. It learns across sessions (memory + skills), delegates to subagents, runs scheduled jobs, and drives a real terminal and browser.

Two design properties shape most changes:

- **Per-conversation prompt caching is sacred.** Do not mutate past context, swap toolsets, or rebuild the system prompt mid-conversation except through the established compression path.
- **The core is a narrow waist; capability lives at the edges.** Prefer extending existing code, CLI commands, skills, plugins, or service-gated tools before adding new core model tools.

## Quick Contribution Rules

For the full contribution rubric, see [`CONTRIBUTING.md`](CONTRIBUTING.md). Keep these rules in mind during routine edits:

- Fix real, reproduced bugs; point to the line/path where behavior changes.
- Preserve message role alternation and prompt-cache stability.
- Do not add speculative hooks or extension points without a concrete consumer.
- Do not add user-facing `HERMES_*` env vars for non-secret config; use `config.yaml`.
- Do not add a core model tool when file/terminal, a CLI command, a skill, or a plugin is sufficient.
- Use behavior/invariant tests, not change-detector snapshots.
- For config, provider, security, remote-backend, and file/network I/O changes, prefer real-path tests with a temp `HERMES_HOME` over mocks.

## Development Environment

```bash
# Prefer .venv; fall back to venv if that is what the checkout has.
source .venv/bin/activate 2>/dev/null || source venv/bin/activate

# Run targeted tests through the project wrapper.
scripts/run_tests.sh tests/path/to/test_file.py -q

# Full suite when needed.
scripts/run_tests.sh tests -q
```

## Project Structure

```text
hermes-agent/
├── run_agent.py              # AIAgent: main conversation loop
├── model_tools.py            # Tool schema discovery/dispatch helpers
├── toolsets.py               # Toolset definitions and core tool lists
├── cli.py                    # Interactive CLI entry point / orchestration
├── hermes_state.py           # SQLite session store
├── agent/                    # Prompt building, init, memory, tools, delegation
├── hermes_cli/               # CLI subcommands, config, setup, slash registry
│   ├── commands.py           # Central slash command registry
│   ├── config.py             # DEFAULT_CONFIG and config helpers
│   └── main.py               # argparse CLI entry point
├── gateway/                  # Messaging gateway and platform adapters
├── plugins/                  # Built-in platform/model/memory/etc. plugins
├── tools/                    # Model tools registered with tools.registry
├── cron/                     # Durable scheduled jobs
├── skills/                   # Bundled skills
├── optional-skills/          # Optional / migration skill bundles
├── tests/                    # Pytest suite
├── website/                  # Docusaurus docs
├── ui-tui/                   # React/Ink terminal UI
├── tui_gateway/              # Local API/bridge for TUI/dashboard
└── apps/desktop/             # Electron desktop app
```

## Runtime Architecture

### AIAgent (`run_agent.py`)

`AIAgent` owns a conversation session: model routing, tool execution, context compression, memory integration, and final response handling.

High-level loop:

1. Build stable/context/volatile system prompt blocks.
2. Send messages + tool schemas to the provider.
3. If tool calls are returned, dispatch through `agent/tool_executor.py`, append tool results, and continue.
4. If a final text response is returned, save/deliver it.
5. Compress context near configured thresholds using the established compression path.

Important invariants:

- Never create two same-role messages in a row.
- Never inject synthetic user messages mid-loop.
- Keep tool schemas and stable prompt content unchanged for the life of a conversation.

### System prompt and context files

- `agent/system_prompt.py` assembles stable/context/volatile blocks.
- `agent/prompt_builder.py` discovers context files such as `AGENTS.md`, `CLAUDE.md`, and `.cursorrules` under `TERMINAL_CWD` / workdir.
- This `AGENTS.md` is deliberately compact because it is loaded automatically. Load `CONTRIBUTING.md` explicitly only when you need contributor policy or deep subsystem instructions.

### CLI and slash commands

- `hermes_cli/commands.py` is the source of truth for slash command metadata.
- CLI handlers live primarily in `cli.py` and `hermes_cli/*` mixins/modules.
- Gateway slash-command handling lives under `gateway/`.

When adding or changing a slash command:

1. Update `COMMAND_REGISTRY` in `hermes_cli/commands.py`.
2. Update CLI handling.
3. Update gateway handling if the command is available through messaging platforms.
4. Add targeted tests.

### Tools

Model tools live under `tools/` and register through `tools.registry`.

Rules:

- Handlers should return JSON strings.
- Add a `check_fn` / requirements gate for optional dependencies or credentials.
- Use `get_hermes_home()` for profile-safe paths.
- Keep schemas concise; every active schema is sent to the model.
- Prefer adding capability to an existing tool over adding a new core tool.

### Plugins

Hermes supports plugin directories for platforms, memory providers, model providers, image generation, dashboard/context integrations, and other edge capabilities. Keep plugin-specific capability at the edge; do not special-case third-party products in core files when a plugin boundary is appropriate.

### Memory

Memory has two layers:

- Built-in memory/profile files under the active `HERMES_HOME`.
- Optional external providers managed by `agent/memory_manager.py` / `agent/memory_provider.py`.

Do not hardcode default-profile paths. External memory providers must preserve session boundaries and should not mutate prompt/history in a way that breaks caching or role alternation.

### Cron

Cron jobs are stored per profile under `get_hermes_home()/cron`. The scheduler runs jobs in fresh sessions and usually sets `skip_memory=True` to avoid corrupting user representations.

Workdir-aware cron jobs can inject project context files from the configured workdir and run tools from that directory. Workdir jobs are serialized to avoid environment races.

### Delegation

`delegate_task` spawns isolated subagents for bounded subtasks. Delegations are not durable. Use cron or tracked background processes for durable long-running work.

### Kanban and Curator

- Kanban is a durable SQLite board for multi-agent work queues.
- Curator manages agent-created skills and usage/archival lifecycle.

Keep worker/tool surfaces gated so normal sessions do not pay for irrelevant schemas.

## Configuration Rules

Behavioral settings belong in `config.yaml`; secrets belong in `.env`.

Use profile-safe config and paths:

- `get_hermes_home()` for active profile state.
- `get_default_hermes_root()` only for truly shared root concerns.
- Avoid hardcoded `~/.hermes` in code paths.

Common config areas:

- `model`, `fallback_providers`, `custom_providers`
- `agent`, `compression`, `display`
- `terminal`, `tools`, `platform_toolsets`
- `gateway`, `cron`, `delegation`, `kanban`
- `memory`, `skills`, `plugins`, `mcp`

## Profile-Safe Coding

Profiles are isolation boundaries. A named profile lives under `~/.hermes/profiles/<name>/`; its config, memory, skills, plugins, cron jobs, sessions, and credentials must not silently bleed into the default profile or another profile.

Rules:

- Use active-profile paths for runtime state.
- Do not assume the default profile when a profile-specific gateway or CLI is active.
- When editing config/memory/skills/plugins, be explicit about which profile is targeted.
- Tests that touch state must use temp `HERMES_HOME` and must not write to the real user home.

## Important Policies

### Prompt caching must not break

A running conversation relies on a stable cached prefix. Do not change system prompt, toolset, tool schemas, provider-specific prompt fragments, or previous message content mid-conversation. If a change is necessary, require a fresh session or use an established compression/rebuild boundary.

### Background process notifications

For long bounded work, use tracked background processes with completion notification. Do not spawn untracked shell background jobs from tools. For long-lived servers/watchers, use tracked background mode and verify readiness with logs or health checks.

### Context size discipline

Any file named `AGENTS.md`, `CLAUDE.md`, or `.cursorrules` can be auto-injected. Keep such files concise and operational. Put deep contributor docs in non-auto-loaded files such as `CONTRIBUTING.md` and link to them.

## Known Pitfalls

### Do not hardcode `~/.hermes`

Use `get_hermes_home()` unless you intentionally need the shared root. Hardcoding breaks profiles and tests.

### Do not introduce new `simple_term_menu` usage

The project uses prompt-toolkit/Ink/TUI surfaces. Avoid adding `simple_term_menu` to new flows.

### Do not use `\033[K` in spinner/display code

Terminal clearing differs across surfaces. Prefer existing display helpers.

### `_last_resolved_tool_names` is process-global

Be careful with tests and code that assume tool resolution state is isolated.

### Do not hardcode cross-tool references in schema descriptions

Tool availability varies by platform/toolset. Schema text should not promise another tool exists unless the registry/toolset guarantees it.

### Gateway message guards

The gateway has multiple message/approval/control guards. Approval and control commands must bypass all relevant guards, not just one path.

### Squash merges from stale branches can revert fixes

Before merging or using a stale branch, compare against current `origin/main` and inspect diff direction. A stale branch can silently reintroduce old code.

### Do not wire in dead code without E2E validation

If a feature path is new, prove it is reachable through the actual CLI/gateway/tool/provider entry point, not only through unit-level imports.

### Tests must not write to real `~/.hermes`

Use temp dirs and monkeypatch `HERMES_HOME` / path constants. Many tests intentionally redirect cron, sessions, skills, and memory state.

## Testing Guidance

Use the wrapper unless there is a strong reason not to:

```bash
scripts/run_tests.sh tests/specific/test_file.py -q
```

Good tests assert behavior and invariants:

- What relationship must hold between two values?
- What user-visible behavior changes?
- What security boundary is preserved?
- What state is written to a temp `HERMES_HOME`?

Avoid tests that only freeze counts, catalog snapshots, current config version strings, or exact provider/model enumerations unless that value is the behavior under test.

## When You Need More Detail

Open [`CONTRIBUTING.md`](CONTRIBUTING.md) for:

- Full contribution rubric and PR review policy.
- The footprint ladder for new capabilities.
- Dependency pinning policy.
- Deep TUI, desktop, skin, plugin, and skill-authoring details.
- Detailed testing philosophy and examples.
