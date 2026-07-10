# Hermes Agent - Startup Contract

This file is loaded into every fresh Hermes development session. Keep it short.
Detailed subsystem guidance lives in `docs/agent-guides/`; read the relevant
guide before editing that area.

## What Hermes Is

Hermes is a personal AI agent that runs the same agent core across CLI,
messaging gateways, TUI, WebUI/dashboard, and Electron desktop. It learns across
sessions through memory and skills, can delegate work, runs scheduled jobs, and
drives terminal/browser tools.

Two design constraints govern most changes:

- Prompt caching is sacred. A long-lived conversation reuses a cached prefix
  every turn. Do not mutate past context, swap toolsets, or rebuild the system
  prompt mid-conversation except through the established compression path.
- The core is a narrow waist. Every core model tool is sent on every API call,
  so new capability should usually live at the edge: existing code, CLI command
  plus skill, service-gated tool, plugin, or MCP server before any new core tool.

## Non-Negotiable Rules

- Verify the real bug and the intended design before changing behavior. Reproduce
  on current `main` and identify the line/path where the bug manifests.
- Preserve prompt-cache stability, message role alternation, and byte-stable
  system prompts for the life of a conversation.
- Do not add speculative hooks, callbacks, managers, tools, or extension points
  without a concrete consumer.
- Do not add user-facing `HERMES_*` env vars for non-secret behavior. `.env` is
  for secrets only: API keys, tokens, passwords. Behavioral settings belong in
  `config.yaml`.
- Do not add a new core tool when terminal/file, a skill, a service-gated tool,
  a plugin, or MCP can solve it.
- Do not add lazy pagination to instructional content the agent must read fully
  such as skills, prompts, or playbooks.
- Do not wire in dead code without E2E validation of the real path.
- Do not write tests that snapshot changing catalogs, model lists, config
  versions, or enumeration counts. Prefer invariants and behavior contracts.

## Footprint Ladder

When adding capability, choose the least permanent surface that works:

1. Extend existing code.
2. Add a CLI command plus skill.
3. Add a service-gated tool with a `check_fn`.
4. Ship a plugin.
5. Add an MCP server to the catalog.
6. Add a new core tool only as a last resort.

If several PRs integrate the same category of backend/provider/notifier, design
one interface and orchestrator instead of merging one-off special cases.

## Project Map

- `run_agent.py` - `AIAgent`, core conversation loop.
- `model_tools.py` - tool orchestration and discovery.
- `toolsets.py` - toolset definitions.
- `cli.py` and `hermes_cli/` - interactive CLI and subcommands.
- `agent/` - provider adapters, prompt assembly, memory, compression, runtime
  helpers.
- `tools/` - registered tool implementations.
- `gateway/` - messaging gateway and platform adapters.
- `plugins/` - bundled plugins and plugin interfaces.
- `skills/` and `optional-skills/` - bundled skill content.
- `ui-tui/` and `tui_gateway/` - Ink TUI and Python JSON-RPC backend.
- `apps/desktop/` - Electron desktop app.
- `web/` - WebUI/dashboard frontend.
- `tests/` - pytest suite.

Use `~/.hermes/config.yaml` for settings and `~/.hermes/.env` for secrets.
Logs live under `~/.hermes/logs/`.

## Read Before Editing

Open the matching guide before non-trivial edits:

- Product intent, contribution rubric, and footprint policy:
  `docs/agent-guides/product-principles.md`
- Environment setup, project structure, TypeScript style, dependency chain:
  `docs/agent-guides/development-environment.md`
- `AIAgent`, agent loop, CLI architecture, slash commands:
  `docs/agent-guides/agent-runtime-cli.md`
- TUI, dashboard-embedded TUI, Electron desktop, desktop slash commands:
  `docs/agent-guides/tui-desktop.md`
- Tool creation, dependency pinning, config and env policy:
  `docs/agent-guides/tools-config.md`
- Skins, plugins, memory/model providers, skills, skill authoring:
  `docs/agent-guides/plugins-skills.md`
- Toolsets, delegation, curator, cron, kanban:
  `docs/agent-guides/operations.md`
- Prompt caching, gateway notifications, profiles, pitfalls, testing:
  `docs/agent-guides/policies-testing.md`

Subdirectory `AGENTS.md` files are discovered progressively as work enters those
directories. Do not preload detailed subsystem docs unless the task touches that
subsystem.

## Testing

Use the canonical test wrapper:

```bash
scripts/run_tests.sh
scripts/run_tests.sh tests/path/to/test_file.py -- -q
```

The wrapper runs each test file in a separate subprocess with deterministic env
settings. If a task touches Node/Electron/TUI code, also run the relevant package
test, typecheck, or lint command from that subsystem guide.

For integration-sensitive changes, test the real path against a temporary
`HERMES_HOME`. Mocks are not enough for config propagation, security boundaries,
remote backends, provider routing, file/network I/O, or prompt assembly.

## Profile-Safe Paths

Hermes supports multiple profiles. Do not hardcode `~/.hermes` for profile-owned
state. Use the profile-aware helpers from `hermes_constants.py` and surrounding
code. New tests must not write to the real user `~/.hermes`; use temp homes and
explicit environment isolation.

Profile-local state includes skills, plugins, cron jobs, memories, sessions,
logs, and config-derived runtime state. Do not modify another profile's state
unless the user explicitly asks.

## Tool and Plugin Rules

For custom or local-only capability, prefer `~/.hermes/plugins/<name>/` and
`ctx.register_tool(...)` over editing core. Built-in tools are justified only
when fundamental, broadly useful, and unreachable through smaller surfaces.

Tool descriptions are paid on every API call. Keep schemas terse. Avoid
hardcoded cross-tool references inside schema descriptions; use toolsets and
runtime guidance instead.

## Skills Rules

Skills are on-demand knowledge documents. Keep skill frontmatter accurate and
descriptions trigger-oriented. Skill bodies may be detailed because they are
loaded only when relevant, but once a skill is selected it must be read fully.

Do not use skills as a dumping ground for always-needed invariants; keep those
in this startup contract.

## Completion Standard

When asked to build, run, or verify something, produce a working artifact backed
by real command output. Do not stop at a stub, plan, or plausible explanation.
If the direct path is blocked, say why and try the next practical alternative.
