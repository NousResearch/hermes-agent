# Hermes Agent - Core Contributor Contract

Instructions for AI coding assistants and developers working on the Hermes Agent codebase.

**Never give up on the right solution.**

This file is intentionally short because it is project context that can be loaded into every agent turn. Detailed contributor guidance lives in `docs/agent-context/`:

- `docs/agent-context/contribution-rubric.md` — contribution intent, PR premise checks, Footprint Ladder.
- `docs/agent-context/architecture-map.md` — repo layout and major runtime surfaces.
- `docs/agent-context/plugin-and-skill-policy.md` — tools, config, dependencies, plugins, providers, skills.
- `docs/agent-context/runtime-systems.md` — toolsets, delegation, curator, cron, kanban, project context files.
- `docs/agent-context/testing-and-pitfalls.md` — prompt caching, profiles, testing, and known pitfalls.

## What Hermes is

Hermes is a personal AI agent that runs the same core across CLI, gateway, TUI, dashboard, desktop, IDE/ACP, cron, and other surfaces. It learns through memory and skills, delegates to subagents, schedules jobs, and uses real terminal/browser/file tools.

Two properties shape almost every design decision:

1. **Per-conversation prompt caching is sacred.** Do not mutate past context, swap toolsets, reload memories, or rebuild system prompts mid-conversation. Context compression is the exception.
2. **The core is a narrow waist; capability lives at the edges.** Every core model tool is sent on every API call. Prefer CLI commands, skills, plugins, service-gated tools, or MCP over growing the core tool schema.

## Contribution priorities

- Fix real bugs with current-`main` reproduction and line-level cause.
- Preserve prompt caching, strict message role alternation, and byte-stable system prompts.
- Prefer existing infrastructure over new managers/hooks/modules.
- Prefer edge expansion over core expansion.
- Preserve contributor authorship when building on external work.
- Use behavior/invariant tests, not snapshots of expected-to-change catalogs.
- Validate real paths with temp `HERMES_HOME` for config, security, file/network I/O, backends, and resolution chains.

## Footprint Ladder

Choose the least permanent surface that solves the problem:

1. Extend existing code.
2. Add CLI command + skill.
3. Add service-gated tool with `check_fn`.
4. Ship a plugin.
5. Ship an MCP server/catalog entry.
6. Add a new core tool only as a last resort.

A new core tool must be fundamental, broadly useful, and not reachable through terminal/file, a skill, a plugin, or MCP.

## Config and secrets

- User-facing non-secret behavior goes in `config.yaml`, not `.env`.
- `.env` is for secrets only: API keys, tokens, passwords.
- Add config defaults in `hermes_cli/config.py::DEFAULT_CONFIG`.
- Bump `_config_version` only for actual migrations/transformations.
- Remember the three config loading paths: CLI-specific, general CLI subcommands, and gateway raw YAML.

## Tool and plugin rules

- Built-in tools require `tools/<name>.py` plus explicit exposure in `toolsets.py`.
- Tool handlers must return JSON strings.
- Tool schemas that mention Hermes paths must use profile-aware display paths.
- Plugins must not modify core files such as `run_agent.py`, `cli.py`, `gateway/run.py`, or `hermes_cli/main.py`.
- If a plugin needs more capability, expand the generic plugin surface instead of hardcoding plugin-specific logic in core.
- New third-party product integrations should ship as standalone plugin repos, not new in-tree plugin directories.

## Profile-safe paths

- Use `get_hermes_home()` for all Hermes state paths.
- Use `display_hermes_home()` for user-facing path messages.
- Do not hardcode `~/.hermes` or `Path.home() / ".hermes"` for profile-scoped state.
- Tests that mock `Path.home()` should also set `HERMES_HOME`.
- Tests must never write to the real `~/.hermes/`.

## Testing rules

Use the canonical runner for final validation:

```bash
scripts/run_tests.sh
scripts/run_tests.sh tests/path/or_file.py
scripts/run_tests.sh tests/path/test_file.py::test_name -v --tb=long
```

Do not rely on raw `pytest` for final validation; the wrapper enforces CI-like env isolation, temp `HERMES_HOME`, credential stripping, timezone/locale settings, xdist, and subprocess-per-test-file isolation.

Do not write change-detector tests. Avoid tests that freeze exact model lists, provider counts, config version literals, or catalog snapshots. Test behavior and invariants instead.

## Known high-risk areas

- System prompt construction and context compression.
- Message role alternation and synthetic message insertion.
- Toolset/schema selection and tool availability.
- Gateway message guards and approval/control commands.
- Profile handling and `HERMES_HOME` scoping.
- Config propagation between CLI and gateway loaders.
- Core tool additions and plugin/provider discovery.
- Squash merges from stale branches.
- Wiring unused modules into live paths without E2E validation.

## Quick orientation

- Core loop: `run_agent.py`.
- Prompt building: `agent/prompt_builder.py`.
- Tool dispatch: `model_tools.py`, `tools/registry.py`, `toolsets.py`.
- CLI: `cli.py`, `hermes_cli/main.py`, `hermes_cli/commands.py`.
- Gateway: `gateway/run.py` and platform adapters.
- Config: `hermes_cli/config.py` plus gateway config loaders.
- Skills: `skills/`, `optional-skills/`, and skill tools.
- Plugins: `plugins/`, user plugins under `$HERMES_HOME/plugins/`.
- Tests: `tests/`, run through `scripts/run_tests.sh`.

When a task touches a specialized area, read the relevant `docs/agent-context/*.md` file before editing.
