# Hermes Agent — AGENTS.md

Instructions for agents and developers working on the Hermes Agent codebase.

This file is intentionally short. The pre-simplification full development guide is preserved at `docs/dev/agents-full-pre-simplification.md`; load it when you need deeper architecture or historical context.

## Mission

Hermes is a personal AI agent framework spanning CLI, messaging gateway, TUI, desktop, cron, skills, memory, plugins, MCP, and real terminal/browser tooling.

The core design principle: **Hermes should grow capability at the edges while keeping the agent core narrow and cache-stable.**

## Non-Negotiable Design Rules

1. **Never give up on the right solution.** Reproduce, understand, and fix root causes rather than papering over symptoms.
2. **Prompt caching is sacred.** Do not mutate past context, swap toolsets, reload memories, or rebuild the system prompt mid-conversation except through the established compression path.
3. **Preserve message role alternation.** Never create two assistant or two user messages in a row; never inject synthetic user messages mid-loop.
4. **Keep the core narrow.** New model tools are expensive because schemas ship on every API call.
5. **Behavioral config belongs in `config.yaml`; secrets belong in `.env`.** Do not add user-facing `HERMES_*` env vars for non-secret settings.
6. **Use profile-safe paths.** Code paths use `get_hermes_home()`; user-facing messages use `display_hermes_home()`.
7. **Never hardcode `~/.hermes` for state.** Profiles must remain isolated.
8. **Plugins must not modify core files.** If a plugin needs a capability, expand the generic plugin surface.
9. **Tests assert behavior/invariants, not snapshots.** Avoid model-list, config-version, enum-count, and other change-detector tests.
10. **E2E validation beats mocks** for config propagation, security boundaries, remote backends, file/network I/O, tool resolution, and plugin loading.

## New Capability Footprint Ladder

Choose the least permanent surface that solves the problem:

1. Extend existing code.
2. CLI command + skill.
3. Service-gated tool with `check_fn`.
4. Plugin.
5. MCP server/catalog entry.
6. New core tool only as last resort.

When multiple PRs add the same category of capability, design one shared interface/ABC and plug implementations into it.

## Development Workflow

- Prefer `.venv`; fall back to `venv` only if that is what the checkout has.
- Before fixing a bug, verify the premise on current `main` and identify where the behavior fails.
- Before restricting behavior, inspect original intent with git history when needed.
- Preserve contributor credit when salvaging external work.
- New dependencies need upper bounds; Git URLs and GitHub Actions should be SHA-pinned where policy requires.

## Testing

Use the project wrapper, not raw pytest, for CI-parity:

```bash
scripts/run_tests.sh
scripts/run_tests.sh tests/gateway/
scripts/run_tests.sh tests/agent/test_foo.py::test_x -v --tb=long
```

The wrapper provides hermetic environment handling, temp `HERMES_HOME`, UTC/C.UTF-8 defaults, xdist, and subprocess-per-test isolation.

## Key Files

| Area | File/dir |
|---|---|
| Agent loop | `run_agent.py` |
| Tool discovery/dispatch | `model_tools.py` |
| Toolsets | `toolsets.py` |
| CLI | `cli.py`, `hermes_cli/` |
| Gateway | `gateway/` |
| TUI | `ui-tui/`, `tui_gateway/` |
| Desktop | `apps/desktop/` |
| Plugins | `plugins/` |
| Skills | `skills/`, `optional-skills/` |
| Cron | `cron/` |
| Tests | `tests/` |
| Docs | `website/docs/`, `docs/` |

## Must-Read References

| Need | Reference |
|---|---|
| Full pre-simplification AGENTS guide | `docs/dev/agents-full-pre-simplification.md` |
| User/developer docs | `website/docs/` |
| Tool authoring details | `docs/dev/agents-full-pre-simplification.md#adding-new-tools` |
| Plugin policy/details | `docs/dev/agents-full-pre-simplification.md#plugins` |
| Testing policy/details | `docs/dev/agents-full-pre-simplification.md#testing` |
| Known pitfalls | `docs/dev/agents-full-pre-simplification.md#known-pitfalls` |

## Before Editing This File

Keep this file to mandatory startup rules. Move architecture explanations, examples, and long procedures into focused docs and link them here.
