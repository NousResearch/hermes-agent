# Hermes Agent - Development Guide

This file is the short, high-signal contract for AI coding assistants working on `hermes-agent`.
Keep it lean: target **≤150 lines**. Put subsystem detail in [`docs/agent-guide/`](docs/agent-guide/README.md) and link it from here.

## Core mission

- Hermes Agent is a Python agent framework with CLI, gateway, tools, skills, profiles, cron, Kanban, TUI, plugins, and tests.
- Optimize for safe, profile-aware, well-tested changes with minimal prompt/context bloat.
- If a task touches a subsystem, open the relevant reference before editing.
- Config belongs in `~/.hermes/config.yaml`; secrets belong only in `~/.hermes/.env` or provider auth stores. Never commit secrets.

## Development environment

```bash
# Prefer .venv; fall back to venv if that is what the checkout has.
source .venv/bin/activate   # or: source venv/bin/activate

# Preferred test entry point; probes .venv, venv, then shared Hermes venv.
scripts/run_tests.sh
```

Useful checks:

```bash
hermes --version
hermes doctor
git status --branch --short
```

## Repository map: load-bearing files

- `run_agent.py` — `AIAgent`, core conversation loop, tool-call iteration.
- `model_tools.py` — tool discovery and `handle_function_call()` dispatch.
- `toolsets.py` — toolset definitions and default/core tool exposure.
- `cli.py`, `hermes_cli/` — interactive CLI, config commands, slash-command registry.
- `agent/` — prompt builder, memory, compression, model/provider routing, skill dispatch.
- `tools/` — tool implementations; `tools/registry.py` is the registration hub.
- `tools/environments/` — local/docker/ssh/modal/etc. terminal backends.
- `gateway/` — messaging gateway and platform adapters.
- `plugins/` — plugin families: kanban, model providers, memory, image gen, dashboard/context, etc.
- `cron/` — scheduler and jobs.
- `ui-tui/`, `tui_gateway/` — Ink TUI and JSON-RPC backend.
- `tests/` — pytest suite; tests must isolate `HERMES_HOME` and never touch the real user home.

Full structure: [`docs/agent-guide/project-structure.md`](docs/agent-guide/project-structure.md).

## Universal coding rules

1. Preserve prompt caching: do not mutate system prompt/tools/context shape mid-conversation unless intentionally designing that feature.
2. Preserve message role alternation: never emit consecutive assistant/user messages in stored histories.
3. Use `get_hermes_home()` / profile-aware helpers for Hermes paths; do not hardcode `~/.hermes` in runtime code.
4. Tool handlers must return JSON strings and register through `tools.registry` with a requirements/check function.
5. Keep secrets out of config, logs, docs, tests, fixtures, commits, and tool outputs.
6. New config options need defaults, migration/check handling, docs, and profile-safe behavior.
7. New CLI/slash commands go through the central registry so CLI/gateway/help/autocomplete stay consistent.
8. Do not add dead code or plan-only wiring; verify end to end.
9. Avoid brittle catalog-count tests; test behavior/invariants, not snapshot counts.
10. Keep this root file short. Add detail to `docs/agent-guide/` and reference it.

Policy details: [`docs/agent-guide/important-policies.md`](docs/agent-guide/important-policies.md), [`docs/agent-guide/known-pitfalls.md`](docs/agent-guide/known-pitfalls.md).

## Change routing: open the matching reference first

- Agent loop / `AIAgent`: [`aiagent-class-run-agent-py.md`](docs/agent-guide/aiagent-class-run-agent-py.md)
- CLI / slash commands: [`cli-architecture-cli-py.md`](docs/agent-guide/cli-architecture-cli-py.md)
- TUI / dashboard chat: [`tui-architecture-ui-tui-tui-gateway.md`](docs/agent-guide/tui-architecture-ui-tui-tui-gateway.md)
- Tools: [`adding-new-tools.md`](docs/agent-guide/adding-new-tools.md)
- Config/env: [`adding-configuration.md`](docs/agent-guide/adding-configuration.md)
- Dependency pins: [`dependency-pinning-policy.md`](docs/agent-guide/dependency-pinning-policy.md)
- Skin/theme: [`skin-theme-system.md`](docs/agent-guide/skin-theme-system.md)
- Plugins/providers/memory: [`plugins.md`](docs/agent-guide/plugins.md)
- Skills: [`skills.md`](docs/agent-guide/skills.md)
- Toolsets: [`toolsets.md`](docs/agent-guide/toolsets.md)
- Delegation: [`delegation-delegate-task.md`](docs/agent-guide/delegation-delegate-task.md)
- Curator: [`curator-skill-lifecycle.md`](docs/agent-guide/curator-skill-lifecycle.md)
- Cron: [`cron-scheduled-jobs.md`](docs/agent-guide/cron-scheduled-jobs.md)
- Kanban: [`kanban-multi-agent-work-queue.md`](docs/agent-guide/kanban-multi-agent-work-queue.md)
- Profiles: [`profiles-multi-instance-support.md`](docs/agent-guide/profiles-multi-instance-support.md)
- Testing: [`testing.md`](docs/agent-guide/testing.md)

Reference index and extraction analysis: [`docs/agent-guide/README.md`](docs/agent-guide/README.md).

## Common implementation patterns

### Adding a tool

1. Create `tools/<name>.py` with `registry.register(...)`.
2. Add/select an appropriate toolset in `toolsets.py` when needed.
3. Return JSON strings from handlers; include a `check_fn` and `requires_env` for optional dependencies.
4. Add focused tests and run the relevant test subset.

Details: [`docs/agent-guide/adding-new-tools.md`](docs/agent-guide/adding-new-tools.md).

### Adding a slash command

1. Add `CommandDef` to `hermes_cli/commands.py`.
2. Add the CLI handler in `cli.py` / relevant `hermes_cli` module.
3. Add gateway handling only if the command must work from messaging platforms.
4. Verify help, aliases, CLI behavior, and gateway behavior if applicable.

Details: [`docs/agent-guide/cli-architecture-cli-py.md`](docs/agent-guide/cli-architecture-cli-py.md).

### Adding config

1. Add defaults and migration/check behavior.
2. Keep secrets in `.env`, not `config.yaml`.
3. Ensure profiles and non-default `HERMES_HOME` work.
4. Document the setting and test both default and overridden values.

Details: [`docs/agent-guide/adding-configuration.md`](docs/agent-guide/adding-configuration.md).

## Testing and verification

- Prefer `scripts/run_tests.sh`; use targeted pytest only when the wrapper is impractical.
- Tests must redirect `HERMES_HOME` to temp dirs and avoid the real `~/.hermes`.
- For platform-specific behavior, patch all relevant platform probes, not only `sys.platform`.
- Before finishing: run format/lint/test checks relevant to the changed area, inspect `git diff`, and verify no secrets.

Testing detail: [`docs/agent-guide/testing.md`](docs/agent-guide/testing.md).

## Commit discipline

- Use concise conventional commits: `fix:`, `feat:`, `refactor:`, `docs:`, `chore:`.
- Commit only intentional changes. Do not include local env files, credentials, generated caches, or unrelated artifacts.
- If a remote exists and the task requires delivery, push and verify `git status --branch --short` plus remote state.

## When unsure

1. Search the repo and `docs/agent-guide/` for an existing pattern.
2. Read the subsystem reference linked above.
3. Make the smallest safe change.
4. Add or update tests that prove behavior, not implementation trivia.
5. Keep root `AGENTS.md` concise; update references instead of expanding it.
