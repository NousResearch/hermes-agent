# Hermes Agent - Development Guide (Trimmed)

> Full upstream version: `AGENTS.upstream-full.md`

## Project Structure

```
hermes-agent/
├── run_agent.py          # AIAgent class — core conversation loop
├── model_tools.py        # Tool orchestration, discover_builtin_tools(), handle_function_call()
├── toolsets.py           # Toolset definitions, _HERMES_CORE_TOOLS list
├── agent/                # Agent internals (provider adapters, memory, caching, compression)
├── tools/                # Tool implementations — auto-discovered via tools/registry.py
├── gateway/              # Messaging gateway — run.py + session.py + platforms/
│   ├── platforms/        # Adapter per platform (telegram, discord, slack, etc.)
│   │   ├── feishu_card.py # Card JSON builder, tool semantics, footer formatting
│   │   └── base.py       # Base adapter class
│   └── run.py            # GatewayRun — message routing, agent lifecycle, heartbeat
├── plugins/              # Plugin system (see Plugins section)
│   └── platforms/feishu/  # Feishu adapter (migrated from gateway/platforms/feishu.py)
│       ├── __init__.py    # register() entry point
│       └── adapter.py     # FeishuAdapter — WebSocket + interactive cards + live progress
├── skills/               # Built-in skills bundled with the repo
├── optional-skills/      # Heavier/niche skills, installed via `hermes skills install`
├── cron/                 # Scheduler — jobs.py, scheduler.py
└── tests/                # Pytest suite
```

**User config:** `~/.hermes/config.yaml` (settings), `~/.hermes/.env` (API keys only).
**Logs:** `~/.hermes/logs/` — `agent.log`, `errors.log`, `gateway.log`.

## Plugins

Two plugin surfaces, both under `plugins/`:

### General plugins (`hermes_cli/plugins.py` + `plugins/<name>/`)

`PluginManager` discovers from `~/.hermes/plugins/`, `./.hermes/plugins/`, and pip entry points.
Each plugin exposes `register(ctx)` that can:
- Register lifecycle hooks: `pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end`
- Register new tools via `ctx.register_tool(...)`
- Register CLI subcommands via `ctx.register_cli_command(...)`

**Rule:** plugins MUST NOT modify core files (`run_agent.py`, `cli.py`, `gateway/run.py`). Expand the generic plugin surface instead.

### Memory-provider plugins (`plugins/memory/<name>/`)

Implements `MemoryProvider` ABC (`agent/memory_provider.py`), orchestrated by `agent/memory_manager.py`.
Built-in: honcho, mem0, supermemory, byterover, hindsight, holographic, openviking, retaindb.
New providers must ship as standalone plugin repos, not in-tree.

### Model-provider plugins (`plugins/model-providers/<name>/`)

Each provider's `__init__.py` calls `providers.register_provider(ProviderProfile(...))`.
Lazy discovery on first `get_provider_profile()` or `list_providers()`.
User plugins override bundled ones (last-writer-wins).

## Skills

- **`skills/`** — built-in, active by default.
- **`optional-skills/`** — heavier/niche, installed via `hermes skills install official/<category>/<skill>`.

### SKILL.md frontmatter

Fields: `name`, `description` (≤60 chars), `version`, `author`, `license`, `platforms`, `metadata.hermes.tags/category/related_skills/config`.

### Skill authoring standards

1. `description` ≤ 60 chars, one sentence, no marketing words
2. Tools referenced must be native Hermes tools or declared MCP servers
3. `platforms:` gating must match actual script imports
4. `author` credits human contributor first
5. Modern section order: title → intro → When to Use → Prerequisites → How to Run → Quick Reference → Procedure → Pitfalls → Verification
6. Scripts in `scripts/`, references in `references/`, templates in `templates/`
7. Tests at `tests/skills/test_<skill>_skill.py`

## Toolsets

Defined in `toolsets.py` as `TOOLSETS` dict. Each platform adapter picks a base toolset.
Enable/disable via `tools.<platform>.enabled/disabled` in `config.yaml`.

## Delegation (`delegate_task`)

`tools/delegate_tool.py` spawns subagent with isolated context + terminal.
- **Single:** `goal` + optional `context`, `toolsets`
- **Batch:** `tasks: [...]` — concurrent, capped by `delegation.max_concurrent_children` (default 3)
- Roles: `leaf` (default, no delegation) / `orchestrator` (can spawn workers)
- NOT durable — for long-running work use `cronjob` or `terminal(background=True)`

## Known Pitfalls

- **DO NOT hardcode `~/.hermes` paths** — use `get_hermes_home()` from `hermes_constants`
- **Gateway has TWO message guards** — both must bypass approval/control commands. Base adapter queues in `_pending_messages`, gateway runner intercepts `/stop`, `/new`, etc.
- **Squash merges from stale branches silently revert fixes** — ensure branch is up to date before squash-merge
- **Feishu adapter moved to plugin** — `gateway/platforms/feishu.py` → `plugins/platforms/feishu/adapter.py`; stale `.pyc` in old path can cause import confusion
- **Tests must not write to `~/.hermes/`** — `_isolate_hermes_home` autouse fixture redirects to temp dir

## Testing

Use `uv run pytest` for this fork. Upstream uses `scripts/run_tests.sh` for CI parity.
