# Hermes Agent - Development Guide

Instructions for AI coding assistants and developers working on the hermes-agent codebase.

**Never give up on the right solution.**

## What Hermes Is

Hermes is a personal AI agent that runs the same agent core across a CLI, a messaging gateway (Telegram, Discord, Slack, etc.), a TUI, and an Electron desktop app. It learns across sessions (memory + skills), delegates to subagents, runs scheduled jobs, and drives a real terminal and browser. It is extended primarily through **plugins and skills**, not by growing the core.

Two properties shape almost every design decision and are the lens for reviewing any change:
- **Per-conversation prompt caching is sacred.** Anything that mutates past context, swaps toolsets, or rebuilds the system prompt mid-conversation invalidates that cache and multiplies costs. Avoid it (except for context compression).
- **The core is a narrow waist; capability lives at the edges.** Every model tool is sent on every API call, so the bar for new core tools is high. Prefer: CLI command + skill → service-gated tool → plugin → MCP server → core tool.

## Contribution Rubric

### What we want
- **Fix real bugs, well.** Reproduce on current `main`, trace to the exact line, and fix the whole bug class, not just the reported symptom.
- **Expand reach at the edges.** platform adapters, channels, providers, models, TUI/dashboard features are welcome. Integrate with config/setup UX.
- **Refactor god-files into clean modules.** Large mechanical refactors (extracting helpers from `cli.py` or `run_agent.py`) are highly valued.
- **Maintain cache safety and role alternation.** Never break strict user-assistant alternation or inject synthetic messages mid-loop.
- **Preserve git authorship.** Rebase-merge/cherry-pick external contributions instead of recreating them.

### What we don't want
- **Speculative infrastructure.** Do not add hooks or extension points without an active, concrete consumer.
- **New `HERMES_*` env vars for non-secret settings.** All behavioral configuration belongs in `config.yaml`. Env vars are for credentials only.
- **Change-detector tests, dead code, or core edits in plugins.** Plugins must not modify core files directly.

### Before you call it a bug — verify the premise
- **Intentional design, not a gap.** Restrictions are often deliberate (e.g. profile isolation). Check original commit intent (`git log -p -S`) first.
- **Verify against actual runtime logic.** Rationale often rests on wrong mental models. Trace the code before accepting bug claims.
- **Avoid "fixes" that destroy the feature they secure.** Find security mitigations that preserve the utility.

### The Footprint Ladder (New feature prioritization)
1. **Extend existing code** — zero new surface.
2. **CLI command + skill** — runs `hermes <cmd>` guided by a skill. Zero model-tool footprint.
3. **Service-gated tool (`check_fn`)** — only loaded when prerequisite is configured (e.g., Home Assistant).
4. **Plugin** — lives in `~/.hermes/plugins/`, loaded dynamically.
5. **MCP server (catalog)** — preferred over new core tools for structured I/O that doesn't need core access.
6. **New core tool** — absolute last resort.

## Project Structure & Dependencies

### Dev Environment
```bash
source .venv/bin/activate  # or: source venv/bin/activate
```
Tests probe `.venv`, then `venv`, then `$HOME/.hermes/hermes-agent/venv`.

### Dependency Pinning Policy
All dependencies must have upper bounds to mitigate supply-chain attacks:
- PyPI package: `>=floor,<next_major` (e.g., `"httpx>=0.28.1,<1"`)
- Pre-1.0 packages: `<0.(current_minor + 2)` (e.g., `>=0.29,<0.32`)
- Git URL: pin to commit SHA.
- GitHub Actions: pin to commit SHA + name tag.
- Run `uv lock` to update lockfile.

### File Dependency Chain
```
tools/registry.py (no deps) -> tools/*.py (register schema) -> model_tools.py (discovery) -> run_agent.py / cli.py
```

## AIAgent Class (run_agent.py)

The core loop in `run_conversation()` is synchronous, checking iteration budget and interrupts:
```python
while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) or self._budget_grace_call:
    if self._interrupt_requested: break
    response = client.chat.completions.create(model=model, messages=messages, tools=tool_schemas)
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = handle_function_call(tool_call.name, tool_call.args, task_id)
            messages.append(tool_result_message(result))
        api_call_count += 1
    else: return response.content
```
Messages follow OpenAI format. Reasoning is saved in `assistant_msg["reasoning"]`.

## CLI & TUI Architecture

- **CLI Engine:** Uses `rich` + `prompt_toolkit` + `SkinConfig` (`hermes_cli/skin_engine.py`). Autocomplete matches commands in `hermes_cli/commands.py`.
- **TUI & Dashboard:** React (Ink) frontend communicates with `tui_gateway` backend via JSON-RPC. Dashboard embeds TUI via WebSocket PTY bridge. See [docs/tui-architecture.md](docs/tui-architecture.md) for detail.
- **Theme/Skins:** Custom YAML configs under `~/.hermes/skins/`. See [docs/skin-theme-system.md](docs/skin-theme-system.md) for detail.

## Adding New Tools & Configuration

### Adding Core Tools
1. Define in `tools/your_tool.py` and register with `registry.register(name, toolset, schema, handler, check_fn, requires_env)`.
2. Add the tool name to a toolset in `toolsets.py` (e.g. `_HERMES_CORE_TOOLS`).
3. State files must resolve under `get_hermes_home()` for profile isolation.

### Adding Configuration
- **config.yaml options:** add to `DEFAULT_CONFIG` in `hermes_cli/config.py`. Bump `_config_version` only if doing active migrations of user data.
- **.env variables (secrets only):** add to `OPTIONAL_ENV_VARS` in `hermes_cli/config.py`.
- **Config loaders:** `load_cli_config()` (CLI), `load_config()` (CLI subcommands), raw YAML load (Gateway).

## Plugins & Skills

### Plugins
- **General plugins:** `PluginManager` loads from `~/.hermes/plugins/`. register lifecycle hooks or custom tools via `ctx`.
- **Memory plugins:** implement `MemoryProvider` ABC under `plugins/memory/<name>/`. Do not submit new in-tree memory providers (policy); publish them as standalone repos instead.
- **Model providers:** implement `ProviderProfile` under `plugins/model-providers/<name>/`.

### Skills
- Built-in skills live in `skills/`. Optional skills live in `optional-skills/` and are installed via `hermes skills install`.
- For skill authoring guidelines, schema frontmatter, and PR checks, refer to [docs/skill-authoring-standards.md](docs/skill-authoring-standards.md).

## Delegation, Curator & Scheduled Jobs

- **Delegation:** `delegate_task` tool spawns subagents in isolated environments (single or batch). Roles: `leaf` (no sub-delegation) or `orchestrator`.
- **Curator:** background worker (`agent/curator.py`) that auto-archives stale agent-created skills to `~/.hermes/skills/.archive/`.
- **Cron:** scheduling loop (`cron/scheduler.py`). Supports durations, cron strings, and ISO times. runs with `skip_memory=True`.

## Isolation & Multi-Instance Profiles

Hermes supports fully isolated profiles via the `HERMES_HOME` env var.
- **Path Rules:** Always use `get_hermes_home()` (from `hermes_constants`) for code paths; use `display_hermes_home()` for user messages. Mock `Path.home()` in profile tests.
- **Caching:** Commands mutating profile state must default to deferred invalidation, or require `--now` to preserve prompt caching.

## Known Pitfalls & Testing

- **simple_term_menu:** do not introduce new usages (causes rendering bugs in tmux); use `hermes_cli/curses_ui.py` instead.
- **ANSI Erase (`\033[K`):** do not use in display code; space-pad instead.
- **Cross-tool references:** do not hardcode other tools in schema descriptions; add them dynamically in `get_tool_definitions()`.
- **Gateway Message Guards:** base adapter queues messages; gateway runner intercepts signals. Ensure commands bypass both when blocked.
- **Testing:** always use `scripts/run_tests.sh`. See [docs/testing-guide.md](docs/testing-guide.md) for environment specs and subprocess isolation details.
