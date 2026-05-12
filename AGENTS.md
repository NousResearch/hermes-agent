# Hermes Agent Development Guide

Instructions for AI coding assistants and developers working on the Hermes Agent codebase.

Use this file for repo-specific rules only. Put long procedures in docs, ADRs, or skills.

## Agent skills

### Issue tracker

Issues are tracked primarily in GitHub Issues for `NousResearch/hermes-agent`. Local markdown under `.scratch/<feature>/` is allowed for drafts before publishing. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default Matt Pocock label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This repo uses root `CONTEXT.md` plus root `docs/adr/`. See `docs/agents/domain.md`.

### Karpathy Guidelines

Behavioral guidelines to reduce common LLM coding mistakes, derived from [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls.

Tradeoff: these guidelines bias toward caution over speed. For trivial tasks, use judgment.

#### 1. Think Before Coding

Do not assume. Do not hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them. Do not pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, ask.

#### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask: would a senior engineer say this is overcomplicated? If yes, simplify.

#### 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:
- Do not improve adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it. Do not delete it.

When your changes create orphans:
- Remove imports, variables, and functions that your changes made unused.
- Do not remove pre-existing dead code unless asked.

Test: every changed line should trace directly to the user's request.

#### 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" -> write tests for invalid inputs, then make them pass.
- "Fix the bug" -> write a test that reproduces it, then make it pass.
- "Refactor X" -> ensure tests pass before and after.

For multi-step tasks, state a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong criteria let agents loop independently. Weak criteria like "make it work" require clarification.

## Development environment

```bash
# Prefer .venv. Fall back to venv if that is what the checkout has.
source .venv/bin/activate  # or: source venv/bin/activate

# Standard test wrapper. Probes .venv, venv, then shared Hermes venv.
scripts/run_tests.sh
```

User config lives in `~/.hermes/config.yaml`. Secrets live in `~/.hermes/.env`. Logs live in `~/.hermes/logs/`.

## Project map

File counts shift. Trust the filesystem over this summary.

Core entry points:
- `run_agent.py` - `AIAgent`, conversation loop, tool-call loop.
- `model_tools.py` - tool discovery and `handle_function_call()`.
- `toolsets.py` - toolset definitions and `_HERMES_CORE_TOOLS`.
- `cli.py` - interactive CLI orchestration.
- `hermes_state.py` - SQLite session store and FTS search.
- `hermes_constants.py` - `get_hermes_home()`, profile-aware paths.
- `hermes_logging.py` - profile-aware logs.
- `batch_runner.py` - parallel batch processing.

Major directories:
- `agent/` - provider adapters, memory, caching, compression, prompt/context internals.
- `hermes_cli/` - CLI subcommands, config, setup, plugin loader, skin engine.
- `tools/` - tool implementations registered through `tools/registry.py`.
- `gateway/` - messaging gateway and platform adapters.
- `plugins/` - plugin system, memory providers, context engines, dashboards, media plugins.
- `skills/` - built-in skills bundled with repo.
- `optional-skills/` - heavier or niche skills not active by default.
- `ui-tui/` - Ink/React terminal UI.
- `tui_gateway/` - Python JSON-RPC backend for TUI.
- `acp_adapter/` - ACP server for IDE integration.
- `cron/` - scheduler.
- `scripts/` - test runner and maintenance scripts.
- `website/` - Docusaurus docs.
- `tests/` - pytest suite.

Dependency chain:

```text
tools/registry.py
  <- tools/*.py register tools
  <- model_tools.py discovers tools
  <- run_agent.py, cli.py, batch_runner.py, environments/
```

## Architecture notes

### Agent loop

`AIAgent.run_conversation()` builds messages, calls the provider with OpenAI-format chat messages and tool schemas, dispatches tool calls through `handle_function_call()`, appends tool results, and returns final text when no tool calls remain.

Rules:
- Preserve OpenAI message shape.
- Preserve role alternation. Never create two assistant or two user messages in a row.
- Reasoning content is stored in `assistant_msg["reasoning"]`.
- Do not change context, tools, or system prompt mid-conversation. This breaks prompt caching.

### CLI

- `cli.py` owns `HermesCLI` and `process_command()`.
- `hermes_cli/commands.py` is the canonical slash command registry.
- Help text, autocomplete, gateway command mapping, and Telegram menus derive from the registry.
- Skill slash commands inject loaded skills as user messages to preserve prompt caching.

### TUI

- `ui-tui/` is Ink/React.
- `tui_gateway/` is the Python JSON-RPC backend.
- Keep CLI, gateway, and TUI command semantics aligned.

### Gateway

- Platform adapters live under `gateway/platforms/`.
- Approval/control commands must bypass both message guards.
- Background process notifications should be rare, rate-limited, and not spam gateway chats.

## Adding or changing code

### Project architecture

Prefer Vertical Slice Architecture (VSA) for new project work: organize code around end-to-end features or use cases instead of broad horizontal layers. Keep each slice thin, testable, and independently useful. Use shared infrastructure only when duplication becomes real, not preemptively.

### Tools

To add a tool:
1. Create `tools/<name>.py`.
2. Register with `tools.registry.registry.register()`.
3. Add toolset metadata in `toolsets.py` when needed.
4. Return JSON strings from handlers.
5. Add `check_fn` and `requires_env` so tools appear only when usable.
6. Use `get_hermes_home()` for paths. Never hardcode `~/.hermes`.
7. Add tests for registration, schema, requirements, and behavior.

### Slash commands

To add a slash command:
1. Add `CommandDef` in `hermes_cli/commands.py`.
2. Add CLI handler in `cli.py`.
3. Add gateway behavior only if platform-specific handling is required.
4. Verify help, autocomplete, gateway command mapping, and Telegram menu behavior.

### Config

Config values belong in `config.yaml`. Secrets belong in `.env` only.

When adding config:
- Update defaults in `hermes_cli/config.py`.
- Update loaders in each path that reads the setting.
- Add migration or backward-compatible default if existing users need it.
- Use profile-aware paths.
- Add focused tests for default, override, and migration behavior.

Loader pitfall: there are multiple config paths. Check CLI, gateway, agent, and plugin paths before claiming a config change works everywhere.

### Plugins

Plugins should be isolated and optional.

Rules:
- Avoid importing heavy plugin deps at startup.
- Do not let plugin absence break core Hermes.
- Keep plugin config under the plugin namespace.
- Add smoke tests for discovery and disabled/missing dependency behavior.
- Reference/docs-companion plugins live in `hermes-example-plugins`, not this tree.

### Skills

Installed runtime skills live under `$HERMES_HOME/skills/<category>/<skill-name>/SKILL.md`, not necessarily the source checkout.

Rules:
- Keep skill frontmatter accurate.
- Directory leaf name should match `name:` when possible.
- Verify with `skill_view(name)` or `hermes skills list`, not only file presence.
- Do not solve discovery bugs with router skills unless the real issue is routing.

## Non-negotiable policies

### Prompt caching

Do not mutate loaded tool schemas, system prompt content, skill context, or enabled toolsets mid-session. Changes should take effect on a fresh session.

### Profile-safe paths

Never hardcode `~/.hermes`. Use `get_hermes_home()` from `hermes_constants` for config, logs, sessions, skills, auth, caches, and generated files.

Tests must not write to real `~/.hermes`. Use temp `HERMES_HOME` or existing test fixtures.

### Secrets

Never put tokens, API keys, OAuth files, private keys, auth DBs, logs, sessions, memory DBs, or caches into repo snapshots.

Before publishing runtime setup snapshots, secret-scan tracked files and verify sanitized paths.

### Display code

Do not introduce new `simple_term_menu` usage. Avoid raw ANSI erase-to-EOL sequences such as `\033[K` in spinner/display paths unless you verify cross-terminal behavior.

### Schema descriptions

Do not hardcode cross-tool references in tool schema descriptions. Tool availability is dynamic.

### Git workflow

Squash merges from stale branches can silently revert recent fixes. Before merging, verify branch freshness and inspect final diff against current main.

## Known pitfalls

- `_last_resolved_tool_names` in `model_tools.py` is process-global. Treat changes carefully.
- Gateway has two message guards. Approval/control commands must bypass both.
- Dead code wired into entry points is worse than unused code. Validate end to end before connecting paths.
- Catalog snapshot tests are brittle. Prefer behavior and invariant tests.
- Do not test raw counts of skills, providers, commands, or models unless the count itself is the contract.
- Do not assert config version literals unless migration behavior is the target.

## Testing

Default:

```bash
scripts/run_tests.sh
```

Focused:

```bash
python -m pytest tests/<area>/ -q -o 'addopts='
```

Use the wrapper when possible. It sets environment and avoids stale baked-in pytest flags.

Test style:
- Prefer behavior over snapshots.
- Test invariants that should remain true as catalogs grow.
- Clear baked-in pytest flags with `-o 'addopts='` when running direct pytest.
- Run focused tests first, then broader tests if the change touches shared paths.

Examples of good test targets:
- Tool registers only when requirements are met.
- Config migration preserves user values and fills defaults.
- Model catalog plumbing returns usable entries without asserting exact counts.
- Gateway command bypass works for approval/control commands.
- Profile paths resolve through `get_hermes_home()`.
