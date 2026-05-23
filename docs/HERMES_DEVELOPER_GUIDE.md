# Hermes Developer Guide

This guide preserves the practical workflows that used to live inline in
`AGENTS.md`. Keep `AGENTS.md` concise; put detailed development procedures here.

## Environment

Prefer the repo venv:

```bash
source .venv/bin/activate
```

If `.venv` is absent, use `venv` or the shared Hermes venv. The canonical test
runner probes these automatically:

```bash
scripts/run_tests.sh
```

## Key File Map

- `run_agent.py`: `AIAgent` and core agent loop.
- `model_tools.py`: tool schema resolution and function-call dispatch.
- `tools/registry.py`: tool registration; no imports from model tools.
- `toolsets.py`: toolset definitions and shared Hermes core tools.
- `cli.py`: classic interactive CLI orchestration.
- `hermes_cli/commands.py`: slash command registry.
- `gateway/run.py`: gateway runtime and platform command dispatch.
- `gateway/platforms/`: messaging/API adapters.
- `hermes_state.py`: SQLite sessions/search state.
- `hermes_constants.py`: profile-aware Hermes paths.
- `hermes_logging.py`: logs and redacting formatters.
- `agent/`: providers, memory, prompts, context, guardrails, redaction.
- `plugins/`: plugin manifests and implementations.
- `cron/`: scheduler and jobs.
- `ui-tui/` and `tui_gateway/`: Ink TUI and Python JSON-RPC backend.

## Workflow Index

- Tools: see "Adding A Built-In Tool" and "Toolsets".
- Slash commands: see "Adding A Slash Command".
- Plugins: see "Plugins".
- Skills: see "Skills".
- TUI/dashboard: see "TUI, Web, And Dashboard".
- Config/profile safety: see "Config And Profile Paths" and "Profiles".
- Cron: see "Cron".
- Kanban: see "Kanban".
- Delegation: see "Delegation".
- Curator: see "Curator".
- Skins/themes: see "Skin And Theme System".
- Testing: see "Testing" and "Why The Wrapper Matters".
- Known pitfalls: see "Known Pitfalls".

## Adding A Built-In Tool

1. Add a tool module under `tools/`.
2. Register it with `tools.registry.register()` at module import time.
3. Add the tool name to the right toolset in `toolsets.py`.
4. Add availability checks that fail closed and do not print secrets.
5. Add focused tests under `tests/tools/`.
6. Run the relevant tier from `HERMES_TESTING_PLAN.md`.

Keep tool handlers small and explicit. Use structured parsing and existing
helpers rather than ad hoc string manipulation.

Built-in/core tool path:

```python
from tools.registry import registry

def check_requirements() -> bool:
    return True

def example_tool(args, **kwargs) -> str:
    return "{\"success\": true}"

registry.register(
    name="example_tool",
    toolset="example",
    schema={"name": "example_tool", "description": "...", "parameters": {...}},
    handler=example_tool,
    check_fn=check_requirements,
)
```

Rules:

- Handlers must return a JSON string.
- Add the tool to a toolset; discovery imports the module, but toolsets expose
  the tool to agents.
- Use `display_hermes_home()` in schema text that mentions Hermes paths.
- Use `get_hermes_home()` for state files.
- Prefer plugin tools for local/user-specific tools; edit core only when the
  tool should ship in Hermes itself.

## Adding A Slash Command

1. Add a `CommandDef` in `hermes_cli/commands.py`.
2. Add CLI handling in `HermesCLI.process_command()` or the appropriate
   `hermes_cli/` module.
3. If available in the gateway, add gateway handling in `gateway/run.py`.
4. Add tests for command resolution, help/autocomplete, and gateway behavior.

The command registry feeds CLI help, gateway help, Telegram bot commands,
Slack subcommands, and autocomplete. Do not duplicate command metadata in
multiple places.

## Config And Profile Paths

Use profile-aware helpers:

- `hermes_constants.get_hermes_home()`
- `hermes_constants.display_hermes_home()`
- `hermes_cli.config.load_config()`
- `hermes_cli.config.save_config_value()`

Do not hardcode `~/.hermes` in new code unless the file is explicitly
Operator-local documentation. Runtime state may be profile-scoped.

Secrets belong in Keychain or `.env` through existing config/auth helpers, not
in docs, plists, logs, or generated reports.

## Gateway And Runtime

The active local gateway path is:

```bash
/Users/agent1/Operator/scripts/hermes-gateway.sh
```

It sources Keychain-backed environment through:

```bash
/Users/agent1/Operator/scripts/hermes-env.sh
```

Do not replace this wrapper with a bare command. If a gateway change requires
restart, use drain-aware Hermes/launchd paths and validate with:

```bash
hermes gateway status
hermes doctor
```

## TUI, Web, And Dashboard

TUI entrypoint:

```bash
hermes --tui
```

Process model:

```text
hermes --tui
  -> Node Ink frontend over stdio JSON-RPC
  -> Python tui_gateway backend
  -> AIAgent, tools, sessions, slash-command dispatch
```

Key surfaces:

- Frontend: `ui-tui/`
- Backend: `tui_gateway/`
- Dashboard PTY bridge: `hermes_cli/pty_bridge.py`
- Dashboard web route: `hermes_cli/web_server.py`
- Chat page: `web/src/pages/ChatPage.tsx`

Dev commands:

```bash
cd ui-tui
npm install
npm run dev
npm start
npm run build
npm run type-check
npm run lint
npm run fmt
npm test
```

Dashboard rule:

- The dashboard embeds the real `hermes --tui`; do not rebuild the primary
  transcript/composer/chat experience in React.
- Supporting React UI is fine for sidebars, inspectors, summaries, and status
  panels when the PTY terminal remains the primary chat surface.

For UI changes, verify the actual UI surface. For local browser checks, use
the available browser tooling when the route/port is clear.

## Dependency Pinning

Core dependency policy is conservative because this is an agent with local
execution and credential-adjacent surfaces.

Rules:

- Keep core dependencies exact-pinned where the repo already does so.
- Keep provider/tool-specific packages in extras or lazy install paths.
- Do not move optional provider, media, browser, or platform dependencies into
  the core dependency set without written justification.
- When dependency files change, regenerate lock files with the repo-standard
  command and run targeted tests.
- GitHub Actions should stay pinned to SHAs with comments for human-readable
  versions.

## Adding Configuration

For config keys:

1. Add defaults in `hermes_cli/config.py`.
2. Bump config version only when existing user config needs migration or shape
   transformation.
3. Use deep-merge defaults for additive keys.
4. Keep secrets out of `config.yaml`; secret values belong in `.env`, Keychain,
   or auth helpers.

Important config sections include:

- `model`
- `agent`
- `terminal`
- `compression`
- `display`
- `stt`
- `tts`
- `memory`
- `security`
- `delegation`
- `smart_model_routing`
- `checkpoints`
- `auxiliary`
- `curator`
- `skills`
- `gateway`
- `logging`
- `cron`
- `profiles`
- `plugins`

Config loaders differ by context:

- CLI commands often use `hermes_cli.config.load_config()`.
- Runtime provider resolution uses `hermes_cli/runtime_provider.py`.
- Gateway/platform config uses `gateway/config.py`.

Know which path you are in before adding config reads.

## Skin And Theme System

The skin engine is data-driven:

- Engine: `hermes_cli/skin_engine.py`
- User skins: `~/.hermes/skins/*.yaml`
- Runtime config key: `display.skin`
- Slash command: `/skin`

Skins can customize:

- Banner colors.
- Response border/labels.
- Spinner faces/verbs/wings.
- Tool prefix and per-tool icons.
- Agent name, welcome text, prompt symbol.
- Status-bar colors.

Adding a built-in skin means adding a data entry to `_BUILTIN_SKINS`; avoid
logic branches for individual themes. User skins should inherit missing values
from `default`.

## Plugins

Hermes has several plugin surfaces. Use a plugin instead of editing core when
the feature is optional, user-specific, or locally installed.

General plugin layout:

```text
~/.hermes/plugins/<name>/plugin.yaml
~/.hermes/plugins/<name>/__init__.py
```

Capabilities:

- `ctx.register_tool(...)`
- lifecycle hooks such as `pre_tool_call`, `post_tool_call`,
  `pre_llm_call`, `post_llm_call`, `on_session_start`, and `on_session_end`
- CLI subcommands through plugin registration

Pitfalls:

- `discover_plugins()` is triggered through `model_tools.py` import paths in
  many flows; code that reads plugin state early may need explicit discovery.
- Standalone plugins are opt-in by config.
- Backend, platform, model-provider, and memory plugins have specialized
  loading rules.

Memory-provider plugins:

- Live under `plugins/memory/<name>/` for built-ins.
- Implement `agent/memory_provider.py`.
- Are orchestrated by `agent/memory_manager.py`.
- Only one external provider should be active at a time.
- New memory backends should generally ship as standalone plugins, not new
  in-tree provider directories.

Model-provider plugins:

- Live under `plugins/model-providers/<name>/`.
- Register a `ProviderProfile`.
- Are discovered lazily by provider registry code, not by the general plugin
  manager.

Dashboard/context/image/video/web plugins follow their own ABC/registry
patterns. When in doubt, add a generic hook or context method instead of
hardcoding plugin-specific logic into core files.

## Skills

Skill locations:

- `skills/`: bundled active skills.
- `optional-skills/`: heavier or niche official skills installed explicitly.

Skill standards:

- `SKILL.md` frontmatter should include name, description, version, author,
  license, platforms, and Hermes metadata where relevant.
- Descriptions should be short, plain, and capability-focused.
- Use Hermes tool names in prose, not raw shell utilities as the primary
  instruction surface.
- Put scripts in `scripts/`, references in `references/`, and templates in
  `templates/`.
- Add tests under `tests/skills/test_<skill>_skill.py`.
- No live network calls in skill tests.
- Heavy or niche skills belong in `optional-skills/`.

Skill body order:

```text
# <Skill> Skill
## When to Use
## Prerequisites
## How to Run
## Quick Reference
## Procedure
## Pitfalls
## Verification
```

Review external skill PRs for platform gating, path safety, test isolation,
and whether the contribution belongs in bundled or optional skills.

## Toolsets

All toolsets are defined in `toolsets.py`.

Important rules:

- `_HERMES_CORE_TOOLS` is the shared default bundle many platforms inherit.
- Adding a registry tool is not enough; expose it through the correct toolset.
- Avoid schema descriptions that mention tools from other toolsets unless the
  reference is injected dynamically only when that tool is available.
- Platform tool availability comes from config enabled/disabled lists and
  toolset composition.

Common toolsets include:

- `web`
- `browser`
- `terminal`
- `file`
- `code_execution`
- `vision`
- `video`
- `image_gen`
- `tts`
- `skills`
- `todo`
- `memory`
- `session_search`
- `clarify`
- `delegation`
- `cronjob`
- `messaging`
- `kanban`

## Delegation

`tools/delegate_tool.py` spawns isolated subagents.

Use it for bounded parallel work that can return a summary to the parent.
Do not use it for durable long-running jobs; use cron or background terminal
processes for that.

Key concepts:

- Single task: pass one goal and optional context/toolsets.
- Batch: pass multiple tasks for parallel workers.
- `role="leaf"` is the safe default.
- `role="orchestrator"` can delegate further and is depth-limited.
- Worker count and depth are bounded by `delegation.*` config.
- Subagent dangerous-command approval is denied by default unless explicitly
  configured otherwise.

## Curator

Curator maintains agent-created skills.

Key files:

- `agent/curator.py`
- `agent/curator_backup.py`
- `hermes_cli/curator.py`
- `tools/skill_usage.py`

Invariants:

- Only touches skills with agent-created provenance.
- Never deletes user/bundled/hub skills automatically.
- Archives instead of deleting.
- Pinned skills are exempt from automatic stale/archive transitions.

Use:

```bash
hermes curator status
hermes curator run
hermes curator pause
hermes curator resume
```

## Cron

Core files:

- `cron/jobs.py`
- `cron/scheduler.py`

User surfaces:

```bash
hermes cron list
hermes cron add
hermes cron edit
hermes cron pause
hermes cron resume
hermes cron run
hermes cron remove
```

Supported schedules include durations, "every" phrases, cron expressions, and
one-shot ISO timestamps.

Hardening invariants:

- Cron sessions are bounded and should not monopolize the scheduler.
- The tick lock prevents duplicate ticks.
- Cron output must be redacted.
- Cron sessions skip memory by default.
- Delivery should not corrupt the target conversation's role alternation.

## Kanban

Kanban is the durable multi-agent work queue.

Core surfaces:

- `hermes_cli/kanban.py`
- `tools/kanban_tools.py`
- `plugins/kanban/dashboard/`
- gateway-dispatched worker runs when enabled

Common commands:

```bash
hermes kanban init
hermes kanban create
hermes kanban list
hermes kanban show
hermes kanban assign
hermes kanban comment
hermes kanban complete
hermes kanban block
hermes kanban unblock
hermes kanban dispatch
```

Isolation model:

- Board is the hard boundary.
- Tenant is a soft namespace inside a board.
- Workers get board/task env and only the kanban toolset when appropriate.
- Repeated failures should block rather than spin.

## Important Runtime Policies

Prompt caching:

- Do not mutate past context mid-conversation.
- Do not change toolsets mid-conversation except through explicit cache-aware
  invalidation.
- Do not reload memories or rebuild system prompts mid-conversation.
- Slash commands that mutate prompt state should default to next-session
  effect, with an explicit immediate-invalidation path when needed.

Background process notifications:

- Gateway can watch background terminal jobs.
- Verbosity is controlled through display/background notification config.
- Keep completion messages concise and redacted.

## Memory Changes

Before changing memory, read `HERMES_MEMORY_PLAN.md`.

Important constraints:

- Built-in memory files are injected as frozen snapshots at session start.
- External memory providers are managed through `MemoryManager`.
- Only one external provider should be active at a time.
- Memory deletion must cover markdown, structured DBs, sessions, logs, caches,
  backups, and profile stores when requested.

## Profiles

Hermes supports multiple isolated profiles with separate `HERMES_HOME`
directories.

Rules for profile-safe code:

- Use `get_hermes_home()` for runtime state paths.
- Use `display_hermes_home()` for user-facing text.
- Do not hardcode `Path.home() / ".hermes"` in code that reads/writes state.
- Tests that mock `Path.home()` should also set `HERMES_HOME`.
- Gateway adapters with unique credentials should use scoped locks to prevent
  two profiles from using the same bot token or credential.
- Profile list/root operations are home-anchored by design so any active
  profile can see the full profile set.

## Security Pitfalls

Before changing approvals, terminal, code execution, env passthrough, logs,
MCP, browser, launchd, or external actions, read `HERMES_SECURITY_MODEL.md`.

Common mistakes:

- Printing raw env or config values.
- Bypassing the gateway wrapper.
- Treating `/health/detailed` 401 as failure when no bearer is supplied.
- Adding a send/post/deploy action without typed confirmation.
- Writing files under unscoped absolute paths.
- Adding dependencies to core instead of lazy extras.

## Known Pitfalls

- Do not hardcode `~/.hermes` paths in code. Use profile-aware helpers.
- Do not introduce new `simple_term_menu` usage; prefer the repo curses UI
  patterns for interactive menus.
- Do not use ANSI erase-to-end-of-line in spinner/display code when
  prompt_toolkit is active; use space padding.
- `_last_resolved_tool_names` is process-global in `model_tools.py`; subagent
  execution saves/restores it, but new code should be aware of staleness risk.
- Do not hardcode cross-tool references in schema descriptions.
- Gateway has both base-adapter and runner guards; approval/control commands
  that must work while an agent is running need to bypass both.
- Do not wire unused/dead modules into live paths without an end-to-end import
  and runtime validation path.
- Tests must not write to the real `~/.hermes`.
- Before squash-merging stale branches, verify the diff does not revert recent
  unrelated fixes.

## Testing

Always prefer `scripts/run_tests.sh` over direct pytest. It enforces CI-like
environment behavior: credential-shaped env vars unset, deterministic timezone
and locale, and a stable worker count.

Use the smallest tier that covers the changed surface:

```bash
scripts/run_tests.sh tests/test_project_metadata.py
scripts/run_tests.sh tests/hermes_cli
scripts/run_tests.sh tests/gateway
scripts/run_tests.sh tests/tools
```

For docs-only/control-plane metadata changes, run:

```bash
git diff --check
hermes doctor
hermes gateway status
scripts/run_tests.sh tests/test_project_metadata.py
```

## Why The Wrapper Matters

The test wrapper closes common local-vs-CI gaps:

- Provider API keys in the developer environment.
- Real `HOME` and `~/.hermes` state.
- Local timezone.
- Local locale.
- Workstation xdist worker count.

If direct pytest is unavoidable, activate the venv and use a CI-like worker
count:

```bash
source .venv/bin/activate
python -m pytest tests/ -q -n 4
```

Do not add tests that merely detect expected data changes such as model catalog
counts, config version numbers, or hardcoded provider-list lengths. Test
behavior and invariants instead.
