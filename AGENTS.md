# Hermes Agent - Development Guide

Instructions for AI coding assistants and developers working on the hermes-agent codebase.

## Development Environment

```bash
source venv/bin/activate  # ALWAYS activate before running Python
```

## Project Structure

```text
hermes-agent/
|- run_agent.py          AIAgent conversation loop
|- model_tools.py        tool discovery, schemas, dispatch
|- toolsets.py           toolset definitions, _HERMES_CORE_TOOLS
|- cli.py                HermesCLI interactive orchestrator
|- hermes_state.py       SQLite session store (FTS5)
|- batch_runner.py       parallel batch processing
|- agent/
|  |- prompt_builder.py
|  |- context_compressor.py
|  |- prompt_caching.py
|  |- auxiliary_client.py
|  |- model_metadata.py
|  |- models_dev.py
|  |- display.py
|  |- skill_commands.py
|  \- trajectory.py
|- hermes_cli/
|  |- main.py            all `hermes` subcommands
|  |- config.py          DEFAULT_CONFIG, OPTIONAL_ENV_VARS, migrations
|  |- commands.py        slash command registry + completer
|  |- callbacks.py       clarify, sudo, approval callbacks
|  |- setup.py           setup wizard
|  |- skin_engine.py     CLI theming
|  |- skills_config.py   `hermes skills`
|  |- tools_config.py    `hermes tools`
|  |- skills_hub.py      `/skills`
|  |- models.py
|  |- model_switch.py
|  \- auth.py
|- tools/
|  |- registry.py
|  |- approval.py
|  |- terminal_tool.py
|  |- process_registry.py
|  |- file_tools.py
|  |- web_tools.py
|  |- browser_tool.py
|  |- code_execution_tool.py
|  |- delegate_tool.py
|  |- mcp_tool.py
|  \- environments/      local, docker, ssh, modal, daytona, singularity
|- gateway/
|  |- run.py
|  |- session.py
|  \- platforms/         telegram, discord, slack, whatsapp, homeassistant, signal
|- acp_adapter/          ACP server for editors
|- cron/                 jobs.py, scheduler.py
|- environments/         RL training environments
\- tests/                pytest suite (~3000 tests)
```

User config:
- `~/.hermes/config.yaml` - settings
- `~/.hermes/.env` - API keys

## File Dependency Chain

```text
tools/registry.py
  <- tools/*.py (register() at import time)
  <- model_tools.py (_discover_tools(), handle_function_call())
  <- run_agent.py, cli.py, batch_runner.py, environments/
```

## AIAgent Class (run_agent.py)

Key API:

```python
class AIAgent:
    def __init__(
        self,
        model: str = "anthropic/claude-opus-4.6",
        max_iterations: int = 90,
        enabled_toolsets: list = None,
        disabled_toolsets: list = None,
        quiet_mode: bool = False,
        save_trajectories: bool = False,
        platform: str = None,      # "cli", "telegram", etc.
        session_id: str = None,
        skip_context_files: bool = False,
        skip_memory: bool = False,
        # plus provider, api_mode, callbacks, routing params
    ): ...

    def chat(self, message: str) -> str: ...
    def run_conversation(
        self,
        user_message: str,
        system_message: str = None,
        conversation_history: list = None,
        task_id: str = None,
    ) -> dict: ...
```

### Agent Loop

`run_conversation()` is synchronous:

```python
while api_call_count < self.max_iterations and self.iteration_budget.remaining > 0:
    response = client.chat.completions.create(
        model=model, messages=messages, tools=tool_schemas
    )
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = handle_function_call(tool_call.name, tool_call.args, task_id)
            messages.append(tool_result_message(result))
        api_call_count += 1
    else:
        return response.content
```

Notes:
- Messages use OpenAI format: `{"role": "system/user/assistant/tool", ...}`
- Reasoning content lives in `assistant_msg["reasoning"]`

## CLI Architecture (cli.py)

- Uses Rich for panels/banner and prompt_toolkit for input + autocomplete.
- `agent/display.py` contains `KawaiiSpinner` and tool-progress formatting.
- `load_cli_config()` merges built-in defaults with user YAML.
- `hermes_cli/skin_engine.py` applies `display.skin` at startup.
- `HermesCLI.process_command()` dispatches on the canonical name from `resolve_command()`.
- `agent/skill_commands.py` scans `~/.hermes/skills/` and injects skill content as a user message, not a system prompt, to preserve prompt caching.

### Slash Command Registry (`hermes_cli/commands.py`)

All slash commands come from `COMMAND_REGISTRY`. Downstream consumers derive from it automatically:

- CLI dispatch: `process_command()` + `resolve_command()`
- Gateway dispatch/help: `GATEWAY_KNOWN_COMMANDS`, `gateway_help_lines()`
- Telegram menu: `telegram_bot_commands()`
- Slack routing: `slack_subcommand_map()`
- Autocomplete: `COMMANDS` -> `SlashCommandCompleter`
- CLI help: `COMMANDS_BY_CATEGORY` -> `show_help()`

### Adding a Slash Command

1. Add a `CommandDef` in `hermes_cli/commands.py`:
```python
CommandDef("mycommand", "Description", "Session", aliases=("mc",), args_hint="[arg]")
```

2. Add CLI handling in `HermesCLI.process_command()`:
```python
elif canonical == "mycommand":
    self._handle_mycommand(cmd_original)
```

3. If gateway-visible, add handling in `gateway/run.py`:
```python
if canonical == "mycommand":
    return await self._handle_mycommand(event)
```

4. For persistent settings, use `save_config_value()` in `cli.py`.

Important `CommandDef` fields:
- `name`, `description`, `category`, `aliases`, `args_hint`
- `cli_only`, `gateway_only`
- `gateway_config_gate` for config-gated gateway exposure

Adding an alias usually means editing only the existing `aliases` tuple. Help, dispatch, Telegram, Slack, and autocomplete update automatically.

## Adding New Tools

New tools require changes in 3 places:

1. Create `tools/your_tool.py`
2. Import it in `model_tools.py` inside `_discover_tools()`
3. Add it to `toolsets.py` (`_HERMES_CORE_TOOLS` or a new toolset)

Minimal pattern:

```python
import json, os
from tools.registry import registry

def check_requirements() -> bool:
    return bool(os.getenv("EXAMPLE_API_KEY"))

def example_tool(param: str, task_id: str = None) -> str:
    return json.dumps({"success": True, "data": "..."})

registry.register(
    name="example_tool",
    toolset="example",
    schema={"name": "example_tool", "description": "...", "parameters": {...}},
    handler=lambda args, **kw: example_tool(
        param=args.get("param", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
    requires_env=["EXAMPLE_API_KEY"],
)
```

Rules:
- All tool handlers MUST return a JSON string.
- The registry handles schema collection, dispatch, availability checks, and error wrapping.
- If a schema mentions file paths, use `display_hermes_home()` so paths are profile-aware.
- If a tool stores state, logs, caches, or checkpoints, use `get_hermes_home()`, never `Path.home() / ".hermes"`.
- Agent-level tools (for example todo or memory) are intercepted in `run_agent.py` before `handle_function_call()`. See `todo_tool.py`.

## Adding Configuration

### config.yaml options:

1. Add the key to `DEFAULT_CONFIG` in `hermes_cli/config.py`
2. Bump `_config_version` (currently 5) so existing installs migrate

### .env variables:

Add entries to `OPTIONAL_ENV_VARS` in `hermes_cli/config.py`:

```python
"NEW_API_KEY": {
    "description": "What it's for",
    "prompt": "Display name",
    "url": "https://...",
    "password": True,
    "category": "tool",  # provider, tool, messaging, setting
},
```

### Config loaders (two separate systems):

- `load_cli_config()` in `cli.py` - interactive CLI
- `load_config()` in `hermes_cli/config.py` - `hermes tools`, `hermes setup`
- direct YAML load in `gateway/run.py` - gateway mode

## Skin/Theme System

The skin engine in `hermes_cli/skin_engine.py` is data-driven. New skins should not require code changes unless you are adding a built-in skin.

### Architecture

- `hermes_cli/skin_engine.py` - `SkinConfig`, built-in skins, YAML loading
- `~/.hermes/skins/*.yaml` - user-installed skins

Core functions:
- `init_skin_from_config()`
- `get_active_skin()`
- `set_active_skin(name)`
- `load_skin(name)`

Behavior:
- Load user skin first, then built-in, then fall back to `default`
- Missing keys inherit from the `default` skin

### What skins customize

Skins can change:
- banner colors
- response border color
- spinner faces, verbs, and optional wings
- tool output prefix
- per-tool emojis
- branding text (`agent_name`, welcome text, response label, prompt symbol)

### Built-in skins

Built-ins:
- `default`
- `ares`
- `mono`
- `slate`

### Adding a built-in skin

Add an entry to `_BUILTIN_SKINS` in `hermes_cli/skin_engine.py`:

```python
"mytheme": {
    "name": "mytheme",
    "description": "Short description",
    "colors": {...},
    "spinner": {...},
    "branding": {...},
    "tool_prefix": "|",
},
```

### User skins (YAML)

User skins live at `~/.hermes/skins/<name>.yaml`:

```yaml
name: cyberpunk
description: Neon terminal theme

colors:
  banner_border: "#FF00FF"
  banner_title: "#00FFFF"
  banner_accent: "#FF1493"

spinner:
  thinking_verbs: ["jacking in", "decrypting", "uploading"]

branding:
  agent_name: "Cyber Agent"
  response_label: " Cyber "

tool_prefix: "|"
```

Activate with `/skin cyberpunk` or `display.skin: cyberpunk` in `config.yaml`.

## Important Policies

### Prompt Caching Must Not Break

Do NOT implement changes that:
- alter past context mid-conversation
- change toolsets mid-conversation
- reload memories or rebuild system prompts mid-conversation

Prompt-cache breaks are expensive. The only allowed context rewrite is context compression.

### Working Directory Behavior

- CLI: current directory (`os.getcwd()`)
- Messaging: `MESSAGING_CWD` env var, defaulting to the user's home directory

### Background Process Notifications (Gateway)

When `terminal(background=true, check_interval=...)` is used, the gateway watcher can push process updates back to chat.

Control with `display.background_process_notifications` in `config.yaml` or `HERMES_BACKGROUND_NOTIFICATIONS`:
- `all` - running updates + final message (default)
- `result` - final completion message only
- `error` - final message only when exit code != 0
- `off` - no watcher messages

## Profiles: Multi-Instance Support

Profiles provide isolated `HERMES_HOME` directories for config, API keys, memory, sessions, skills, and gateway state.

The key mechanism is `_apply_profile_override()` in `hermes_cli/main.py`, which sets `HERMES_HOME` before module imports. Code that uses `get_hermes_home()` becomes profile-aware automatically.

### Rules for profile-safe code

1. Use `get_hermes_home()` from `hermes_constants` for Hermes state paths.
   Never hardcode `~/.hermes` or `Path.home() / ".hermes"` for read/write state.

```python
from hermes_constants import get_hermes_home
config_path = get_hermes_home() / "config.yaml"
```

2. Use `display_hermes_home()` from `hermes_constants` for user-facing paths.

```python
from hermes_constants import display_hermes_home
print(f"Config saved to {display_hermes_home()}/config.yaml")
```

3. Module-level constants are fine if they use `get_hermes_home()`. Import-time caching happens after `_apply_profile_override()`.

4. Tests that mock `Path.home()` must also set `HERMES_HOME`:

```python
with patch.object(Path, "home", return_value=tmp_path), \
     patch.dict(os.environ, {"HERMES_HOME": str(tmp_path / ".hermes")}):
    ...
```

5. Gateway platform adapters that connect with unique credentials must use token locks:
   - call `acquire_scoped_lock()` from `gateway.status` in `connect()`/`start()`
   - call `release_scoped_lock()` in `disconnect()`/`stop()`
   This prevents multiple profiles from using the same token. See `gateway/platforms/telegram.py`.

6. Profile operations are HOME-anchored, not HERMES_HOME-anchored:
   `_get_profiles_root()` returns `Path.home() / ".hermes" / "profiles"`, not `get_hermes_home() / "profiles"`.
   This is intentional so profile commands can see all profiles.

## Known Pitfalls

### DO NOT hardcode `~/.hermes` paths

Use `get_hermes_home()` for code paths and `display_hermes_home()` for user-visible output. Hardcoded `~/.hermes` breaks profiles.

### DO NOT use `simple_term_menu` for interactive menus

It has rendering bugs in tmux and iTerm2. Use `curses` instead. See `hermes_cli/tools_config.py`.

### DO NOT use `\033[K` (ANSI erase-to-EOL) in spinner/display code

Under prompt_toolkit's `patch_stdout` it can leak as literal `?[K`. Use space-padding instead:

```python
f"\r{line}{' ' * pad}"
```

### `_last_resolved_tool_names` is process-global in `model_tools.py`

`_run_single_child()` in `delegate_tool.py` saves and restores it around subagent execution. New code that reads this global may see temporarily stale values during child-agent runs.

### DO NOT hardcode cross-tool references in schema descriptions

A tool schema must not mention tools from other toolsets by name. Those tools may be unavailable due to config, missing API keys, or disabled toolsets, which can cause hallucinated calls. If a cross-reference is truly needed, inject it dynamically in `get_tool_definitions()` in `model_tools.py`. See the `browser_navigate` and `execute_code` post-processing blocks.

### Tests must not write to `~/.hermes/`

The `_isolate_hermes_home` autouse fixture in `tests/conftest.py` redirects `HERMES_HOME` to a temp directory. Do not bypass it.

For profile tests, also mock `Path.home()` so `_get_profiles_root()` and `_get_default_hermes_home()` stay inside the temp tree. Pattern from `tests/hermes_cli/test_profiles.py`:

```python
@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home
```

## Testing

```bash
source venv/bin/activate
python -m pytest tests/ -q
python -m pytest tests/test_model_tools.py -q
python -m pytest tests/test_cli_init.py -q
python -m pytest tests/gateway/ -q
python -m pytest tests/tools/ -q
```

Always run the full suite before pushing changes.
