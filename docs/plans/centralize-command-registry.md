# Plan: Centralize Slash Command Registry

## Problem

Slash command definitions are scattered across 7+ locations with significant drift:

| Location | What it defines | Commands |
|----------|----------------|----------|
| `hermes_cli/commands.py` | COMMANDS_BY_CATEGORY dict | 34 commands |
| `cli.py` process_command() | if/elif dispatch chain | ~30 branches |
| `gateway/run.py` _known_commands | Hook emission set | 25 entries |
| `gateway/run.py` _handle_message() | if dispatch chain | ~22 branches |
| `gateway/run.py` _handle_help_command() | Hardcoded help text list | 22 lines |
| `gateway/platforms/telegram.py` | BotCommand registration | 20 commands |
| `gateway/platforms/discord.py` | @tree.command decorators | 22 commands |
| `gateway/platforms/slack.py` | subcommand_map dict | 20 mappings |

**Known drift:**
- Telegram missing: `/rollback`, `/background`, `/bg`, `/plan`, `/set-home`
- Slack missing: `/sethome`, `/set-home`, `/update`, `/voice`, `/reload-mcp`, `/plan`
- Gateway help text missing: `/bg` alias mention
- Gateway `_known_commands` has duplicate `"reasoning"` entry
- Gateway dispatch has dead code: second `"reasoning"` check (line 1384) never executes
- Adding one alias (`/bg`) required touching 6 files + 1 test file

## Goal

Single source of truth for "what commands exist, what are their aliases, and
what platforms support them." Adding a command or alias should require exactly
one definition change + the handler implementation.

## Design

### 1. CommandDef dataclass (hermes_cli/commands.py)

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class CommandDef:
    name: str                          # canonical name without slash: "background"
    description: str                   # human-readable description
    category: str                      # "Session", "Configuration", "Tools & Skills", "Info", "Exit"
    aliases: tuple[str, ...] = ()      # alternative names: ("bg",)
    args_hint: str = ""                # argument placeholder: "<prompt>", "[name]", "[level|show|hide]"
    gateway: bool = True               # available in gateway (Telegram/Discord/Slack/etc.)
    cli_only: bool = False             # only available in CLI (e.g., /clear, /paste, /skin)
    gateway_only: bool = False         # only available in gateway (e.g., /status, /sethome, /update)
```

### 2. COMMAND_REGISTRY list (hermes_cli/commands.py)

Replace COMMANDS_BY_CATEGORY with a flat list of CommandDef objects:

```python
COMMAND_REGISTRY: list[CommandDef] = [
    # Session
    CommandDef("new", "Start a new session (fresh session ID + history)", "Session", aliases=("reset",)),
    CommandDef("clear", "Clear screen and start a new session", "Session", cli_only=True),
    CommandDef("history", "Show conversation history", "Session", cli_only=True),
    CommandDef("save", "Save the current conversation", "Session", cli_only=True),
    CommandDef("retry", "Retry the last message (resend to agent)", "Session"),
    CommandDef("undo", "Remove the last user/assistant exchange", "Session"),
    CommandDef("title", "Set a title for the current session", "Session", args_hint="[name]"),
    CommandDef("compress", "Manually compress conversation context", "Session"),
    CommandDef("rollback", "List or restore filesystem checkpoints", "Session", args_hint="[number]"),
    CommandDef("stop", "Kill all running background processes", "Session"),
    CommandDef("background", "Run a prompt in the background", "Session", aliases=("bg",), args_hint="<prompt>"),
    CommandDef("status", "Show session info", "Session", gateway_only=True),
    CommandDef("sethome", "Set this chat as the home channel", "Session", gateway_only=True, aliases=("set-home",)),
    CommandDef("resume", "Resume a previously-named session", "Session", args_hint="[name]"),

    # Configuration
    CommandDef("config", "Show current configuration", "Configuration", cli_only=True),
    CommandDef("model", "Show or change the current model", "Configuration", args_hint="[name]"),
    CommandDef("provider", "Show available providers and current provider", "Configuration"),
    CommandDef("prompt", "View/set custom system prompt", "Configuration", cli_only=True, args_hint="[text]"),
    CommandDef("personality", "Set a predefined personality", "Configuration", args_hint="[name]"),
    CommandDef("verbose", "Cycle tool progress display: off → new → all → verbose", "Configuration", cli_only=True),
    CommandDef("reasoning", "Manage reasoning effort and display", "Configuration", args_hint="[level|show|hide]"),
    CommandDef("skin", "Show or change the display skin/theme", "Configuration", cli_only=True, args_hint="[name]"),
    CommandDef("voice", "Toggle voice mode", "Configuration", args_hint="[on|off|tts|status]"),

    # Tools & Skills
    CommandDef("tools", "List available tools", "Tools & Skills", cli_only=True),
    CommandDef("toolsets", "List available toolsets", "Tools & Skills", cli_only=True),
    CommandDef("skills", "Search, install, inspect, or manage skills", "Tools & Skills", cli_only=True),
    CommandDef("cron", "Manage scheduled tasks", "Tools & Skills", cli_only=True, args_hint="[subcommand]"),
    CommandDef("reload-mcp", "Reload MCP servers from config", "Tools & Skills", aliases=("reload_mcp",)),
    CommandDef("plugins", "List installed plugins and their status", "Tools & Skills", cli_only=True),

    # Info
    CommandDef("help", "Show available commands", "Info"),
    CommandDef("usage", "Show token usage for the current session", "Info"),
    CommandDef("insights", "Show usage insights and analytics", "Info", args_hint="[days]"),
    CommandDef("platforms", "Show gateway/messaging platform status", "Info", cli_only=True, aliases=("gateway",)),
    CommandDef("paste", "Check clipboard for an image and attach it", "Info", cli_only=True),
    CommandDef("update", "Update Hermes Agent to the latest version", "Info", gateway_only=True),

    # Exit
    CommandDef("quit", "Exit the CLI", "Exit", cli_only=True, aliases=("exit", "q")),
]
```

### 3. Derived data structures (hermes_cli/commands.py)

Build all downstream dicts/sets from the registry automatically:

```python
# --- derived lookups (rebuilt on import, all consumers use these) ---

# name_or_alias -> CommandDef  (used by dispatch to resolve aliases)
_COMMAND_LOOKUP: dict[str, CommandDef] = {}
for _cmd in COMMAND_REGISTRY:
    _COMMAND_LOOKUP[_cmd.name] = _cmd
    for _alias in _cmd.aliases:
        _COMMAND_LOOKUP[_alias] = _cmd

def resolve_command(name: str) -> CommandDef | None:
    """Resolve a command name or alias to its CommandDef."""
    return _COMMAND_LOOKUP.get(name.lower().lstrip("/"))

# Backwards-compat: flat COMMANDS dict (slash-prefixed key -> description)
COMMANDS: dict[str, str] = {}
for _cmd in COMMAND_REGISTRY:
    desc = _cmd.description
    if _cmd.args_hint:
        desc = f"{desc} (usage: /{_cmd.name} {_cmd.args_hint})"
    COMMANDS[f"/{_cmd.name}"] = desc
    for _alias in _cmd.aliases:
        alias_desc = f"{desc} (alias for /{_cmd.name})" if _alias not in ("reset",) else desc
        COMMANDS[f"/{_alias}"] = alias_desc

# Backwards-compat: COMMANDS_BY_CATEGORY
COMMANDS_BY_CATEGORY: dict[str, dict[str, str]] = {}
for _cmd in COMMAND_REGISTRY:
    cat = COMMANDS_BY_CATEGORY.setdefault(_cmd.category, {})
    cat[f"/{_cmd.name}"] = COMMANDS[f"/{_cmd.name}"]
    for _alias in _cmd.aliases:
        cat[f"/{_alias}"] = COMMANDS[f"/{_alias}"]

# Gateway known commands set (for hook emission)
GATEWAY_KNOWN_COMMANDS: set[str] = set()
for _cmd in COMMAND_REGISTRY:
    if not _cmd.cli_only:
        GATEWAY_KNOWN_COMMANDS.add(_cmd.name)
        GATEWAY_KNOWN_COMMANDS.update(_cmd.aliases)

# Gateway help lines (for _handle_help_command)
def gateway_help_lines() -> list[str]:
    """Generate gateway help text from the registry."""
    lines = []
    for cmd in COMMAND_REGISTRY:
        if cmd.cli_only:
            continue
        args = f" {cmd.args_hint}" if cmd.args_hint else ""
        alias_note = ""
        if cmd.aliases:
            alias_strs = ", ".join(f"`/{a}`" for a in cmd.aliases)
            alias_note = f" (alias: {alias_strs})"
        lines.append(f"`/{cmd.name}{args}` — {cmd.description}{alias_note}")
    return lines

# Telegram BotCommand list
def telegram_bot_commands() -> list[tuple[str, str]]:
    """Return (command_name, description) pairs for Telegram's setMyCommands."""
    result = []
    for cmd in COMMAND_REGISTRY:
        if cmd.cli_only:
            continue
        # Telegram doesn't support hyphens in command names
        tg_name = cmd.name.replace("-", "_")
        result.append((tg_name, cmd.description))
    return result

# Slack subcommand map
def slack_subcommand_map() -> dict[str, str]:
    """Return subcommand -> /command mapping for Slack's /hermes handler."""
    mapping = {}
    for cmd in COMMAND_REGISTRY:
        if cmd.cli_only:
            continue
        mapping[cmd.name] = f"/{cmd.name}"
        for alias in cmd.aliases:
            mapping[alias] = f"/{alias}"
    return mapping
```

### 4. Consumer changes

#### cli.py — process_command()

The dispatch chain stays as-is (if/elif is fine for ~30 commands), but alias
resolution moves to the top:

```python
def process_command(self, command: str) -> bool:
    cmd_original = command.strip()
    cmd_lower = cmd_original.lower()
    base = cmd_lower.split()[0].lstrip("/")

    # Resolve alias to canonical name
    cmd_def = resolve_command(base)
    if cmd_def:
        canonical = cmd_def.name
    else:
        canonical = base

    # Dispatch on canonical name
    if canonical in ("quit", "exit", "q"):
        ...
    elif canonical == "help":
        ...
    elif canonical == "background":  # no more "or startswith /bg"
        ...
```

This eliminates every `or cmd_lower.startswith("/bg")` style alias check.

#### gateway/run.py — _handle_message()

```python
from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

# Replace hardcoded _known_commands set
if command and command in GATEWAY_KNOWN_COMMANDS:
    await self.hooks.emit(f"command:{command}", {...})

# Resolve aliases before dispatch
cmd_def = resolve_command(command)
canonical = cmd_def.name if cmd_def else command

if canonical in ("new",):
    return await self._handle_reset_command(event)
elif canonical == "background":
    return await self._handle_background_command(event)
...
```

#### gateway/run.py — _handle_help_command()

```python
from hermes_cli.commands import gateway_help_lines

async def _handle_help_command(self, event):
    lines = gateway_help_lines()
    # ... append skill commands, format, return
```

Delete the hardcoded 22-line list entirely.

#### gateway/platforms/telegram.py — set_my_commands()

```python
from hermes_cli.commands import telegram_bot_commands

async def set_my_commands(self):
    commands = [BotCommand(name, desc) for name, desc in telegram_bot_commands()]
    await self._bot.set_my_commands(commands)
```

Delete the hardcoded 20-entry list.

#### gateway/platforms/slack.py — _handle_slash_command()

```python
from hermes_cli.commands import slack_subcommand_map

async def _handle_slash_command(self, command: dict):
    ...
    subcommand_map = slack_subcommand_map()
    ...
```

Delete the hardcoded dict.

#### gateway/platforms/discord.py — _register_slash_commands()

Discord is the **exception**. Its `@tree.command()` decorators need typed
parameters, custom descriptions, and platform-specific interaction handling
(defer, ephemeral, followups). These can't be generated from a simple registry.

**Approach:** Keep the decorator registrations, but validate at startup that
every registered Discord command has a matching entry in COMMAND_REGISTRY
(except platform-specific ones like `/ask` and `/thread`). Add a test for this.

```python
# In _register_slash_commands(), after all decorators:
_DISCORD_ONLY_COMMANDS = {"ask", "thread"}
registered = {cmd.name for cmd in tree.get_commands()}
registry_names = {c.name for c in COMMAND_REGISTRY if not c.cli_only}
# Warn about Discord commands not in registry (excluding Discord-only)
for name in registered - registry_names - _DISCORD_ONLY_COMMANDS:
    logger.warning("Discord command /%s not in central registry", name)
```

## Files Changed

| File | Change |
|------|--------|
| `hermes_cli/commands.py` | Add `CommandDef`, `COMMAND_REGISTRY`, derived structures, helper functions |
| `cli.py` | Add alias resolution at top of `process_command()`, remove per-command alias checks |
| `gateway/run.py` | Import `GATEWAY_KNOWN_COMMANDS` + `resolve_command` + `gateway_help_lines`, delete hardcoded sets/lists |
| `gateway/platforms/telegram.py` | Import `telegram_bot_commands()`, delete hardcoded BotCommand list |
| `gateway/platforms/slack.py` | Import `slack_subcommand_map()`, delete hardcoded dict |
| `gateway/platforms/discord.py` | Add startup validation against registry |
| `tests/hermes_cli/test_commands.py` | Update to test registry, derived structures, helper functions |
| `tests/gateway/test_background_command.py` | Simplify — no more source-code-inspection tests |

## Bugfixes included for free

1. **Telegram missing commands**: `/rollback`, `/background`, `/bg` automatically added
2. **Slack missing commands**: `/voice`, `/update`, `/reload-mcp` automatically added
3. **Gateway duplicate "reasoning"**: Eliminated (generated from registry)
4. **Gateway dead code**: Second `"reasoning"` dispatch branch removed
5. **Help text drift**: Gateway help now generated from same source as CLI help

## What stays the same

- CLI dispatch remains an if/elif chain (readable, fast, explicit)
- Gateway dispatch remains an if chain
- Discord slash command decorators stay platform-specific
- Handler function signatures and locations don't change
- Quick commands and skill commands remain separate (config-driven / dynamic)

## Migration / backwards compat

- `COMMANDS` flat dict and `COMMANDS_BY_CATEGORY` dict are rebuilt from the
  registry, so any code importing them continues to work unchanged
- `SlashCommandCompleter` continues to read from `COMMANDS` dict
- No config changes, no user-facing behavior changes

## Risks

- **Import ordering**: `gateway/run.py` importing from `hermes_cli/commands.py` — 
  verify no circular import. Currently `gateway/run.py` doesn't import from
  `hermes_cli/` at all. Need to confirm this works or move the registry to a
  shared location (e.g., `commands_registry.py` at the top level).
- **Telegram command name sanitization**: Telegram doesn't allow hyphens in
  command names. The `telegram_bot_commands()` helper handles this with
  `.replace("-", "_")`, but the gateway dispatch must still accept both forms.
  Currently handled via the `("reload-mcp", "reload_mcp")` alias.

## Estimated scope

- ~200 lines of new code in `commands.py` (dataclass + registry + helpers)
- ~100 lines deleted across gateway/run.py, telegram.py, slack.py (hardcoded lists)
- ~50 lines changed in cli.py (alias resolution refactor)
- ~80 lines of new/updated tests
- Net: roughly even LOC, dramatically less maintenance surface
