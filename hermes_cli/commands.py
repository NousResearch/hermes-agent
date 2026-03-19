"""Slash command definitions and autocomplete for the Hermes CLI.

Central registry for all slash commands. Every consumer -- CLI help, gateway
dispatch, Telegram BotCommands, Slack subcommand mapping, autocomplete --
derives its data from ``COMMAND_REGISTRY``.

To add a command: add a ``CommandDef`` entry to ``COMMAND_REGISTRY``.
To add an alias: set ``aliases=("short",)`` on the existing ``CommandDef``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.completion import Completer, Completion


# ---------------------------------------------------------------------------
# CommandDef dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommandDef:
    """Definition of a single slash command."""

    name: str                          # canonical name without slash: "background"
    description: str                   # human-readable description
    category: str                      # "Session", "Configuration", etc.
    aliases: tuple[str, ...] = ()      # alternative names: ("bg",)
    args_hint: str = ""                # argument placeholder: "<prompt>", "[name]"
    subcommands: tuple[str, ...] = ()  # tab-completable subcommands
    cli_only: bool = False             # only available in CLI
    gateway_only: bool = False         # only available in gateway/messaging


# ---------------------------------------------------------------------------
# Central registry -- single source of truth
# ---------------------------------------------------------------------------

COMMAND_REGISTRY: list[CommandDef] = [
    # Session
    CommandDef("new", "Start a new session (fresh session ID + history)", "Session",
               aliases=("reset",)),
    CommandDef("clear", "Clear screen and start a new session", "Session",
               cli_only=True),
    CommandDef("history", "Show conversation history", "Session",
               cli_only=True),
    CommandDef("save", "Save the current conversation", "Session",
               cli_only=True),
    CommandDef("retry", "Retry the last message (resend to agent)", "Session"),
    CommandDef("undo", "Remove the last user/assistant exchange", "Session"),
    CommandDef("title", "Set a title for the current session", "Session",
               args_hint="[name]"),
    CommandDef("compress", "Manually compress conversation context", "Session"),
    CommandDef("rollback", "List or restore filesystem checkpoints", "Session",
               args_hint="[number]"),
    CommandDef("stop", "Kill all running background processes", "Session"),
    CommandDef("approve", "Approve a pending dangerous command", "Session",
               gateway_only=True, args_hint="[session|always]"),
    CommandDef("deny", "Deny a pending dangerous command", "Session",
               gateway_only=True),
    CommandDef("background", "Run a prompt in the background", "Session",
               aliases=("bg",), args_hint="<prompt>"),
    CommandDef("status", "Show session info", "Session",
               gateway_only=True),
    CommandDef("sethome", "Set this chat as the home channel", "Session",
               gateway_only=True, aliases=("set-home",)),
    CommandDef("resume", "Resume a previously-named session", "Session",
               args_hint="[name]"),

    # Configuration
    CommandDef("config", "Show current configuration", "Configuration",
               cli_only=True),
    CommandDef("model", "Show or change the current model", "Configuration",
               args_hint="[name]"),
    CommandDef("provider", "Show available providers and current provider",
               "Configuration"),
    CommandDef("prompt", "View/set custom system prompt", "Configuration",
               cli_only=True, args_hint="[text]", subcommands=("clear",)),
    CommandDef("personality", "Set a predefined personality", "Configuration",
               args_hint="[name]"),
    CommandDef("statusbar", "Toggle the context/model status bar", "Configuration",
               cli_only=True, aliases=("sb",)),
    CommandDef("verbose", "Cycle tool progress display: off -> new -> all -> verbose",
               "Configuration", cli_only=True),
    CommandDef("reasoning", "Manage reasoning effort and display", "Configuration",
               args_hint="[level|show|hide]",
               subcommands=("none", "low", "minimal", "medium", "high", "xhigh", "show", "hide", "on", "off")),
    CommandDef("skin", "Show or change the display skin/theme", "Configuration",
               cli_only=True, args_hint="[name]"),
    CommandDef("voice", "Toggle voice mode", "Configuration",
               args_hint="[on|off|tts|status]", subcommands=("on", "off", "tts", "status")),

    # Tools & Skills
    CommandDef("tools", "Manage tools: /tools [list|disable|enable] [name...]", "Tools & Skills",
               args_hint="[list|disable|enable] [name...]", cli_only=True),
    CommandDef("toolsets", "List available toolsets", "Tools & Skills",
               cli_only=True),
    CommandDef("skills", "Search, install, inspect, or manage skills",
               "Tools & Skills", cli_only=True,
               subcommands=("search", "browse", "inspect", "install")),
    CommandDef("cron", "Manage scheduled tasks", "Tools & Skills",
               cli_only=True, args_hint="[subcommand]",
               subcommands=("list", "add", "create", "edit", "pause", "resume", "run", "remove")),
    CommandDef("reload-mcp", "Reload MCP servers from config", "Tools & Skills",
               aliases=("reload_mcp",)),
    CommandDef("browser", "Connect browser tools to your live Chrome via CDP", "Tools & Skills",
               cli_only=True, args_hint="[connect|disconnect|status]",
               subcommands=("connect", "disconnect", "status")),
    CommandDef("plugins", "List installed plugins and their status",
               "Tools & Skills", cli_only=True),

    # Info
    CommandDef("help", "Show available commands", "Info"),
    CommandDef("usage", "Show token usage for the current session", "Info"),
    CommandDef("insights", "Show usage insights and analytics", "Info",
               args_hint="[days]"),
    CommandDef("platforms", "Show gateway/messaging platform status", "Info",
               cli_only=True, aliases=("gateway",)),
    CommandDef("paste", "Check clipboard for an image and attach it", "Info",
               cli_only=True),
    CommandDef("update", "Update Hermes Agent to the latest version", "Info",
               gateway_only=True),

    # Exit
    CommandDef("quit", "Exit the CLI", "Exit",
               cli_only=True, aliases=("exit", "q")),
]


# ---------------------------------------------------------------------------
# Derived lookups -- rebuilt once at import time
# ---------------------------------------------------------------------------

def _build_command_lookup() -> dict[str, CommandDef]:
    """Map every name and alias to its CommandDef."""
    lookup: dict[str, CommandDef] = {}
    for cmd in COMMAND_REGISTRY:
        lookup[cmd.name] = cmd
        for alias in cmd.aliases:
            lookup[alias] = cmd
    return lookup


_COMMAND_LOOKUP: dict[str, CommandDef] = _build_command_lookup()


def resolve_command(name: str) -> CommandDef | None:
    """Resolve a command name or alias to its CommandDef.

    Accepts names with or without the leading slash.
    """
    return _COMMAND_LOOKUP.get(name.lower().lstrip("/"))


def _build_description(cmd: CommandDef) -> str:
    """Build a CLI-facing description string including usage hint."""
    if cmd.args_hint:
        return f"{cmd.description} (usage: /{cmd.name} {cmd.args_hint})"
    return cmd.description


# Backwards-compatible flat dict: "/command" -> description
COMMANDS: dict[str, str] = {}
for _cmd in COMMAND_REGISTRY:
    if not _cmd.gateway_only:
        COMMANDS[f"/{_cmd.name}"] = _build_description(_cmd)
        for _alias in _cmd.aliases:
            COMMANDS[f"/{_alias}"] = f"{_cmd.description} (alias for /{_cmd.name})"

# Backwards-compatible categorized dict
COMMANDS_BY_CATEGORY: dict[str, dict[str, str]] = {}
for _cmd in COMMAND_REGISTRY:
    if not _cmd.gateway_only:
        _cat = COMMANDS_BY_CATEGORY.setdefault(_cmd.category, {})
        _cat[f"/{_cmd.name}"] = COMMANDS[f"/{_cmd.name}"]
        for _alias in _cmd.aliases:
            _cat[f"/{_alias}"] = COMMANDS[f"/{_alias}"]


# Subcommands lookup: "/cmd" -> ["sub1", "sub2", ...]
SUBCOMMANDS: dict[str, list[str]] = {}
for _cmd in COMMAND_REGISTRY:
    if _cmd.subcommands:
        SUBCOMMANDS[f"/{_cmd.name}"] = list(_cmd.subcommands)

# Also extract subcommands hinted in args_hint via pipe-separated patterns
# e.g. args_hint="[on|off|tts|status]" for commands that don't have explicit subcommands.
# NOTE: If a command already has explicit subcommands, this fallback is skipped.
# Use the `subcommands` field on CommandDef for intentional tab-completable args.
_PIPE_SUBS_RE = re.compile(r"[a-z]+(?:\|[a-z]+)+")
for _cmd in COMMAND_REGISTRY:
    key = f"/{_cmd.name}"
    if key in SUBCOMMANDS or not _cmd.args_hint:
        continue
    m = _PIPE_SUBS_RE.search(_cmd.args_hint)
    if m:
        SUBCOMMANDS[key] = m.group(0).split("|")


# ---------------------------------------------------------------------------
# Gateway helpers
# ---------------------------------------------------------------------------

# Set of all command names + aliases recognized by the gateway
GATEWAY_KNOWN_COMMANDS: frozenset[str] = frozenset(
    name
    for cmd in COMMAND_REGISTRY
    if not cmd.cli_only
    for name in (cmd.name, *cmd.aliases)
)


def gateway_help_lines() -> list[str]:
    """Generate gateway help text lines from the registry."""
    lines: list[str] = []
    for cmd in COMMAND_REGISTRY:
        if cmd.cli_only:
            continue
        args = f" {cmd.args_hint}" if cmd.args_hint else ""
        alias_parts: list[str] = []
        for a in cmd.aliases:
            # Skip internal aliases like reload_mcp (underscore variant)
            if a.replace("-", "_") == cmd.name.replace("-", "_") and a != cmd.name:
                continue
            alias_parts.append(f"`/{a}`")
        alias_note = f" (alias: {', '.join(alias_parts)})" if alias_parts else ""
        lines.append(f"`/{cmd.name}{args}` -- {cmd.description}{alias_note}")
    return lines


def telegram_bot_commands() -> list[tuple[str, str]]:
    """Return (command_name, description) pairs for Telegram setMyCommands.

    Telegram command names cannot contain hyphens, so they are replaced with
    underscores.  Aliases are skipped -- Telegram shows one menu entry per
    canonical command.
    """
    result: list[tuple[str, str]] = []
    for cmd in COMMAND_REGISTRY:
        if cmd.cli_only:
            continue
        tg_name = cmd.name.replace("-", "_")
        result.append((tg_name, cmd.description))
    return result


def slack_subcommand_map() -> dict[str, str]:
    """Return subcommand -> /command mapping for Slack /hermes handler.

    Maps both canonical names and aliases so /hermes bg do stuff works
    the same as /hermes background do stuff.
    """
    mapping: dict[str, str] = {}
    for cmd in COMMAND_REGISTRY:
        if cmd.cli_only:
            continue
        mapping[cmd.name] = f"/{cmd.name}"
        for alias in cmd.aliases:
            mapping[alias] = f"/{alias}"
    return mapping


# ---------------------------------------------------------------------------
# Autocomplete
# ---------------------------------------------------------------------------

class SlashCommandCompleter(Completer):
    """Autocomplete for built-in slash commands, subcommands, and skill commands."""

    def __init__(
        self,
        skill_commands_provider: Callable[[], Mapping[str, dict[str, Any]]] | None = None,
        model_completer_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._skill_commands_provider = skill_commands_provider
        # model_completer_provider returns {"current_provider": str,
        #   "providers": {id: label, ...}, "models_for": callable(provider) -> list[str]}
        self._model_completer_provider = model_completer_provider
        self._model_info_cache: dict[str, Any] | None = None
        self._model_info_cache_time: float = 0

    def _get_model_info(self) -> dict[str, Any]:
        """Get cached model/provider info for /model autocomplete."""
        import time
        now = time.monotonic()
        if self._model_info_cache is not None and now - self._model_info_cache_time < 60:
            return self._model_info_cache
        if self._model_completer_provider is None:
            return {}
        try:
            self._model_info_cache = self._model_completer_provider() or {}
            self._model_info_cache_time = now
        except Exception:
            self._model_info_cache = self._model_info_cache or {}
        return self._model_info_cache

    def _iter_skill_commands(self) -> Mapping[str, dict[str, Any]]:
        if self._skill_commands_provider is None:
            return {}
        try:
            return self._skill_commands_provider() or {}
        except Exception:
            return {}

    @staticmethod
    def _completion_text(cmd_name: str, word: str) -> str:
        """Return replacement text for a completion.

        When the user has already typed the full command exactly (``/help``),
        returning ``help`` would be a no-op and prompt_toolkit suppresses the
        menu. Appending a trailing space keeps the dropdown visible and makes
        backspacing retrigger it naturally.
        """
        return f"{cmd_name} " if cmd_name == word else cmd_name

    @staticmethod
    def _extract_path_word(text: str) -> str | None:
        """Extract the current word if it looks like a file path.

        Returns the path-like token under the cursor, or None if the
        current word doesn't look like a path.  A word is path-like when
        it starts with ``./``, ``../``, ``~/``, ``/``, or contains a
        ``/`` separator (e.g. ``src/main.py``).
        """
        if not text:
            return None
        # Walk backwards to find the start of the current "word".
        # Words are delimited by spaces, but paths can contain almost anything.
        i = len(text) - 1
        while i >= 0 and text[i] != " ":
            i -= 1
        word = text[i + 1:]
        if not word:
            return None
        # Only trigger path completion for path-like tokens
        if word.startswith(("./", "../", "~/", "/")) or "/" in word:
            return word
        return None

    @staticmethod
    def _path_completions(word: str, limit: int = 30):
        """Yield Completion objects for file paths matching *word*."""
        expanded = os.path.expanduser(word)
        # Split into directory part and prefix to match inside it
        if expanded.endswith("/"):
            search_dir = expanded
            prefix = ""
        else:
            search_dir = os.path.dirname(expanded) or "."
            prefix = os.path.basename(expanded)

        try:
            entries = os.listdir(search_dir)
        except OSError:
            return

        count = 0
        prefix_lower = prefix.lower()
        for entry in sorted(entries):
            if prefix and not entry.lower().startswith(prefix_lower):
                continue
            if count >= limit:
                break

            full_path = os.path.join(search_dir, entry)
            is_dir = os.path.isdir(full_path)

            # Build the completion text (what replaces the typed word)
            if word.startswith("~"):
                display_path = "~/" + os.path.relpath(full_path, os.path.expanduser("~"))
            elif os.path.isabs(word):
                display_path = full_path
            else:
                # Keep relative
                display_path = os.path.relpath(full_path)

            if is_dir:
                display_path += "/"

            suffix = "/" if is_dir else ""
            meta = "dir" if is_dir else _file_size_label(full_path)

            yield Completion(
                display_path,
                start_position=-len(word),
                display=entry + suffix,
                display_meta=meta,
            )
            count += 1

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            # Try file path completion for non-slash input
            path_word = self._extract_path_word(text)
            if path_word is not None:
                yield from self._path_completions(path_word)
            return

        # Check if we're completing a subcommand (base command already typed)
        parts = text.split(maxsplit=1)
        base_cmd = parts[0].lower()
        if len(parts) > 1 or (len(parts) == 1 and text.endswith(" ")):
            sub_text = parts[1] if len(parts) > 1 else ""
            sub_lower = sub_text.lower()

            # /model gets two-stage completion:
            #   Stage 1: provider names (with : suffix)
            #   Stage 2: after "provider:", list that provider's models
            if base_cmd == "/model" and " " not in sub_text:
                info = self._get_model_info()
                if info:
                    current_prov = info.get("current_provider", "")
                    providers = info.get("providers", {})
                    models_for = info.get("models_for")

                    if ":" in sub_text:
                        # Stage 2: "anthropic:cl" → models for anthropic
                        prov_part, model_part = sub_text.split(":", 1)
                        model_lower = model_part.lower()
                        if models_for:
                            try:
                                prov_models = models_for(prov_part)
                            except Exception:
                                prov_models = []
                            for mid in prov_models:
                                if mid.lower().startswith(model_lower) and mid.lower() != model_lower:
                                    full = f"{prov_part}:{mid}"
                                    yield Completion(
                                        full,
                                        start_position=-len(sub_text),
                                        display=mid,
                                    )
                    else:
                        # Stage 1: providers sorted: non-current first, current last
                        for pid, plabel in sorted(
                            providers.items(),
                            key=lambda kv: (kv[0] == current_prov, kv[0]),
                        ):
                            display_name = f"{pid}:"
                            if display_name.lower().startswith(sub_lower):
                                meta = f"({plabel})" if plabel != pid else ""
                                if pid == current_prov:
                                    meta = f"(current — {plabel})" if plabel != pid else "(current)"
                                yield Completion(
                                    display_name,
                                    start_position=-len(sub_text),
                                    display=display_name,
                                    display_meta=meta,
                                )
                return

            # Static subcommand completions
            if " " not in sub_text and base_cmd in SUBCOMMANDS:
                for sub in SUBCOMMANDS[base_cmd]:
                    if sub.startswith(sub_lower) and sub != sub_lower:
                        yield Completion(
                            sub,
                            start_position=-len(sub_text),
                            display=sub,
                        )
            return

        word = text[1:]

        for cmd, desc in COMMANDS.items():
            cmd_name = cmd[1:]
            if cmd_name.startswith(word):
                yield Completion(
                    self._completion_text(cmd_name, word),
                    start_position=-len(word),
                    display=cmd,
                    display_meta=desc,
                )

        for cmd, info in self._iter_skill_commands().items():
            cmd_name = cmd[1:]
            if cmd_name.startswith(word):
                description = str(info.get("description", "Skill command"))
                short_desc = description[:50] + ("..." if len(description) > 50 else "")
                yield Completion(
                    self._completion_text(cmd_name, word),
                    start_position=-len(word),
                    display=cmd,
                    display_meta=f"⚡ {short_desc}",
                )


# ---------------------------------------------------------------------------
# Inline auto-suggest (ghost text) for slash commands
# ---------------------------------------------------------------------------

class SlashCommandAutoSuggest(AutoSuggest):
    """Inline ghost-text suggestions for slash commands and their subcommands.

    Shows the rest of a command or subcommand in dim text as you type.
    Falls back to history-based suggestions for non-slash input.
    """

    def __init__(
        self,
        history_suggest: AutoSuggest | None = None,
        completer: SlashCommandCompleter | None = None,
    ) -> None:
        self._history = history_suggest
        self._completer = completer  # Reuse its model cache

    def get_suggestion(self, buffer, document):
        text = document.text_before_cursor

        # Only suggest for slash commands
        if not text.startswith("/"):
            # Fall back to history for regular text
            if self._history:
                return self._history.get_suggestion(buffer, document)
            return None

        parts = text.split(maxsplit=1)
        base_cmd = parts[0].lower()

        if len(parts) == 1 and not text.endswith(" "):
            # Still typing the command name: /upd → suggest "ate"
            word = text[1:].lower()
            for cmd in COMMANDS:
                cmd_name = cmd[1:]  # strip leading /
                if cmd_name.startswith(word) and cmd_name != word:
                    return Suggestion(cmd_name[len(word):])
            return None

        # Command is complete — suggest subcommands or model names
        sub_text = parts[1] if len(parts) > 1 else ""
        sub_lower = sub_text.lower()

        # /model gets two-stage ghost text
        if base_cmd == "/model" and " " not in sub_text and self._completer:
            info = self._completer._get_model_info()
            if info:
                providers = info.get("providers", {})
                models_for = info.get("models_for")
                current_prov = info.get("current_provider", "")

                if ":" in sub_text:
                    # Stage 2: after provider:, suggest model
                    prov_part, model_part = sub_text.split(":", 1)
                    model_lower = model_part.lower()
                    if models_for:
                        try:
                            for mid in models_for(prov_part):
                                if mid.lower().startswith(model_lower) and mid.lower() != model_lower:
                                    return Suggestion(mid[len(model_part):])
                        except Exception:
                            pass
                else:
                    # Stage 1: suggest provider name with :
                    for pid in sorted(providers, key=lambda p: (p == current_prov, p)):
                        candidate = f"{pid}:"
                        if candidate.lower().startswith(sub_lower) and candidate.lower() != sub_lower:
                            return Suggestion(candidate[len(sub_text):])

        # Static subcommands
        if base_cmd in SUBCOMMANDS and SUBCOMMANDS[base_cmd]:
            if " " not in sub_text:
                for sub in SUBCOMMANDS[base_cmd]:
                    if sub.startswith(sub_lower) and sub != sub_lower:
                        return Suggestion(sub[len(sub_text):])

        # Fall back to history
        if self._history:
            return self._history.get_suggestion(buffer, document)
        return None


def _file_size_label(path: str) -> str:
    """Return a compact human-readable file size, or '' on error."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return ""
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f}K"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f}M"
    return f"{size / (1024 * 1024 * 1024):.1f}G"
