"""Slash command definitions and autocomplete for the Hermes CLI.

Contains the shared built-in ``COMMANDS`` dict and ``SlashCommandCompleter``.
The completer can optionally include dynamic skill slash commands supplied by the
interactive CLI.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from prompt_toolkit.completion import Completer, Completion


# Commands organized by category for better help display
COMMANDS_BY_CATEGORY = {
    "Session": {
        "/new": "Start a new conversation (reset history)",
        "/reset": "Reset conversation only (keep screen)",
        "/clear": "Clear screen and reset conversation (fresh start)",
        "/history": "Show conversation history",
        "/save": "Save the current conversation",
        "/retry": "Retry the last message (resend to agent)",
        "/undo": "Remove the last user/assistant exchange",
        "/title": "Set a title for the current session (usage: /title My Session Name)",
        "/compress": "Manually compress conversation context (flush memories + summarize)",
        "/rollback": "List or restore filesystem checkpoints (usage: /rollback [number])",
        "/background": "Run a prompt in the background (usage: /background <prompt>)",
    },
    "Configuration": {
        "/config": "Show current configuration",
        "/model": "Show or change the current model",
        "/provider": "Show available providers and current provider",
        "/prompt": "View/set custom system prompt",
        "/personality": "Set a predefined personality",
        "/verbose": "Cycle tool progress display: off → new → all → verbose",
        "/skin": "Show or change the display skin/theme",
    },
    "Tools & Skills": {
        "/tools": "List available tools",
        "/toolsets": "List available toolsets",
        "/skills": "Search, install, inspect, or manage skills from online registries",
        "/cron": "Manage scheduled tasks (list, add, remove)",
        "/reload-mcp": "Reload MCP servers from config.yaml",
    },
    "Info": {
        "/help": "Show this help message",
        "/usage": "Show token usage for the current session",
        "/insights": "Show usage insights and analytics (last 30 days)",
        "/platforms": "Show gateway/messaging platform status",
        "/paste": "Check clipboard for an image and attach it",
    },
    "Exit": {
        "/quit": "Exit the CLI (also: /exit, /q)",
    },
}

# Flat dict for backwards compatibility and autocomplete
COMMANDS = {}
for category_commands in COMMANDS_BY_CATEGORY.values():
    COMMANDS.update(category_commands)


class SlashCommandCompleter(Completer):
    """Autocomplete for built-in slash commands and optional skill commands."""

    def __init__(
        self,
        skill_commands_provider: Callable[[], Mapping[str, dict[str, Any]]] | None = None,
    ) -> None:
        self._skill_commands_provider = skill_commands_provider

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

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
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
