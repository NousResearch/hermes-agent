"""Slash command definitions and autocomplete for the Hermes CLI.

Contains the COMMANDS dict and the SlashCommandCompleter class.
These are pure data/UI with no HermesCLI state dependency.
"""

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
    },
    "Configuration": {
        "/config": "Show current configuration",
        "/model": "Show or change the current model",
        "/prompt": "View/set custom system prompt",
        "/personality": "Set a predefined personality",
    },
    "Tools & Skills": {
        "/tools": "List available tools",
        "/toolsets": "List available toolsets",
        "/skills": "Search, install, inspect, or manage skills",
        "/cron": "Manage scheduled tasks (list, add, remove)",
    },
    "Info": {
        "/help": "Show this help message",
        "/platforms": "Show gateway/messaging platform status",
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
    """Autocomplete for /commands in the input area."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        word = text[1:]
        for cmd, desc in COMMANDS.items():
            cmd_name = cmd[1:]
            if cmd_name.startswith(word):
                yield Completion(
                    cmd_name,
                    start_position=-len(word),
                    display=cmd,
                    display_meta=desc,
                )
