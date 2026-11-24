"""Stable Discord slash command manifest for Hermes.

This manifest defines ALL native Discord slash commands with stable English descriptions.
It serves as the single source of truth for command registration, ensuring:
- Stable command fingerprints across restarts/model switches/language changes
- Version-controlled command definitions in Git
- Easy backup/restore of command state
- No dependency on runtime translation catalogs for command identity

The relay lane has its own manifest at gateway/relay/command_manifest.py.
This manifest covers the native commands registered by the main Discord adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Discord option types
_STR = 3
_INT = 4
_BOOL = 5


def _opt(
    name: str,
    description: str,
    *,
    required: bool = False,
    choices: Optional[List[Tuple[str, str]]] = None,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a Discord command option definition."""
    row: Dict[str, Any] = {
        "type": _STR,
        "name": name,
        "description": description,
        "required": required,
    }
    if choices:
        row["choices"] = [{"name": label, "value": value} for label, value in choices]
    if min_value is not None:
        row["min_value"] = min_value
    if max_value is not None:
        row["max_value"] = max_value
    if min_length is not None:
        row["min_length"] = min_length
    if max_length is not None:
        row["max_length"] = max_length
    return row


def _int_opt(
    name: str,
    description: str,
    *,
    required: bool = False,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> Dict[str, Any]:
    """Create an integer option."""
    row = _opt(name, description, required=required, min_value=min_value, max_value=max_value)
    row["type"] = _INT
    return row


def _cmd(
    name: str,
    description: str,
    *options: Dict[str, Any],
    template: Optional[str] = None,
    followup_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a Discord command definition."""
    row: Dict[str, Any] = {
        "name": name,
        "description": description,
    }
    if options:
        row["options"] = list(options)
    if template is not None:
        row["template"] = template
    if followup_key is not None:
        row["followup_key"] = followup_key
    return row


# Reasoning effort choices (stable identifiers, not translated)
_REASONING_CHOICES = [
    ("none", "none"),
    ("minimal", "minimal"),
    ("low", "low"),
    ("medium", "medium"),
    ("high", "high"),
    ("xhigh", "xhigh"),
    ("max", "max"),
    ("ultra", "ultra"),
    ("reset", "reset"),
    ("show", "show"),
    ("hide", "hide"),
]

# Voice mode choices
_VOICE_CHOICES = [
    ("join", "join"),
    ("channel", "channel"),
    ("leave", "leave"),
    ("on", "on"),
    ("tts", "tts"),
    ("off", "off"),
    ("status", "status"),
]

# Approve scope choices
_APPROVE_CHOICES = [
    ("once", "once"),
    ("session", "session"),
    ("always", "always"),
    ("all", "all"),
]


def build_native_command_manifest() -> List[Dict[str, Any]]:
    """Build the complete native Discord slash command manifest.

    Returns a list of command definitions with stable English descriptions.
    Each command has: name, description, options (optional), template, followup_key.
    """
    return [
        # Core session commands
        _cmd(
            "new",
            "Start a new conversation",
            template="/reset",
            followup_key="platform.discord.command.new.followup",
        ),
        _cmd(
            "reset",
            "Reset your Hermes session",
            template="/reset",
            followup_key="platform.discord.command.reset.followup",
        ),
        _cmd(
            "model",
            "Show or change the model",
            _opt("name", "Model name. Leave empty to see current."),
            template="/model {name}",
        ),
        _cmd(
            "reasoning",
            "Show/change reasoning effort, or toggle showing it",
            _opt("effort", "Level, reset, or show/hide. Leave empty to see current.", choices=_REASONING_CHOICES),
            template="/reasoning {effort}",
        ),
        _cmd(
            "personality",
            "Set a personality",
            _opt("name", "Personality name. Leave empty to list."),
            template="/personality {name}",
        ),
        _cmd(
            "retry",
            "Retry your last message",
            template="/retry",
            followup_key="platform.discord.command.retry.followup",
        ),
        _cmd(
            "undo",
            "Remove the last exchange",
            template="/undo",
        ),
        _cmd(
            "status",
            "Show Hermes session status",
            template="/status",
            followup_key="platform.discord.command.status.followup",
        ),
        _cmd(
            "sethome",
            "Set this chat as the home channel",
            template="/sethome",
        ),
        _cmd(
            "start",
            "Start a Hermes session in this channel",
            template="/start",
            followup_key="platform.discord.start.followup",
        ),
        _cmd(
            "stop",
            "Stop the running Hermes agent",
            template="/stop",
            followup_key="platform.discord.command.stop.followup",
        ),
        _cmd(
            "steer",
            "Inject a message after the next tool call (no interrupt)",
            _opt("prompt", "What to tell the agent", required=True),
            template="/steer {prompt}",
        ),
        _cmd(
            "plan",
            "Plan a task for the agent",
            _opt("task", "The task to plan"),
            template="/plan {task}",
        ),
        _cmd(
            "compress",
            "Compress conversation context",
            template="/compress",
        ),
        _cmd(
            "title",
            "Set or show the session title",
            _opt("name", "New title. Leave empty to show."),
            template="/title {name}",
        ),
        _cmd(
            "resume",
            "Resume a previously-named session",
            _opt("name", "Session title or id"),
            template="/resume {name}",
        ),
        _cmd(
            "usage",
            "Show token usage for this session",
            template="/usage",
        ),
        _cmd(
            "help",
            "Show available commands",
            template="/help",
        ),
        _cmd(
            "insights",
            "Show usage insights and analytics",
            _int_opt("days", "Number of days to analyze", min_value=1, max_value=365),
            template="/insights {days}",
        ),
        _cmd(
            "reload-mcp",
            "Reload MCP servers from config",
            template="/reload-mcp",
        ),
        _cmd(
            "reload-skills",
            "Re-scan skills for new or removed entries",
            template="/reload-skills",
        ),
        # /sesh: direct entry point for the sesh AFK worker skill
        _cmd(
            "sesh",
            "Run the sesh AFK worker across selected projects",
            template="/sesh",
        ),
        _cmd(
            "voice",
            "Toggle voice reply mode",
            _opt("mode", "Voice mode: join, channel, leave, on, tts, off, status", choices=_VOICE_CHOICES),
            template="/voice {mode}",
        ),
        # /vc: short alias for /voice
        _cmd(
            "vc",
            "Toggle voice reply mode (alias for /voice)",
            _opt("mode", "Voice mode: join, channel, leave, on, tts, off, status", choices=_VOICE_CHOICES),
            template="/voice {mode}",
        ),
        _cmd(
            "update",
            "Update Hermes Agent to the latest version",
            template="/update",
            followup_key="platform.discord.command.update.followup",
        ),
        _cmd(
            "restart",
            "Gracefully restart the Hermes gateway",
            template="/restart",
            followup_key="platform.discord.command.restart.followup",
        ),
        _cmd(
            "approve",
            "Approve a pending dangerous command",
            _opt("scope", "Approval scope", choices=_APPROVE_CHOICES),
            template="/approve {scope}",
        ),
        _cmd(
            "deny",
            "Deny a pending dangerous command",
            _opt("scope", "Denial scope"),
            template="/deny {scope}",
        ),
        # /thread: registered separately by _register_thread_slash (auth-gated defer)
        _cmd(
            "thread",
            "Create a new thread and start a Hermes session in it",
            _opt("name", "Thread name"),
        ),
        _cmd(
            "queue",
            "Queue a prompt for the next turn (doesn't interrupt)",
            _opt("prompt", "The prompt to queue", required=True),
            template="/queue {prompt}",
            followup_key="platform.discord.command.queue.followup",
        ),
        _cmd(
            "bg",
            "Run a prompt in a separate background session",
            _opt("prompt", "The prompt to run", required=True),
            template="/bg {prompt}",
            followup_key="platform.discord.command.bg.followup",
        ),
        _cmd(
            "btw",
            "Ask a side question about the current conversation",
            _opt("question", "The question to answer", required=True),
            template="/btw {question}",
            followup_key="platform.discord.command.btw.followup",
        ),
    ]


def get_command_names() -> List[str]:
    """Get all native command names from the manifest."""
    return [cmd["name"] for cmd in build_native_command_manifest()]


def get_command_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Get a command definition by name."""
    for cmd in build_native_command_manifest():
        if cmd["name"] == name:
            return cmd
    return None


# For backward compatibility with tests that import _NATIVE_SLASH_COMMAND_SPECS
# This provides the same tuple structure but with resolved English descriptions
def _build_legacy_specs() -> Tuple:
    """Build legacy _NATIVE_SLASH_COMMAND_SPECS format for compatibility."""
    specs = []
    for cmd in build_native_command_manifest():
        name = cmd["name"]
        description = cmd["description"]
        template = cmd.get("template")
        followup_key = cmd.get("followup_key")

        # Convert options to legacy format
        args = []
        for opt in cmd.get("options", []):
            arg_name = opt["name"]
            arg_type = str  # Default
            default = "" if not opt.get("required") else object()  # _REQUIRED placeholder
            desc_key = opt["description"]  # Use description directly as key for compatibility
            choices = None
            if "choices" in opt:
                choices = tuple((c["name"], c["value"]) for c in opt["choices"])
            args.append((arg_name, arg_type, default, desc_key, choices))

        specs.append((name, description, tuple(args), template, followup_key))
    return tuple(specs)


# Legacy format for compatibility - DO NOT USE FOR NEW CODE
_NATIVE_SLASH_COMMAND_SPECS: Tuple = _build_legacy_specs()


__all__ = [
    "build_native_command_manifest",
    "get_command_names",
    "get_command_by_name",
    "_NATIVE_SLASH_COMMAND_SPECS",  # Legacy compatibility
]