"""Gateway-declared Discord slash-command manifest for the relay lane (Phase 4).

The CONNECTOR holds the Discord token, so the gateway declares its command set on
the ``hello`` frame and the connector reconciles Discord's registration (idempotent,
best-effort). MIRRORS the native tree (plugins/platforms/discord/adapter.py
``_register_slash_commands``) — same names, same descriptions; interactions return
via the passthrough plane as ordinary "/name args" COMMAND events, so a new entry
needs NO new handler. Wire shape per entry: {name, description, options?} with
Discord option objects verbatim; names must match ``[a-z0-9_-]{1,32}`` (the
connector drops invalid entries, never the whole manifest).
"""

from __future__ import annotations

from typing import Any, Dict, List

# Discord option type 3 = STRING.
_STR = 3


def _opt(name: str, description: str, *, choices: List[str] | None = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "type": _STR,
        "name": name,
        "description": description,
        "required": False,
    }
    if choices:
        row["choices"] = [{"name": c, "value": c} for c in choices]
    return row


def _cmd(name: str, description: str, *options: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"name": name, "description": description}
    if options:
        row["options"] = list(options)
    return row


_REASONING_CHOICES = [
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
    "reset", "show", "hide",
]


def build_relay_command_manifest() -> List[Dict[str, Any]]:
    """The relay lane's Discord slash-command manifest (native-tree mirror)."""
    return [
        _cmd("new", "Start a new conversation"),
        _cmd("reset", "Reset your Hermes session"),
        _cmd("model", "Show or change the model",
             _opt("name", "Model name. Leave empty to see current.")),
        _cmd("reasoning", "Show/change reasoning effort, or toggle showing it",
             _opt("effort", "Level, reset, or show/hide. Leave empty to see current.",
                  choices=_REASONING_CHOICES)),
        _cmd("personality", "Set a personality",
             _opt("name", "Personality name. Leave empty to list.")),
        _cmd("retry", "Retry your last message"),
        _cmd("undo", "Remove the last exchange"),
        _cmd("status", "Show Hermes session status"),
        _cmd("sethome", "Set this chat as the home channel"),
        _cmd("stop", "Stop the running Hermes agent"),
        _cmd("steer", "Inject a message after the next tool call (no interrupt)",
             _opt("text", "What to tell the agent")),
        _cmd("compress", "Compress conversation context"),
        _cmd("title", "Set or show the session title",
             _opt("text", "New title. Leave empty to show.")),
        _cmd("resume", "Resume a previously-named session",
             _opt("name", "Session title or id")),
        _cmd("usage", "Show token usage for this session"),
        _cmd("help", "Show available commands"),
        _cmd("insights", "Show usage insights and analytics"),
        _cmd("reload-mcp", "Reload MCP servers from config"),
        _cmd("reload-skills", "Re-scan skills for new or removed entries"),
        _cmd("voice", "Toggle voice reply mode"),
        _cmd("update", "Update Hermes Agent to the latest version"),
        _cmd("restart", "Gracefully restart the Hermes gateway"),
        _cmd("approve", "Approve a pending dangerous command",
             _opt("scope", "Approval scope", choices=["once", "session", "always", "all"])),
        _cmd("deny", "Deny a pending dangerous command",
             _opt("reason", "Why (relayed to the agent)")),
        _cmd("thread", "Create a new thread and start a Hermes session in it",
             _opt("name", "Thread name")),
        _cmd("queue", "Queue a prompt for the next turn (doesn't interrupt)",
             _opt("text", "The prompt to queue")),
        _cmd("bg", "Run a prompt in a separate background session",
             _opt("text", "The prompt to run")),
        _cmd("btw", "Ask a side question about the current conversation",
             _opt("text", "The question to answer")),
    ]
