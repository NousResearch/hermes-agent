"""Gateway-declared slash-command manifest for the relay lane (Phase 4).

The native Discord adapter registers its slash commands directly on the
Discord command tree (`_register_slash_commands`,
plugins/platforms/discord/adapter.py) — it holds the bot token. Over the
relay the CONNECTOR holds the token, so the gateway DECLARES the same
command set on its `hello` frame (`command_manifest`) and the connector
reconciles Discord's global application-command registration against it
(gateway-gateway `DiscordCommandRegistrar`: GET → diff → bulk PUT,
idempotent, best-effort).

This module is that declaration: the single source of truth for what the
relay lane advertises. It MIRRORS the native tree — same names, same
descriptions — so a user moving between a native-Discord deployment and a
hosted/relay one sees the same command palette. Interactions come back over
the passthrough plane and are normalized by
RelayAdapter._discord_interaction_to_event into the same "/name args"
COMMAND events the dispatcher already routes, so declaring a command here
requires NO new handler — the dispatcher's existing slash surface is the
handler.

Wire shape (per entry): {name, description, options?} where options rows are
Discord option objects passed through verbatim. Names must satisfy
Discord's CHAT_INPUT rules ([a-z0-9_-]{1,32}); the connector drops invalid
entries (fail-open per entry, never the whole manifest).
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


def build_relay_command_manifest() -> List[Dict[str, Any]]:
    """The relay lane's Discord slash-command manifest (native-tree mirror)."""
    return [
        {"name": "new", "description": "Start a new conversation"},
        {"name": "reset", "description": "Reset your Hermes session"},
        {
            "name": "model",
            "description": "Show or change the model",
            "options": [_opt("name", "Model name. Leave empty to see current.")],
        },
        {
            "name": "reasoning",
            "description": "Show/change reasoning effort, or toggle showing it",
            "options": [
                _opt(
                    "effort",
                    "Level, reset, or show/hide. Leave empty to see current.",
                    choices=[
                        "none",
                        "minimal",
                        "low",
                        "medium",
                        "high",
                        "xhigh",
                        "max",
                        "ultra",
                        "reset",
                        "show",
                        "hide",
                    ],
                )
            ],
        },
        {
            "name": "personality",
            "description": "Set a personality",
            "options": [_opt("name", "Personality name. Leave empty to list.")],
        },
        {"name": "retry", "description": "Retry your last message"},
        {"name": "undo", "description": "Remove the last exchange"},
        {"name": "status", "description": "Show Hermes session status"},
        {"name": "sethome", "description": "Set this chat as the home channel"},
        {"name": "stop", "description": "Stop the running Hermes agent"},
        {
            "name": "steer",
            "description": "Inject a message after the next tool call (no interrupt)",
            "options": [_opt("text", "What to tell the agent")],
        },
        {"name": "compress", "description": "Compress conversation context"},
        {
            "name": "title",
            "description": "Set or show the session title",
            "options": [_opt("text", "New title. Leave empty to show.")],
        },
        {
            "name": "resume",
            "description": "Resume a previously-named session",
            "options": [_opt("name", "Session title or id")],
        },
        {"name": "usage", "description": "Show token usage for this session"},
        {"name": "help", "description": "Show available commands"},
        {"name": "insights", "description": "Show usage insights and analytics"},
        {"name": "reload-mcp", "description": "Reload MCP servers from config"},
        {
            "name": "reload-skills",
            "description": "Re-scan skills for new or removed entries",
        },
        {"name": "voice", "description": "Toggle voice reply mode"},
        {"name": "update", "description": "Update Hermes Agent to the latest version"},
        {"name": "restart", "description": "Gracefully restart the Hermes gateway"},
        {
            "name": "approve",
            "description": "Approve a pending dangerous command",
            "options": [
                _opt("scope", "Approval scope", choices=["once", "session", "always", "all"])
            ],
        },
        {
            "name": "deny",
            "description": "Deny a pending dangerous command",
            "options": [_opt("reason", "Why (relayed to the agent)")],
        },
        {
            "name": "thread",
            "description": "Create a new thread and start a Hermes session in it",
            "options": [_opt("name", "Thread name")],
        },
        {
            "name": "queue",
            "description": "Queue a prompt for the next turn (doesn't interrupt)",
            "options": [_opt("text", "The prompt to queue")],
        },
        {
            "name": "bg",
            "description": "Run a prompt in a separate background session",
            "options": [_opt("text", "The prompt to run")],
        },
        {
            "name": "btw",
            "description": "Ask a side question about the current conversation",
            "options": [_opt("text", "The question to answer")],
        },
    ]
