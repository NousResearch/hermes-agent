"""Current-main Discord relay compatibility projection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .core import STRING_OPTION
from .model import DiscordCommandProjection, project_discord_commands


def _opt(
    name: str,
    description: str,
    *,
    choices: Sequence[str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": STRING_OPTION,
        "name": name,
        "description": description,
        "required": False,
    }
    if choices:
        row["choices"] = [
            {"name": choice, "value": choice}
            for choice in choices
        ]
    return row


def _relay_compatibility_payloads() -> list[dict[str, Any]]:
    """Return the accepted relay/native compatibility surface on current main."""
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
                _opt(
                    "scope",
                    "Approval scope",
                    choices=["once", "session", "always", "all"],
                )
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


def build_relay_discord_projection() -> DiscordCommandProjection:
    """Build the relay lane's immutable Discord command projection."""
    return project_discord_commands(_relay_compatibility_payloads())


def build_relay_discord_manifest() -> list[dict[str, Any]]:
    """Compatibility export used by the relay hello frame."""
    return build_relay_discord_projection().wire_commands()
