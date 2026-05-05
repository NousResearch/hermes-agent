"""Slash command helpers for the multitenancy plugin.

The router consumes Feishu messages before Hermes' normal gateway dispatcher,
so slash commands must be recognized here. Recognition is intentionally
derived from Hermes' central command registry when available; the small
fallback only keeps the plugin testable outside a Hermes checkout.
"""
from __future__ import annotations

from typing import Optional


_FALLBACK_ALIASES = {
    "provider": "model",
    "reset": "new",
    "bg": "background",
    "btw": "background",
    "tasks": "agents",
    "q": "queue",
    "fork": "branch",
    "set-home": "sethome",
    "reload_mcp": "reload-mcp",
}

_FALLBACK_GATEWAY_COMMANDS = frozenset(
    {
        "agents",
        "approve",
        "background",
        "branch",
        "commands",
        "compress",
        "debug",
        "deny",
        "fast",
        "help",
        "insights",
        "model",
        "new",
        "personality",
        "profile",
        "queue",
        "reasoning",
        "reload-mcp",
        "restart",
        "resume",
        "retry",
        "rollback",
        "sethome",
        "status",
        "steer",
        "stop",
        "title",
        "undo",
        "update",
        "usage",
        "verbose",
        "voice",
        "yolo",
    }
)


def parse_command(text: str) -> Optional[tuple[str, str]]:
    """Parse a slash command. Returns ``(canonical_command, args)`` or ``None``.

    Unknown slash commands are still returned so the router can send Hermes'
    native unknown-command guidance instead of leaking ``/unknown`` into the
    LLM as a normal prompt.
    """
    if not text or not text.startswith("/"):
        return None
    parts = text.split(maxsplit=1)
    raw = parts[0][1:].lower()
    args = parts[1] if len(parts) > 1 else ""
    # Reject paths and known-bad shapes
    if "/" in raw or not raw:
        return None
    canonical = resolve_command_name(raw) or raw
    return (canonical, args)


def resolve_command_name(name: str) -> Optional[str]:
    """Return Hermes' canonical command name when ``name`` is gateway-known."""
    raw = name.lower().lstrip("/")
    hyphenated = raw.replace("_", "-")
    try:
        from hermes_cli.commands import (  # type: ignore
            is_gateway_known_command,
            resolve_command,
        )

        cmd_def = resolve_command(raw) or resolve_command(hyphenated)
        if cmd_def is not None:
            canonical = getattr(cmd_def, "name", raw)
            if is_gateway_known_command(raw) or is_gateway_known_command(hyphenated) or is_gateway_known_command(canonical):
                return str(canonical)
        if is_gateway_known_command(raw):
            return raw
        if is_gateway_known_command(hyphenated):
            return hyphenated
    except Exception:
        pass

    canonical = _FALLBACK_ALIASES.get(raw) or _FALLBACK_ALIASES.get(hyphenated) or hyphenated
    if canonical in _FALLBACK_GATEWAY_COMMANDS:
        return canonical
    return None


def is_known_command(name: str) -> bool:
    """Return True when ``name`` is recognized by Hermes' gateway command set."""
    return resolve_command_name(name) is not None


def unknown_command_message(command: str) -> str:
    """Match Hermes' gateway wording for unknown slash commands."""
    raw = command.lower().lstrip("/")
    return (
        f"Unknown command `/{raw}`. "
        "Type /commands to see what's available, "
        "or resend without the leading slash to send as a regular message."
    )
