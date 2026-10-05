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

from agent.i18n import t
from gateway.platforms.base import _prefix_within_utf16_limit

# Discord option type 3 = STRING.
_STR = 3
# Discord rejects the whole bulk overwrite (error 50035) when ONE description exceeds 100 UTF-16
# units, so every localized description is cut at the cap AFTER translation.
_DISCORD_DESCRIPTION_LIMIT = 100


def _text(key: str) -> str:
    """``t(key)`` for the active language, cut to Discord's description cap."""
    return _prefix_within_utf16_limit(t(key), _DISCORD_DESCRIPTION_LIMIT)


def _opt(name: str, description_key: str, *, choices: List[str] | None = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "type": _STR,
        "name": name,
        "description": _text(description_key),
        "required": False,
    }
    if choices:
        row["choices"] = [{"name": c, "value": c} for c in choices]
    return row


def _cmd(name: str, description_key: str, *options: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"name": name, "description": _text(description_key)}
    if options:
        row["options"] = list(options)
    return row


# Choice names are command identifiers (name == value), never prose.
_REASONING_CHOICES = [
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
    "reset", "show", "hide",
]


def build_relay_command_manifest() -> List[Dict[str, Any]]:
    """The relay lane's Discord slash-command manifest (native-tree mirror), with descriptions
    resolved for the active language. Built per ``hello``, so a language change reaches Discord on
    the next dial: the connector's GET → diff → PUT sees the changed descriptions."""
    return [
        _cmd("new", "platform.discord.command.new.description"),
        _cmd("reset", "platform.discord.command.reset.description"),
        _cmd("model", "platform.discord.command.model.description",
             _opt("name", "platform.relay.command.model.arg_name")),
        _cmd("reasoning", "platform.discord.command.reasoning.description",
             _opt("effort", "platform.relay.command.reasoning.arg_effort", choices=_REASONING_CHOICES)),
        _cmd("personality", "platform.discord.command.personality.description",
             _opt("name", "platform.relay.command.personality.arg_name")),
        _cmd("retry", "platform.discord.command.retry.description"),
        _cmd("undo", "platform.discord.command.undo.description"),
        _cmd("status", "platform.discord.command.status.description"),
        _cmd("sethome", "slash.sethome.description"),
        _cmd("stop", "platform.discord.command.stop.description"),
        _cmd("steer", "platform.discord.command.steer.description",
             _opt("text", "platform.relay.command.steer.arg_text")),
        _cmd("compress", "platform.discord.command.compress.description"),
        _cmd("title", "platform.discord.command.title.description",
             _opt("text", "platform.relay.command.title.arg_text")),
        _cmd("resume", "slash.resume.description",
             _opt("name", "platform.relay.command.resume.arg_name")),
        _cmd("usage", "platform.discord.command.usage.description"),
        _cmd("help", "platform.discord.command.help.description"),
        _cmd("insights", "slash.insights.description"),
        _cmd("reload-mcp", "slash.reload_mcp.description"),
        _cmd("reload-skills", "platform.relay.command.reload_skills.description"),
        _cmd("voice", "platform.discord.command.voice.description"),
        _cmd("update", "slash.update.description"),
        _cmd("restart", "platform.discord.command.restart.description"),
        _cmd("approve", "slash.approve.description",
             _opt("scope", "platform.relay.command.approve.arg_scope", choices=["once", "session", "always", "all"])),
        _cmd("deny", "platform.discord.command.deny.description",
             _opt("reason", "platform.relay.command.deny.arg_reason")),
        _cmd("thread", "platform.discord.command.thread.description",
             _opt("name", "platform.discord.command.thread.arg_name")),
        _cmd("queue", "platform.discord.command.queue.description",
             _opt("text", "platform.discord.command.queue.arg_prompt")),
        _cmd("bg", "slash.bg.description",
             _opt("text", "platform.relay.command.bg.arg_text")),
        _cmd("btw", "platform.discord.command.btw.description",
             _opt("text", "platform.relay.command.btw.arg_text")),
    ]
