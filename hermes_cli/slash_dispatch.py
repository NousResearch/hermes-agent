"""Shared slash-command classification for CLI and gateway entrypoints."""

from __future__ import annotations

from dataclasses import dataclass


NONINTERACTIVE_SAFE_SLASH_COMMANDS = frozenset(
    {
        "goal",
        "closure",
        "help",
        "resume",
        "sessions",
        "status",
        "usage",
    }
)

_NONINTERACTIVE_GOAL_SUBCOMMANDS = frozenset(
    {
        "",
        "clear",
        "done",
        "pause",
        "resume",
        "show",
        "status",
        "stop",
        "unwait",
    }
)

_DEDICATED_COMMAND_HINTS = {
    "goal": "hermes goal <status|pause|resume|clear|complete> --session-id <id>",
    "closure": "hermes closure latest|resume [--session-id ID|--task-id ID]",
    "kanban": "hermes kanban <subcommand>",
    "model": "hermes model",
    "runtime": "hermes runtime <subcommand>",
    "sessions": "hermes sessions <subcommand>",
}


@dataclass(frozen=True)
class SlashClassification:
    text: str
    looks_like_command: bool
    command: str | None = None
    raw_args: str = ""
    canonical: str | None = None
    known: bool = False


def looks_like_slash_command(text: str | None) -> bool:
    """Return True for slash commands while leaving absolute paths alone."""
    if not text:
        return False
    stripped = str(text).strip()
    if not stripped.startswith("/"):
        return False
    first_word = stripped.split(maxsplit=1)[0]
    raw = first_word[1:]
    if not raw:
        return False
    if "@" in raw:
        raw = raw.split("@", 1)[0]
    return bool(raw) and "/" not in raw


def classify_slash_text(text: str | None) -> SlashClassification:
    value = "" if text is None else str(text)
    stripped = value.strip()
    if not looks_like_slash_command(stripped):
        return SlashClassification(text=value, looks_like_command=False)

    parts = stripped.split(maxsplit=1)
    command = parts[0][1:].lower()
    if "@" in command:
        command = command.split("@", 1)[0]
    raw_args = parts[1] if len(parts) > 1 else ""

    try:
        from hermes_cli.commands import resolve_command

        cmd_def = resolve_command(command) or resolve_command(command.replace("_", "-"))
    except Exception:
        cmd_def = None

    canonical = cmd_def.name if cmd_def is not None else command.replace("_", "-")
    return SlashClassification(
        text=value,
        looks_like_command=True,
        command=command,
        raw_args=raw_args,
        canonical=canonical,
        known=cmd_def is not None,
    )


def noninteractive_slash_error_message(info: SlashClassification) -> str:
    command = info.command or ""
    if not info.looks_like_command:
        return (
            "Expected a slash command beginning with '/'. "
            "Use --query for ordinary text."
        )
    if not info.known:
        return (
            f"Unknown command `/{command}`. Use interactive /help to see commands, "
            "or resend without the leading slash to send it as a regular message."
        )
    hint = _DEDICATED_COMMAND_HINTS.get(info.canonical or "")
    if hint:
        return (
            f"Command `/{command}` is not available through noninteractive "
            f"`hermes chat --query`. Use `{hint}` or run interactive chat."
        )
    return (
        f"Command `/{command}` is not available through noninteractive "
        "`hermes chat --query`. Run interactive chat for this slash command, "
        "or resend without the leading slash to send it as a regular message."
    )


def is_noninteractive_safe_slash(info: SlashClassification) -> bool:
    """Return True when a slash command is safe to execute without a REPL loop."""
    if not info.known or not info.canonical:
        return False
    if info.canonical == "goal":
        raw = info.raw_args.strip().lower()
        verb = raw.split(None, 1)[0] if raw else ""
        return verb in _NONINTERACTIVE_GOAL_SUBCOMMANDS or verb == "wait"
    return info.canonical in NONINTERACTIVE_SAFE_SLASH_COMMANDS
