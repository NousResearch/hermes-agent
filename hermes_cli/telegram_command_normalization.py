"""Typed Telegram ``/command@bot`` normalization for the command dispatcher."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from hermes_cli.telegram_command_projection import (
    TelegramCommandProjection,
    resolve_telegram_command_binding,
)


_TELEGRAM_INVOCATION_RE = re.compile(
    r"^/(?P<name>[A-Za-z0-9_][A-Za-z0-9_-]*)"
    r"(?:@(?P<bot>[A-Za-z0-9_]+))?"
    r"(?P<tail>\s.*)?$",
    re.DOTALL,
)


class TelegramCommandAttemptStatus(str, Enum):
    """Typed classification of one Telegram text input."""

    NOT_COMMAND = "not_command"
    KNOWN_COMMAND = "known_command"
    UNKNOWN_COMMAND = "unknown_command"
    NOT_FOR_THIS_BOT = "not_for_this_bot"
    INVALID_COMMAND = "invalid_command"


@dataclass(frozen=True, slots=True)
class TelegramCommandAttempt:
    """Normalized Telegram text classification and canonical binding."""

    status: TelegramCommandAttemptStatus
    raw_input: str
    entered_name: str | None = None
    addressed_bot: str | None = None
    raw_arguments: str = ""
    command_id: str | None = None
    canonical_name: str | None = None
    canonical_input: str | None = None


def _nonblank_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _not_command_slash_text(text: str) -> bool:
    token = text.split(None, 1)[0]
    if token.startswith("//"):
        return True
    remainder = token[1:]
    return any(character in remainder for character in ("/", "\\", ".", ":"))


def normalize_telegram_command_attempt(
    text: object,
    projection: TelegramCommandProjection,
    *,
    bot_username: str | None = None,
) -> TelegramCommandAttempt:
    """Classify and normalize one Telegram text without prompt fallthrough.

    A matching ``/command@bot args`` and ``/command args`` produce the same
    canonical identity, arguments, and dispatcher input. Unknown inputs that
    satisfy Telegram's command grammar remain typed ``unknown_command``
    attempts; slash-prefixed paths/code remain ordinary text.
    """

    if not isinstance(text, str):
        return TelegramCommandAttempt(
            status=TelegramCommandAttemptStatus.NOT_COMMAND,
            raw_input="" if text is None else str(text),
        )

    raw_input = text
    normalized = text.strip()
    if not normalized.startswith("/"):
        return TelegramCommandAttempt(
            status=TelegramCommandAttemptStatus.NOT_COMMAND,
            raw_input=raw_input,
        )

    match = _TELEGRAM_INVOCATION_RE.fullmatch(normalized)
    if match is None:
        status = (
            TelegramCommandAttemptStatus.NOT_COMMAND
            if _not_command_slash_text(normalized)
            else TelegramCommandAttemptStatus.INVALID_COMMAND
        )
        return TelegramCommandAttempt(status=status, raw_input=raw_input)

    entered_name = match.group("name")
    addressed_bot = match.group("bot")
    raw_arguments = (match.group("tail") or "").lstrip()

    if addressed_bot is not None:
        current_bot = _nonblank_text(bot_username)
        current_bot = current_bot.lstrip("@") if current_bot is not None else None
        if current_bot is None or addressed_bot.casefold() != current_bot.casefold():
            return TelegramCommandAttempt(
                status=TelegramCommandAttemptStatus.NOT_FOR_THIS_BOT,
                raw_input=raw_input,
                entered_name=entered_name,
                addressed_bot=addressed_bot,
                raw_arguments=raw_arguments,
            )

    binding = resolve_telegram_command_binding(projection, entered_name)
    if binding is None:
        canonical_input = f"/{entered_name}"
        if raw_arguments:
            canonical_input = f"{canonical_input} {raw_arguments}"
        return TelegramCommandAttempt(
            status=TelegramCommandAttemptStatus.UNKNOWN_COMMAND,
            raw_input=raw_input,
            entered_name=entered_name,
            addressed_bot=addressed_bot,
            raw_arguments=raw_arguments,
            canonical_input=canonical_input,
        )

    canonical_input = f"/{binding.canonical_name}"
    if raw_arguments:
        canonical_input = f"{canonical_input} {raw_arguments}"
    return TelegramCommandAttempt(
        status=TelegramCommandAttemptStatus.KNOWN_COMMAND,
        raw_input=raw_input,
        entered_name=entered_name,
        addressed_bot=addressed_bot,
        raw_arguments=raw_arguments,
        command_id=binding.command_id,
        canonical_name=binding.canonical_name,
        canonical_input=canonical_input,
    )
