"""Who a Telegram group message is addressed to, kept out of the adapter facade.

``exclusive_bot_mentions`` drops a group message addressed to another bot. Only a LEADING run of
handles (``@BotA @BotB: do this``, ``/status@OtherBot``) addresses anyone: a foreign handle later
in the text talks about that bot and must not silence this one (#136236). Our own handle anywhere
still wins, unless the opening command explicitly targets another bot.

``/cmd@ourbot args`` is how Telegram's group command menu sends a command; stripping our handle
must keep the separator before the arguments (#56337), while prose cleanup stays as it was.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterator, Optional

from plugins.platforms.telegram.telegram_entities import addressing_sources

_SEPARATORS = " \t\n\r,:;"
_RAW_LEADING_HANDLE_RE = re.compile(r"(?:/[A-Za-z0-9_]+)?@([A-Za-z0-9_]{2,31})\b")


def _entity_type(entity: Any) -> str:
    return str(getattr(entity, "type", "")).split(".")[-1].lower()


def _utf16_to_str_offsets(text: str) -> list[int]:
    """``offsets[utf16_index] -> str index``; a surrogate pair's low half maps to the pair itself."""
    offsets: list[int] = []
    for index, ch in enumerate(text):
        offsets.append(index)
        if ord(ch) > 0xFFFF:
            offsets.append(index)
    offsets.append(len(text))
    return offsets


def _entity_handle(etype: str, span: str) -> Optional[str]:
    """``mention`` -> the handle; ``bot_command`` -> its ``@bot`` suffix (bare ``/cmd`` addresses no one)."""
    span = span.strip()
    if etype == "mention":
        return span.lstrip("@").lower()
    at = span.find("@")
    return span[at + 1:].strip().lower() if at >= 0 else None


def _leading_handles(text: str, entities: list) -> Iterator[str]:
    """Handles at the head of one text/caption, separated by whitespace/``,;:``. With entities a handle
    counts only where a ``mention``/``bot_command`` entity starts (UTF-16 offsets), so a handle in a
    code span addresses no one; without entities ``@handle`` / ``/cmd@handle`` is read from the text."""
    starts: dict[int, tuple[int, Optional[str]]] = {}
    if entities:
        offsets = _utf16_to_str_offsets(text)
        for entity in entities:
            etype = _entity_type(entity)
            off, length = int(getattr(entity, "offset", -1)), int(getattr(entity, "length", 0))
            if etype not in {"mention", "bot_command"} or off < 0 or length <= 0 or off + length >= len(offsets):
                continue
            start, end = offsets[off], offsets[off + length]
            starts.setdefault(start, (end, _entity_handle(etype, text[start:end])))
    pos = len(text) - len(text.lstrip())
    while pos < len(text):
        if entities:
            if pos not in starts:
                return
            pos, handle = starts[pos]
        else:
            match = _RAW_LEADING_HANDLE_RE.match(text, pos)
            if match is None:
                return
            pos, handle = match.end(), match.group(1).lower()
        if handle is None:
            return
        yield handle
        while pos < len(text) and text[pos] in _SEPARATORS:
            pos += 1


def leading_bot_handles(message: Any, is_bot_handle: Callable[[str], bool]) -> set[str]:
    """Bot handles the head of the message text, caption or rich blocks is addressed to."""
    found: set[str] = set()
    for text, entities in addressing_sources(message):
        found.update(h for h in _leading_handles(text, entities) if is_bot_handle(h))
    return found


def _command_after_address(text: str, bot_username: str, *, has_entities: bool = False) -> Optional[re.Match[str]]:
    return re.match(rf"(?i)\s*(?:@{re.escape(bot_username)}\b[,:-]*\s*)*"
                    r"(/[A-Za-z0-9_-]++)(?:@([A-Za-z0-9_]{1,32}))?"
                    + (r"(?![/\\])" if has_entities else r"(?=$|[\s,:;-])"), text)


def foreign_command_target(message: Any, bot_username: str) -> Optional[str]:
    """Only an opening command, optionally after our address, can override our later mention."""
    for text, entities in addressing_sources(message):
        command = _command_after_address(text, bot_username, has_entities=bool(entities))
        if not command or not command.group(2) or command.group(2).lower() == bot_username.lower():
            continue
        offset = len(text[:command.start(1)].encode("utf-16-le")) // 2
        if not entities or any(_entity_type(e) == "bot_command" and e.offset == offset
                               and (e.length == command.end() - command.start(1)
                                    or command.group(1)[e.length:e.length + 1] == "-") for e in entities):
            return command.group(2).lower()
    return None


def clean_bot_trigger_text(text: Optional[str], bot_username: str) -> Optional[str]:
    """Drop our ``@handle`` from trigger text. A leading ``/cmd@ourbot`` keeps one space before its
    arguments (``/reasoning@Bot medium`` -> ``/reasoning medium``), never gluing them to the command."""
    if not text or not bot_username:
        return text
    handle = re.escape(bot_username)
    command = _command_after_address(text, bot_username)
    if command and (not command.group(2) or command.group(2).lower() == bot_username.lower()):
        rest = text[command.end():]
        if command.group(2):
            rest = " " + re.sub(r"^[,:-]*\s*", "", rest)
        rest = re.sub(rf"(?i)^(\s*)(?:@{handle}\b[,:-]*\s*)+", r"\1", rest)
        rest = re.sub(rf"(?i)(?:@{handle}\b[,:-]*\s*+)+(?=$|[-–—]++\w)", "", rest)
        return (command.group(1) + rest).strip()
    # Preserve the legacy cleanup for paths and all other non-command text.
    command = re.match(rf"(?i)\s*(/[A-Za-z0-9_]+)@{handle}\b[,:\-]*\s*", text)
    if command:
        rest = re.sub(rf"(?i)@{handle}\b[,:\-]*\s*", "", text[command.end():]).strip()
        return f"{command.group(1)} {rest}" if rest else command.group(1)
    cleaned = re.sub(rf"(?i)@{handle}\b[,:\-]*\s*", "", text).strip()
    return cleaned or text
