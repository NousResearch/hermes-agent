"""Telegram bot-mention routing helpers, kept out of the adapter facade.

``exclusive_bot_mentions`` must only drop a group message that is *addressed* to another bot —
a leading run of handles (``@BotA @BotB: do this``). A foreign handle that merely appears later
in the text talks *about* that bot and must not silence this one (#136236).
"""

from __future__ import annotations

import re
from typing import Any, Iterator, Optional


FOREIGN_BOT_HANDLE_RE = re.compile(r"[a-z0-9_]{2,29}bot", re.IGNORECASE)

# Handles may be chained at the head of a message with these separators only (#136236).
_LEADING_HANDLE_SEPARATOR_CHARS = " \t\n\r,:;"

_RAW_LEADING_HANDLE_RE = re.compile(r"(?:/[A-Za-z0-9_]+)?@([A-Za-z0-9_]{2,31})\b")


def entity_sources(message: Any):
    """``(text, entities)`` pairs for the message text and caption."""
    yield getattr(message, "text", None) or "", getattr(message, "entities", None) or []
    yield getattr(message, "caption", None) or "", getattr(message, "caption_entities", None) or []


def entity_type(entity: Any) -> str:
    return str(getattr(entity, "type", "")).split(".")[-1].lower()


def telegram_entity_text(source_text: str, offset: int, length: int) -> str:
    """Return a Telegram entity span using UTF-16 code-unit offsets."""
    if offset < 0 or length <= 0:
        return ""
    try:
        return source_text.encode("utf-16-le")[offset * 2:(offset + length) * 2].decode("utf-16-le")
    except UnicodeDecodeError:
        return ""


def entity_span(source_text: str, entity: Any) -> Optional[str]:
    """The entity's text, or None when its offsets are unusable."""
    # Telegram's official group-disambiguation form for slash commands (``/cmd@botname``) is emitted as
    # a single ``bot_command`` entity covering the whole span — there is no accompanying ``mention``
    # entity. Treat it as a direct address to this bot when the ``@botname`` suffix matches. This is the
    # form Telegram's own command menu autocomplete produces in groups, so dropping it at the mention
    # gate would break /new, /reset, /help, ... for every group that has ``require_mention`` enabled
    # (#15415).
    offset = int(getattr(entity, "offset", -1))
    length = int(getattr(entity, "length", 0))
    if offset < 0 or length <= 0:
        return None
    return telegram_entity_text(source_text, offset, length)


def _is_bot_handle(handle: str, own: str) -> bool:
    if not handle:
        return False
    if own and handle == own:
        return True
    return bool(FOREIGN_BOT_HANDLE_RE.fullmatch(handle))


def extract_bot_mention_usernames(message: Any, self_username: str = "") -> set[str]:
    """Explicit bot usernames mentioned anywhere in text/captions: foreign handles count only when
    bot-shaped (``...bot``), ``self_username`` opts our OWN handle in regardless of shape. Entity
    mentions are authoritative; the raw-text fallback is deliberately narrow."""
    mentioned_bot_usernames: set[str] = set()
    own = (self_username or "").lstrip("@").lower()

    for source_text, entities in entity_sources(message):
        for entity in entities:
            etype = entity_type(entity)
            if etype not in {"mention", "bot_command"}:
                continue
            entity_text = entity_span(source_text, entity)
            if entity_text is None:
                continue
            entity_text = entity_text.strip()
            if etype == "mention":
                handle = entity_text.lstrip("@").lower()
                if _is_bot_handle(handle, own):
                    mentioned_bot_usernames.add(handle)
                continue
            # /cmd@botname is one bot_command entity; its suffix is an explicit bot address.
            at_index = entity_text.find("@")
            if at_index < 0:
                continue
            command_target = entity_text[at_index + 1:].strip().lower()
            if _is_bot_handle(command_target, own):
                mentioned_bot_usernames.add(command_target)
    # Entity-less fallback only: if Telegram supplied entities, trust them (no URL/code rescue).
    for raw_text, entities in entity_sources(message):
        if not raw_text or entities:
            continue
        for match in re.finditer(r"(?i)(?<![A-Za-z0-9_`/])@([A-Za-z0-9_]{2,31})\b", raw_text):
            handle = match.group(1).lower()
            if _is_bot_handle(handle, own):
                mentioned_bot_usernames.add(handle)
    return mentioned_bot_usernames


def _utf16_to_str_offsets(source_text: str) -> list[int]:
    """``offsets[u16_offset] -> str index`` (length == UTF-16 length + 1); a surrogate pair's low
    half maps to the pair's own index so an entity never ends mid-pair unseen."""
    offsets: list[int] = []
    for index, ch in enumerate(source_text):
        offsets.append(index)
        if ord(ch) > 0xFFFF:
            offsets.append(index)
    offsets.append(len(source_text))
    return offsets


def _handle_from_entity(etype: str, span: str) -> Optional[str]:
    """The handle an entity span addresses: ``mention`` -> the handle; ``bot_command`` -> its
    ``@botname`` suffix (a bare ``/cmd`` addresses no one -> None)."""
    span = span.strip()
    if etype == "mention":
        return span.lstrip("@").lower()
    at_index = span.find("@")
    if at_index < 0:
        return None
    return span[at_index + 1:].strip().lower()


def _leading_handles(source_text: str, entities: list) -> Iterator[str]:
    """Handles addressed at the head of one text/caption source, in order.

    The leading run is handles separated by whitespace/`,`/`;`/`:` and ends at the first token
    that is not a handle. With entities, a handle counts only where a ``mention``/``bot_command``
    entity starts (a handle inside a code span addresses no one); entity offsets are UTF-16.
    Without entities, ``@handle`` — and a leading ``/cmd@handle`` — is matched in the raw text.
    """
    spans_by_start: dict[int, tuple[int, str, str]] = {}
    if entities:
        offsets = _utf16_to_str_offsets(source_text)
        for entity in entities:
            etype = entity_type(entity)
            if etype not in {"mention", "bot_command"}:
                continue
            u16_offset = int(getattr(entity, "offset", -1))
            u16_length = int(getattr(entity, "length", 0))
            if u16_offset < 0 or u16_length <= 0 or u16_offset + u16_length >= len(offsets):
                continue
            start, end = offsets[u16_offset], offsets[u16_offset + u16_length]
            if end <= start:
                continue
            spans_by_start.setdefault(start, (end, etype, source_text[start:end]))
    pos = 0
    text_len = len(source_text)
    while pos < text_len:
        entry = spans_by_start.get(pos) if entities else None
        if entry is not None:
            end, etype, span = entry
            handle = _handle_from_entity(etype, span)
            if handle is None:
                break
            pos = end
        elif not entities:
            match = _RAW_LEADING_HANDLE_RE.match(source_text, pos)
            if match is None:
                break
            handle = match.group(1).lower()
            pos = match.end()
        else:
            break
        yield handle
        while pos < text_len and source_text[pos] in _LEADING_HANDLE_SEPARATOR_CHARS:
            pos += 1


def leading_bot_mention_usernames(message: Any, self_username: str = "") -> set[str]:
    """Bot usernames the head of the text/caption is addressed to (#136236).

    A foreign handle later in the message only talks *about* another bot and no longer silences
    this one. ``self_username`` counts as addressed-to-us anywhere in the run, regardless of
    shape (collectible usernames need not end in ``bot``)."""
    own = (self_username or "").lstrip("@").lower()
    handles: set[str] = set()
    for source_text, entities in entity_sources(message):
        if not source_text:
            continue
        for handle in _leading_handles(source_text, entities):
            if _is_bot_handle(handle, own):
                handles.add(handle)
    return handles
