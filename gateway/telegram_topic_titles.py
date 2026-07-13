"""Policy helpers for Telegram DM topic title presentation.

The default readable style preserves Hermes' existing auto-title behavior.
Operators can opt into compact, Unicode-aware topic names through Telegram
platform configuration without changing title generation for other surfaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

_READABLE_STYLE = "readable"
_COMPACT_STYLE = "compact"
_MIN_COMPACT_WORDS = 1
_MAX_COMPACT_WORDS = 6
_DEFAULT_COMPACT_WORDS = 2
_MAX_TITLE_CHARS = 120
_SAFE_FALLBACK_TITLE = "Hermes Chat"
_GENERIC_GENERATED_TITLES = {
    "hermes agent",
    "hermes chat",
    "new chat",
    "new session",
    "new topic",
    "untitled",
    "untitled session",
}


@dataclass(frozen=True)
class TelegramTopicTitleOptions:
    """Resolved user-facing options for Telegram DM topic titles."""

    style: str = _READABLE_STYLE
    compact_max_words: int = _DEFAULT_COMPACT_WORDS
    generic_titles: tuple[str, ...] = ()


def telegram_topic_title_options(extra: Optional[Mapping[str, Any]]) -> TelegramTopicTitleOptions:
    """Resolve nested topic-title options from Telegram platform ``extra``.

    Invalid values fail safe to the backward-compatible readable style. The
    compact word count is clamped so configuration cannot create empty or
    unreasonably long topic names.
    """

    raw = (extra or {}).get("dm_topic_titles")
    config = raw if isinstance(raw, Mapping) else {}

    style = str(config.get("style") or _READABLE_STYLE).strip().lower()
    if style not in {_READABLE_STYLE, _COMPACT_STYLE}:
        style = _READABLE_STYLE

    raw_max_words = config.get("compact_max_words", _DEFAULT_COMPACT_WORDS)
    try:
        max_words = (
            _DEFAULT_COMPACT_WORDS
            if isinstance(raw_max_words, bool)
            else int(raw_max_words)
        )
    except (TypeError, ValueError):
        max_words = _DEFAULT_COMPACT_WORDS
    max_words = max(_MIN_COMPACT_WORDS, min(_MAX_COMPACT_WORDS, max_words))

    raw_generic_titles = config.get("generic_titles")
    generic_titles = (
        tuple(
            title
            for item in raw_generic_titles
            if (title := re.sub(r"\s+", " ", str(item or "")).strip())
        )
        if isinstance(raw_generic_titles, (list, tuple))
        else ()
    )

    return TelegramTopicTitleOptions(
        style=style,
        compact_max_words=max_words,
        generic_titles=generic_titles,
    )


def _readable_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
    if not cleaned:
        return _SAFE_FALLBACK_TITLE
    if len(cleaned) > _MAX_TITLE_CHARS:
        return cleaned[: _MAX_TITLE_CHARS - 3].rstrip() + "..."
    return cleaned


def _unicode_words(text: str) -> list[str]:
    """Return letters/digits in source order without language-specific rules."""

    return re.findall(r"[^\W_]+", str(text or ""), flags=re.UNICODE)


def _is_generic_generated_title(
    title: str,
    options: TelegramTopicTitleOptions,
) -> bool:
    normalized = re.sub(r"\s+", " ", str(title or "")).strip().casefold()
    configured = {item.casefold() for item in options.generic_titles}
    return not normalized or normalized in _GENERIC_GENERATED_TITLES or normalized in configured


def sanitize_telegram_topic_title(
    title: str,
    *,
    options: TelegramTopicTitleOptions,
    fallback_text: Optional[str] = None,
) -> str:
    """Return a Bot API-safe title under the configured presentation policy."""

    if options.style != _COMPACT_STYLE:
        return _readable_title(title)

    source = (
        fallback_text
        if _is_generic_generated_title(title, options) and fallback_text
        else title
    )
    words = _unicode_words(source)
    if not words and source != fallback_text and fallback_text:
        words = _unicode_words(fallback_text)
    if not words:
        return _SAFE_FALLBACK_TITLE
    compact = "-".join(word.lower() for word in words[: options.compact_max_words])
    return compact[:_MAX_TITLE_CHARS].rstrip("-")


def dedupe_telegram_topic_title(title: str, existing_titles: Iterable[str]) -> str:
    """Append the first free integer suffix when a title already exists."""

    existing = {str(item or "").strip().casefold() for item in existing_titles}
    if title.casefold() not in existing:
        return title

    suffix = 2
    while True:
        suffix_text = str(suffix)
        base = title[: _MAX_TITLE_CHARS - len(suffix_text)].rstrip("-")
        candidate = f"{base}{suffix_text}"
        if candidate.casefold() not in existing:
            return candidate
        suffix += 1


def resolve_telegram_topic_title_contexts(
    contexts: Sequence[Mapping[str, Any]],
    *,
    options: TelegramTopicTitleOptions,
    title_overrides: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """Resolve every ordered context through one shared dedupe state machine.

    The returned list is aligned with ``contexts``. Runtime rename and channel
    directory rebuild both use this function so they cannot disagree about
    which stable thread receives a collision suffix.
    """
    overrides = title_overrides or {}
    used_by_chat: dict[str, list[str]] = {}
    resolved_titles: list[str] = []

    for context in contexts:
        chat_id = str(context.get("chat_id") or "")
        session_id = str(context.get("session_id") or "")
        title = overrides.get(session_id, str(context.get("title") or ""))
        candidate = sanitize_telegram_topic_title(
            title,
            options=options,
            fallback_text=context.get("first_user_message"),
        )
        used = used_by_chat.setdefault(chat_id, [])
        resolved = (
            dedupe_telegram_topic_title(candidate, used)
            if options.style == _COMPACT_STYLE
            else candidate
        )
        used.append(resolved)
        resolved_titles.append(resolved)

    return resolved_titles
