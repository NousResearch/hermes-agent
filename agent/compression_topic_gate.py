"""Cheap, opt-in relevance gate for automatic context compression."""

from __future__ import annotations

import re
from typing import Any


_WORDS = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_COMMON_WORDS = frozenset({
    "about", "after", "again", "also", "been", "being", "could", "does", "from", "have",
    "into", "just", "like", "make", "more", "much", "only", "other", "over", "should",
    "that", "their", "them", "then", "there", "these", "they", "this", "those", "through",
    "very", "want", "what", "when", "where", "which", "with", "would", "your",
})


def _words(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {word.casefold() for word in _WORDS.findall(value) if word.casefold() not in _COMMON_WORDS}


def _current_user_index(messages: list[dict[str, Any]]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], dict) and messages[index].get("role") == "user":
            return index
    return None


def should_defer_automatic_compression(agent: Any, messages: list[dict[str, Any]], *, current_tokens: int) -> bool:
    """Return whether an unrelated current turn may safely defer auto-compression.

    This deliberately fails open: malformed or text-free messages, an unknown context window,
    and a request at the context boundary all retain the established compression path. Manual
    compression never calls this helper.
    """
    if not getattr(agent, "compression_topic_gate_enabled", False):
        return False
    context_length = getattr(getattr(agent, "context_compressor", None), "context_length", 0)
    if (
        not isinstance(context_length, int)
        or not isinstance(current_tokens, int)
        or context_length <= 0
        or current_tokens >= context_length
    ):
        return False
    current_index = _current_user_index(messages)
    if current_index is None:
        return False
    current_words = _words(messages[current_index].get("content"))
    if not current_words:
        return False
    history_words: set[str] = set()
    for message in messages[:current_index]:
        if isinstance(message, dict) and message.get("role") in {"user", "assistant"}:
            history_words.update(_words(message.get("content")))
    return bool(history_words) and not bool(current_words & history_words)
