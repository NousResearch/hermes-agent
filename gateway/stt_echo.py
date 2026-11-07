"""Presentation and lossless HTML chunking for gateway STT transcript echoes."""

from __future__ import annotations

import html
import re
from typing import Any, Callable

from agent.i18n import t
from gateway.config import Platform

TELEGRAM_STT_ECHO_METADATA_KEY = "telegram_stt_echo"
_TELEGRAM_QUOTE_OPEN = "<blockquote expandable>"
_TELEGRAM_QUOTE_CLOSE = "</blockquote>"
_TELEGRAM_ECHO_PREFIX = f"🎙️\n{_TELEGRAM_QUOTE_OPEN}"
_ESCAPED_BODY_TOKEN = re.compile(r"&(?:amp|lt|gt);|[\s\S]")


def format_stt_transcript_echo(transcript: str, platform: Platform) -> str:
    """Wrap Telegram transcripts in an escaped quote; localize other platforms."""
    if platform == Platform.TELEGRAM:
        return f"{_TELEGRAM_ECHO_PREFIX}{html.escape(transcript, quote=False)}{_TELEGRAM_QUOTE_CLOSE}"
    return t("gateway.voice.transcript_echo_short", text=transcript)


def stt_echo_metadata(
    platform: Platform, metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Copy Telegram metadata to select STT HTML delivery without changing its route."""
    if platform == Platform.TELEGRAM:
        return {**(metadata or {}), TELEGRAM_STT_ECHO_METADATA_KEY: True}
    return metadata


def chunk_telegram_stt_echo_html(
    content: str, max_length: int, len_fn: Callable[[str], int] = len,
) -> list[str]:
    """Split a formatted STT quote within a complete wire-payload budget.

    Args:
        content: HTML produced by ``format_stt_transcript_echo`` for Telegram.
        max_length: Maximum length including the quote, prefix and chunk indicator.
        len_fn: Length metric; Telegram delivery supplies ``utf16_len``.

    Returns:
        Independently expandable quotes whose decoded bodies concatenate exactly
        to the original transcript, including all whitespace.

    Raises:
        ValueError: The content is not a formatted STT echo or the budget cannot
            fit its wrapper and one escaped character.
    """
    if not content.startswith(_TELEGRAM_ECHO_PREFIX) or not content.endswith(_TELEGRAM_QUOTE_CLOSE):
        raise ValueError("Expected a formatted Telegram STT echo")
    if len_fn(content) <= max_length:
        return [content]
    body = content[len(_TELEGRAM_ECHO_PREFIX):-len(_TELEGRAM_QUOTE_CLOSE)]
    wrapper_length = len_fn(f"{_TELEGRAM_ECHO_PREFIX}{_TELEGRAM_QUOTE_CLOSE}")
    total = 2
    # * Recompute the budget when pagination gains digits; escaped tokens stay whole.
    while True:
        budget = max_length - wrapper_length - len_fn(f" ({total}/{total})")
        bodies = _split_escaped_body(body, budget, len_fn)
        if len(bodies) <= total:
            total = len(bodies)
            return [
                f"{_TELEGRAM_ECHO_PREFIX}{part}{_TELEGRAM_QUOTE_CLOSE} ({index}/{total})"
                for index, part in enumerate(bodies, 1)
            ]
        total = len(bodies)


def _split_escaped_body(text: str, budget: int, len_fn: Callable[[str], int]) -> list[str]:
    """Pack escaped characters without cutting entities or Unicode code points."""
    if budget < 1:
        raise ValueError("Telegram STT echo budget cannot fit a quote body")
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for match in _ESCAPED_BODY_TOKEN.finditer(text):
        token = match.group()
        token_length = len_fn(token)
        if token_length > budget:
            raise ValueError("Telegram STT echo budget cannot fit an escaped character")
        if current_length + token_length > budget:
            chunks.append("".join(current))
            current = []
            current_length = 0
        current.append(token)
        current_length += token_length
    if current or not chunks:
        chunks.append("".join(current))
    return chunks


def telegram_stt_echo_plain(content: str) -> str:
    """Decode the generated quote markup after a confirmed HTML parse rejection."""
    return html.unescape(content.replace(_TELEGRAM_QUOTE_OPEN, "", 1).replace(_TELEGRAM_QUOTE_CLOSE, "", 1))
