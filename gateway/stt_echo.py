"""User-facing formatting for gateway STT transcript echoes."""

from __future__ import annotations

import html
from typing import Any, Callable, Dict, List, Optional


_TELEGRAM_EXPANDABLE_OPEN = "<blockquote expandable>"
_TELEGRAM_EXPANDABLE_CLOSE = "</blockquote>"
# * Matches truncate_message()'s reserve so " (99/99)" never overflows a chunk.
_CHUNK_INDICATOR_RESERVE = 10


def _platform_key(platform: Any) -> str:
    """Normalize a Platform enum / string into a lowercase key."""
    if platform is None:
        return ""
    value = getattr(platform, "value", platform)
    return str(value or "").strip().lower().replace("-", "_")


def _is_telegram_platform(platform: Any) -> bool:
    key = _platform_key(platform)
    return key == "telegram" or key.endswith("_telegram") or key.startswith("telegram_")


def format_stt_transcript_echo(transcript: str, platform: Optional[Any] = None) -> str:
    """Format a successful STT transcript for the optional user-facing echo.

    Telegram receives an HTML expandable blockquote (collapsed quote) so long
    transcripts stay out of the way and markdown characters in the transcript
    cannot break formatting. Other platforms keep the classic ``🎙️ "..."``
    plain line.
    """
    text = (transcript or "").strip("\n")
    if not text.strip():
        return "🎙️"
    if _is_telegram_platform(platform):
        return _format_telegram_expandable_stt_echo(text)
    return f'🎙️ "{text}"'


def stt_echo_metadata(
    platform: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return send metadata for an STT echo, enabling Telegram HTML delivery."""
    if not _is_telegram_platform(platform):
        return metadata
    merged = dict(metadata or {})
    merged["telegram_html"] = True
    return merged


def chunk_telegram_stt_echo_html(
    content: str,
    max_length: int,
    len_fn: Optional[Callable[[str], int]] = None,
) -> Optional[List[str]]:
    """Split a Telegram STT echo so each chunk is a complete expandable quote.

    ``format_stt_transcript_echo`` wraps the whole transcript in one
    ``<blockquote expandable>``. A naive length split then leaves the opening
    tag on chunk 1 and the tail as plain text (or broken HTML that Telegram
    rejects). This re-wraps every slice.

    Args:
        content: Already-formatted STT HTML from ``format_stt_transcript_echo``.
        max_length: Per-message cap in the same units as *len_fn*.
        len_fn: Length function; defaults to ``len``. Pass ``utf16_len`` for
            Telegram's UTF-16 budget.

    Returns:
        A list of sendable HTML chunks, or ``None`` when *content* is not an
        expandable STT quote (caller should use its generic splitter).
    """
    _len = len_fn or len
    parsed = _parse_telegram_expandable_stt_html(content)
    if parsed is None:
        return None

    prefix, inner, suffix = parsed
    if _len(content) <= max_length:
        return [content]

    wrapper = f"{prefix}{_TELEGRAM_EXPANDABLE_OPEN}{_TELEGRAM_EXPANDABLE_CLOSE}"
    body_budget = max_length - _len(wrapper) - _CHUNK_INDICATOR_RESERVE
    if body_budget < 1:
        body_budget = max(1, max_length // 2)

    bodies = _split_html_escaped_body(inner, body_budget, _len)
    total = len(bodies)
    chunks: List[str] = []
    for index, body in enumerate(bodies):
        indicator = f" ({index + 1}/{total})" if total > 1 else ""
        # * Trailing suffix (normally empty) stays on the last chunk only.
        tail = suffix if index == total - 1 else ""
        chunks.append(
            f"{prefix}{_TELEGRAM_EXPANDABLE_OPEN}{body}"
            f"{_TELEGRAM_EXPANDABLE_CLOSE}{indicator}{tail}"
        )
    return chunks


def _format_telegram_expandable_stt_echo(transcript: str) -> str:
    """Wrap *transcript* as a Telegram HTML expandable blockquote.

    Callers must send this with ``metadata["telegram_html"]=True`` so the
    Telegram adapter skips MarkdownV2 conversion (which would otherwise mangle
    ``*`` / ``**`` inside the quote body).
    """
    escaped = html.escape(transcript, quote=False)
    return f"🎙️\n{_TELEGRAM_EXPANDABLE_OPEN}{escaped}{_TELEGRAM_EXPANDABLE_CLOSE}"


def _parse_telegram_expandable_stt_html(
    content: str,
) -> Optional[tuple[str, str, str]]:
    """Return ``(prefix, inner, suffix)`` for an expandable STT HTML echo."""
    start = content.find(_TELEGRAM_EXPANDABLE_OPEN)
    if start < 0:
        return None
    end = content.rfind(_TELEGRAM_EXPANDABLE_CLOSE)
    close_len = len(_TELEGRAM_EXPANDABLE_CLOSE)
    if end < 0 or end < start + len(_TELEGRAM_EXPANDABLE_OPEN):
        return None
    prefix = content[:start]
    inner = content[start + len(_TELEGRAM_EXPANDABLE_OPEN):end]
    suffix = content[end + close_len:]
    return prefix, inner, suffix


def _split_html_escaped_body(
    text: str,
    budget: int,
    len_fn: Callable[[str], int],
) -> List[str]:
    """Split already-escaped HTML body text without breaking ``&amp;`` / ``&lt;`` / ``&gt;``."""
    if budget < 1:
        budget = 1
    if not text:
        return [text]
    if len_fn(text) <= budget:
        return [text]

    chunks: List[str] = []
    remaining = text
    while remaining:
        if len_fn(remaining) <= budget:
            chunks.append(remaining)
            break

        cp_limit = _codepoint_limit(remaining, budget, len_fn)
        region = remaining[:cp_limit]
        split_at = region.rfind("\n")
        if split_at < cp_limit // 2:
            split_at = region.rfind(" ")
        if split_at < 1:
            split_at = max(1, cp_limit)
        split_at = _entity_safe_split(remaining, split_at)
        if split_at < 1:
            split_at = max(1, cp_limit)

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
        if not remaining and not chunks[-1]:
            break
    return chunks or [text]


def _codepoint_limit(text: str, budget: int, len_fn: Callable[[str], int]) -> int:
    """Largest codepoint offset whose ``len_fn`` length is within *budget*."""
    if len_fn(text) <= budget:
        return len(text)
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len_fn(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _entity_safe_split(text: str, split_at: int) -> int:
    """Move *split_at* before an HTML entity if the cut would land inside it."""
    amp = text.rfind("&", 0, split_at)
    if amp < 0:
        return split_at
    semi = text.find(";", amp)
    if 0 <= semi < split_at:
        return split_at
    if amp > 0:
        return amp
    # ! Entity starts at 0 and is wider than the budget — keep it whole.
    if semi >= 0:
        return semi + 1
    return split_at
