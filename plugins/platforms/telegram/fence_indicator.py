"""Chunk-fence indicator handling split out of ``adapter.py`` (file-size offset, moved as-is)."""

import re

_CHUNK_INDICATOR_ON_FENCE_RE = re.compile(
    r"(?m)^``` (?P<indicator>(?:\\)?\(\d+/\d+(?:\\)?\))$"
)


def _separate_chunk_indicator_from_fence(text: str) -> str:
    """Move a ``(N/M)`` chunk marker that ``truncate_message()`` appended to a synthesized closing
    fence onto its own line — Telegram rejects ````` \\(1/2\\)`` as a fence."""
    return _CHUNK_INDICATOR_ON_FENCE_RE.sub(r"```\n\g<indicator>", text)
