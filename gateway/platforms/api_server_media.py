"""Inline ``MEDIA:<path>`` image tags as base64 data URLs for API-server clients.

Remote frontends can't read server paths, so a reply's image tags are replaced by data URLs before
they leave the API server. Non-image and unreadable paths stay as written.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE, _terminal_sentinel_start, validate_media_delivery_path

_MEDIA_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
               ".webp": "image/webp", ".bmp": "image/bmp"}
_MEDIA_IMG_EXT = set(_MEDIA_MIME)
_MEDIA_DATA_URL_MAX_BYTES = 5 * 1024 * 1024  # skip images larger than 5MB


def _resolve_media_to_data_urls(text: str) -> str:
    """Replace ``MEDIA:<path>`` tags with inline base64 data URLs (remote frontends can't read
    server paths); non-image/unreadable paths stay untouched. Security: the shared
    ``MEDIA_TAG_CLEANUP_RE`` anchor + ``validate_media_delivery_path`` denylist — a bare-token
    match would let a traversal path in the reply exfiltrate any readable image."""
    if not text or "MEDIA:" not in text:
        return text
    import base64

    def _to_data_url(path_str: str) -> Optional[str]:
        # validate_media_delivery_path() strips wrapping quotes/trailing punctuation itself.
        safe_path = validate_media_delivery_path(path_str)
        p = Path(safe_path) if safe_path else None
        suffix = p.suffix.lower() if p else ""
        if suffix not in _MEDIA_IMG_EXT:
            return None
        try:
            if p.stat().st_size > _MEDIA_DATA_URL_MAX_BYTES:
                return None
            b64 = base64.b64encode(p.read_bytes()).decode()
        except OSError:
            return None
        return f"![image](data:{_MEDIA_MIME[suffix]};base64,{b64})"

    def _repl(m: re.Match[str]) -> str:
        return _to_data_url(m.group("path")) or m.group(0)
    try:
        # A leaked terminal <|eos|> glued to the last tag is not a path terminator (#111046):
        # scan without it, and drop it (control token, never content) only when a tag resolved.
        sentinel_start = _terminal_sentinel_start(text)
        scan = text[:sentinel_start] if sentinel_start >= 0 else text
        resolved = MEDIA_TAG_CLEANUP_RE.sub(_repl, scan)
        return text if resolved == scan else resolved
    except Exception:
        return text


_MEDIA_HOLDBACK_WORD = "MEDIA"
# How much text may follow a "MEDIA:" before it is released as plain text: generous for any real
# path, small enough that a stray "MEDIA:" in prose doesn't visibly stall a live stream.
_MEDIA_HOLDBACK_CAP = 400


class StreamingMediaTagResolver:
    """Buffers streamed text so a ``MEDIA:<path>`` tag is never split across two delta chunks, then
    resolves completed tags to inline data URLs.

    Token-by-token streaming lands a tag's characters in several deltas, and
    ``_resolve_media_to_data_urls`` only recognizes a complete tag, so resolving each delta on its own
    let split tags reach clients as literal ``MEDIA:/path`` text.

    Call :meth:`feed` with each delta and emit what it returns (possibly empty while text is held
    back); call :meth:`flush` once at stream end and emit its return value.
    """

    __slots__ = ("_pending",)

    def __init__(self) -> None:
        self._pending: str = ""

    def feed(self, text: str) -> str:
        """Feed one delta chunk; return the text now safe to emit (resolved)."""
        if not text:
            return ""
        self._pending += text
        safe, self._pending = self._split_safe_boundary(self._pending)
        return _resolve_media_to_data_urls(safe) if safe else ""

    def flush(self) -> str:
        """Release and resolve whatever is still held back (stream end)."""
        if not self._pending:
            return ""
        remainder = self._pending
        self._pending = ""
        return _resolve_media_to_data_urls(remainder)

    @staticmethod
    def _split_safe_boundary(buffer: str) -> tuple[str, str]:
        """Split *buffer* into ``(safe_to_emit, held_back)``.

        Anchored on ``MEDIA:`` with the colon: the bare word also appears inside paths
        (``social_media_post.png``), and matching that would release part of a real tag as text.
        Without a confirmed ``MEDIA:``, only a trailing partial prefix of it is held, since the next
        chunk may complete it. A held tag longer than ``_MEDIA_HOLDBACK_CAP`` is released, and
        ``_resolve_media_to_data_urls`` then leaves an invalid path as visible text.
        """
        upper = buffer.upper()
        idx = upper.rfind(_MEDIA_HOLDBACK_WORD + ":")
        if idx != -1:
            tail = buffer[idx:]
            if len(tail) > _MEDIA_HOLDBACK_CAP:
                return buffer, ""
            return buffer[:idx], tail
        word = _MEDIA_HOLDBACK_WORD
        for n in range(min(len(word), len(buffer)), 0, -1):
            if upper.endswith(word[:n]):
                return buffer[:-n], buffer[-n:]
        return buffer, ""
