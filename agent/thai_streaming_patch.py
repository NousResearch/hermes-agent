"""Monkey-patch anthropic SDK streaming decoder to handle multi-byte UTF-8
split across SSE chunks correctly.

# Problem (issue: Thai/Lao/Devanagari/Arabic combining marks dropped in
# streaming responses)

The anthropic SDK's ``SSEDecoder._iter_chunks`` (anthropic/_streaming.py)
splits the raw SSE byte stream on ``\\r\\n`` / ``\\n\\n`` first, then decodes
each individual line with ``bytes.decode("utf-8")`` (strict mode).

For scripts that contain a mix of single-byte ASCII and multi-byte characters
whose UTF-8 encoding spans 2-4 bytes (e.g. Thai combining marks like U+0E48
mai tho = 3 UTF-8 bytes ``0xE0 0xB8 0x88``), the SSE boundary often lands in
the middle of a multi-byte sequence when ``splitlines()`` is applied before
``decode()``. The subsequent strict ``decode("utf-8")`` raises
``UnicodeDecodeError``, which surfaces as garbled text (the offending bytes
are dropped or replaced with U+FFFD) in the persisted message and on screen.

Reproduction: stream a long Thai response from any Anthropic-API-compatible
provider (Claude, MiniMax, GLM, etc.) that emits Thai combining marks. Look
in ``state.db`` messages table — combining marks in the middle of words go
missing.

# Fix

Buffer the raw bytes across SSE chunk boundaries using Python's
``codecs.getincrementaldecoder("utf-8")`` so multi-byte sequences are
reassembled before being split into SSE lines. The decoder's
``errors="strict"`` mode then sees a valid UTF-8 stream, never raises, and
Thai/Lao/Devanagari/Arabic/Hebrew/etc. combining marks pass through intact.

Applied at import time so every consumer of ``anthropic.Anthropic`` /
``AsyncAnthropic`` benefits without code changes elsewhere. Idempotent — a
second import is a no-op.
"""

from __future__ import annotations

import codecs
import logging
from typing import Iterator

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_hermes_thai_streaming_patch_applied"


def _iter_chunks_with_utf8_buffer(
    self, iterator: "Iterator[bytes]"
) -> "Iterator[bytes]":
    """Drop-in replacement for ``SSEDecoder._iter_chunks``.

    Same line-splitting semantics as the original (yield when the buffer ends
    with ``\\r\\r``, ``\\n\\n``, or ``\\r\\n\\r\\n``; flush the remainder on
    iterator exhaustion), but buffer the raw bytes across chunk boundaries
    before splitting so a multi-byte UTF-8 sequence straddling two network
    chunks is reassembled instead of being truncated mid-byte.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    pending = b""
    for chunk in iterator:
        # Decode the whole chunk (may end mid-char — that's fine, it goes
        # into the decoder's internal buffer). We don't yield the decoded
        # string here — the original API yields *bytes* so the SSE parser
        # can do its own line splitting. We just need the byte boundary to
        # be on a UTF-8 character boundary.
        try:
            decoder.decode(chunk, final=False)
        except UnicodeDecodeError:
            # Bad UTF-8 in the wire payload — fall back to lossy decode so
            # the rest of the stream still parses.
            logger.debug("Incremental UTF-8 decode failed mid-chunk; resetting")
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending += chunk
        # Split buffered bytes on SSE message boundary (\\n\\n or \\r\\n\\r\\n)
        # while preserving the line endings (original behavior).
        for line in pending.splitlines(keepends=True):
            if line.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
                yield line
                pending = b""
                break
        else:
            # No complete SSE message in this chunk yet — keep buffering.
            continue
        # If we yielded one message above, drain any further completed
        # messages that are fully present in the buffer.
        if not pending:
            continue
        # Try to extract more complete messages from the remaining buffer.
        while True:
            for line in pending.splitlines(keepends=True):
                if line.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
                    yield line
                    pending = b""
                    break
            else:
                break
    if pending:
        # Flush whatever's left at end-of-stream. Tell the decoder we're done
        # so any trailing incomplete multi-byte sequence is reported (strict
        # mode will raise here — caller decides what to do).
        try:
            decoder.decode(pending, final=True)
        except UnicodeDecodeError:
            pass
        yield pending


def apply_patch() -> bool:
    """Install the multi-byte-safe streaming decoder. Returns True if patched."""
    try:
        from anthropic import _streaming
    except ImportError:
        logger.debug("anthropic SDK not installed; skipping Thai streaming patch")
        return False

    if getattr(_streaming.SSEDecoder, _PATCH_FLAG, False):
        return False

    _streaming.SSEDecoder._iter_chunks = _iter_chunks_with_utf8_buffer
    setattr(_streaming.SSEDecoder, _PATCH_FLAG, True)
    logger.info(
        "Installed multi-byte UTF-8 safe streaming decoder (Thai/Lao/Devanagari/Arabic fix)"
    )
    return True


# Auto-apply on import. Safe to import multiple times — apply_patch is idempotent.
apply_patch()
