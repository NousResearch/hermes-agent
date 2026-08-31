"""Streaming results helper.

Provides chunked iteration over tool results so callers can show progress
and react faster than waiting for the full result to be serialised.

Design:
- ``StreamedToolResult.stream_text()`` is the canonical sync iterator.
- ``StreamedToolResult.stream_text_async()`` delegates to ``stream_text()``
  so chunking / serialisation logic lives in exactly one place.
- ``should_stream_tool()`` gates purely on result *size and type*, not on a
  hard-coded list of tool names, so it works for any current and future tool.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Iterator

logger = logging.getLogger(__name__)

# Threshold above which streaming is worthwhile (bytes).
_STREAM_TEXT_MIN_BYTES = 1024
# Threshold above which a list result is worth streaming.
_STREAM_LIST_MIN_ITEMS = 2


def _iter_chunks(text: str, chunk_size: int) -> Iterator[str]:
    """Yield *text* in successive slices of at most *chunk_size* chars."""
    if not text:
        yield ""
        return
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


class StreamedToolResult:
    """Wrapper that yields a tool result as a stream of chunks."""

    def __init__(self, tool_name: str, result: Any, chunk_size: int = 512):
        self.tool_name = tool_name
        self.result = result
        self.chunk_size = chunk_size

    def stream_text(self) -> Iterator[str]:
        """Yield the result as text chunks.

        Non-string results are JSON-serialised into a single chunk so that
        the caller always receives ``str`` chunks regardless of the original type.
        """
        if not isinstance(self.result, str):
            yield json.dumps(self.result, default=str)
            return
        yield from _iter_chunks(self.result, self.chunk_size)

    async def stream_text_async(self) -> AsyncGenerator[str, None]:
        """Async variant of ``stream_text``.

        Delegates directly to ``stream_text()`` so the chunking and
        serialisation logic lives in one place.
        """
        for chunk in self.stream_text():
            yield chunk

    def stream_multimodal(self) -> Iterator[dict]:
        """Yield a multimodal result.

        If the result carries a ``_multimodal`` marker it is emitted as-is
        (images + text).  Otherwise it is wrapped in a plain dict.
        """
        if isinstance(self.result, dict) and self.result.get("_multimodal"):
            yield self.result
            return
        yield {"_multimodal": False, "content": self.result}


def should_stream_tool(tool_name: str, result: Any) -> bool:  # noqa: ARG001
    """Return True when streaming *result* is worthwhile.

    The decision is based entirely on result *size and type* — no hard-coded
    list of tool names.  Any tool that produces a large text blob or a
    multi-item list is worth streaming; short results are not.

    ``tool_name`` is accepted as a parameter for logging / future extension
    but is not used in the decision logic itself.
    """
    if result is None:
        return False

    # Large text responses: any tool
    if isinstance(result, str) and len(result) >= _STREAM_TEXT_MIN_BYTES:
        return True

    # Multi-item lists: worth showing incrementally
    if isinstance(result, list) and len(result) >= _STREAM_LIST_MIN_ITEMS:
        return True

    return False


def stream_tool_result(
    tool_name: str, result: Any, chunk_size: int = 512
) -> StreamedToolResult:
    """Create a ``StreamedToolResult`` wrapper for *result*."""
    return StreamedToolResult(tool_name, result, chunk_size)
