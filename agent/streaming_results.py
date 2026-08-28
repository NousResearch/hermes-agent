"""Streaming results middleware.

Progressively stream tool results instead of batching them.
Enables early interruption if the agent goes off track.

Pattern: instead of emitting one result at the end,
emit chunks as they're available, allowing the UI/caller
to show progress and react faster.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Iterator, Optional

logger = logging.getLogger(__name__)


class StreamedToolResult:
    """Wrapper for a tool result that can be streamed."""

    def __init__(self, tool_name: str, result: Any, chunk_size: int = 512):
        self.tool_name = tool_name
        self.result = result
        self.chunk_size = chunk_size

    def stream_text(self) -> Iterator[str]:
        """Stream text result in chunks."""
        if not isinstance(self.result, str):
            # Non-text results: emit as JSON chunk
            import json

            yield json.dumps(self.result, default=str)
            return

        # Emit in chunks
        text = self.result
        if not text:
            yield ""
            return
        for i in range(0, len(text), self.chunk_size):
            yield text[i : i + self.chunk_size]

    def stream_multimodal(self) -> Iterator[dict]:
        """Stream multimodal result (dict with _multimodal marker)."""
        if isinstance(self.result, dict) and self.result.get("_multimodal"):
            # Multimodal result: emit as-is (usually images + text)
            yield self.result
            return

        # Not multimodal, wrap it
        yield {"_multimodal": False, "content": self.result}

    async def stream_text_async(self) -> AsyncGenerator[str, None]:
        """Async version of stream_text."""
        if not isinstance(self.result, str):
            import json

            yield json.dumps(self.result, default=str)
            return

        text = self.result
        for i in range(0, len(text), self.chunk_size):
            yield text[i : i + self.chunk_size]


def should_stream_tool(tool_name: str, result: Any) -> bool:
    """
    Decide if this tool result should be streamed progressively.

    Stream when the result is large enough that showing it incrementally is
    meaningfully better than waiting for the whole thing:
    - Text results over 1 KB (any tool)
    - Lists with more than one item from web/extract tools

    Don't stream:
    - Short text (sub-1 KB) — the round-trip delay isn't worth it
    - Unknown tools — we don't know the result shape
    - Non-string / non-list types
    """
    if result is None:
        return False

    known_tools = {
        "read_file", "write_file", "patch", "search_files",
        "terminal", "web_search", "web_extract",
    }
    if tool_name not in known_tools:
        return False

    if isinstance(result, str) and len(result) > 1024:
        return True

    if isinstance(result, list) and len(result) > 1:
        if tool_name in ("web_search", "web_extract"):
            return True

    return False


def stream_tool_result(
    tool_name: str, result: Any, chunk_size: int = 512
) -> StreamedToolResult:
    """Create a streaming wrapper for a tool result."""
    return StreamedToolResult(tool_name, result, chunk_size)
