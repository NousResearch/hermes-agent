"""claude-code-core: Stream-json parser and types for Claude Code CLI integration.

Minimal subset copied from CCDB for Hermes integration — only the parser
and type definitions needed by ClaudeCliTransport and ClaudeCliRunner.
"""

from .parser import parse_line
from .types import (
    TOOL_CATEGORIES,
    ContentBlockType,
    ImageData,
    MessageType,
    StreamEvent,
    ToolCategory,
    ToolUseEvent,
)

__all__ = [
    "ContentBlockType",
    "ImageData",
    "MessageType",
    "StreamEvent",
    "TOOL_CATEGORIES",
    "ToolCategory",
    "ToolUseEvent",
    "parse_line",
]
