"""Vendored copy of the mcp-unicode-sanitization library (v1.0.0).

This is the dependency-free (stdlib-only) sanitization core built by upstream
task t_8f7c33c4. It is vendored here so the Hermes MCP Gateway plugin carries
no third-party runtime dependency. The public API surface:

    sanitize_tool_metadata(tool: dict) -> SanitizedTool
    SanitizedTool.safe -> bool

``tool`` is shaped like an MCP ``tools/list`` entry:
    {"name": str, "description": str, "inputSchema": { ... }}
"""

from .core import (
    BIDI_CONTROLS,
    INVISIBLE,
    TAG_BLOCK,
    SanitizeResult,
    SanitizedTool,
    definition_mutated,
    is_dangerous_default,
    namespaces_collide,
    sanitize,
    sanitize_tool_metadata,
    tool_hash,
)

__all__ = [
    "BIDI_CONTROLS",
    "INVISIBLE",
    "TAG_BLOCK",
    "SanitizeResult",
    "SanitizedTool",
    "definition_mutated",
    "is_dangerous_default",
    "namespaces_collide",
    "sanitize",
    "sanitize_tool_metadata",
    "tool_hash",
]

__version__ = "1.0.0"
