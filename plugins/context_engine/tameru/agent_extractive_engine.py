"""Alias for tameru.transcript for backwards compatibility."""
from .transcript import (
    apply_extractive_tool_prune,
    last_user_text,
    MIN_TOOL_CHARS,
    PROTECT_LAST_TOOL,
)

__all__ = [
    "apply_extractive_tool_prune",
    "last_user_text",
    "MIN_TOOL_CHARS",
    "PROTECT_LAST_TOOL",
]
