"""Alias for hermes_extractive_engine for backwards compatibility."""
from .hermes_extractive_engine import (
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
