"""Backwards-compatible alias for the generic transcript adapter.

The implementation moved to ``tameru.transcript`` — it was always
harness-agnostic (OpenAI-style ``role``/``content`` dicts); only the name
said "hermes". This shim keeps existing imports working, including
private helpers used by tests.
"""
from . import transcript as _transcript
from .transcript import (
    MIN_TOOL_CHARS,
    PROTECT_LAST_TOOL,
    apply_extractive_tool_prune,
    bulky_tools_dropped,
    last_user_text,
    query_facts_lost,
)


def __getattr__(name: str):
    return getattr(_transcript, name)


__all__ = [
    "MIN_TOOL_CHARS",
    "PROTECT_LAST_TOOL",
    "apply_extractive_tool_prune",
    "bulky_tools_dropped",
    "last_user_text",
    "query_facts_lost",
]
