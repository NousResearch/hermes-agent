"""Hermes Approval Guard — two-stage tool call approval plugin.

Stage 1: Rule engine (<1ms) + LLM fast classification (~500ms)
Stage 2: ACP Agent deep review (3-8s)

Covers all tools, provides structured denial feedback, Hindsight memory learning.
Disabled by default; requires plugin_guard.enabled: true."""

from __future__ import annotations

from .guard import pre_tool_call_handler


def register(ctx) -> None:
    """Register pre_tool_call hook."""
    ctx.register_hook("pre_tool_call", pre_tool_call_handler)
