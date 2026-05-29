"""Hermes Approval Guard — two-stage tool-call approval plugin.

Stage 1: rule engine (<1ms) + LLM fast classification (~500ms)
Stage 2: ACP Agent deep review (3-8s)

Covers all tools, provides structured denial feedback, Hindsight memory learning.
Disabled by default; requires plugin_guard.enabled: true to activate.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(plugin_context):
    """Register the pre_tool_call hook."""
    from .guard import pre_tool_call_handler

    plugin_context.register_hook("pre_tool_call", pre_tool_call_handler)
    logger.info("Hermes Approval Guard registered (pre_tool_call hook)")
