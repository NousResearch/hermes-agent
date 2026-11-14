"""Steer-buffer retirement across the compaction boundary (#154763).

When compaction rebuilds the context, the rebuilt transcript can carry a steer
that was previously delivered inside a tool result or as a typed steer row. If
the same steer is still buffered in the agent's steer parts, the next drain
would inject a second live copy — the duplication behind #154763. This sibling
owns the retire step; the facade late-imports it (sibling convention: never a
module-level cycle with the facade)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def retire_buffered_steer(agent: Any, messages: list) -> None:
    """The rebuilt context already carries a steer (tool-borne marker or typed steer row); retire any
    buffered copy of that exact text so the next drain cannot inject a second carrier. Text-equality
    scoped: a buffered steer whose text differs is still live intent and stays buffered."""
    if agent is None:
        return
    from agent.conversation_compression import _extract_steer_text_from_message
    from agent.interrupt_control import _steer_parts, consume_buffered_steer
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool" or message.get("display_kind") == "steer":
            steer_text = _extract_steer_text_from_message(message)
            if steer_text and consume_buffered_steer(agent, steer_text):
                return
    if _steer_parts(agent):
        logger.info("compaction carries a steer but the buffered steer(s) did not match its text (session=%s)",
                    getattr(agent, "session_id", "") or "none")
