"""Decide whether a reference-only compaction handoff would drive the next model call (#80622)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.context_compressor import ContextCompressor, is_compaction_summary_message


def _handoff_carries_live_user_content(message: Any) -> bool:
    """True when a summary-bearing row still carries a live user ask (pre-filter with ``is_compaction_summary_message``)."""
    return isinstance(message, dict) and ContextCompressor._strip_context_summary_handoff_message(message) is not None


def reference_handoff_would_drive_next_model_call(messages: Optional[List[Dict[str, Any]]]) -> bool:
    """True when the next model call would be driven only by a handoff; trailing tool rows mean an in-flight exchange."""
    if not messages:
        return False

    last_driving_handoff = -1
    for index, message in enumerate(messages):
        if not is_compaction_summary_message(message):
            continue
        merged_completed_assistant = (
            isinstance(message, dict) and message.get("role") == "assistant"
            and ContextCompressor.classify_summary_content(message.get("content"), paraphrased=True) == "merged"
            and message.get("finish_reason") == "stop" and not message.get("tool_calls")
        )
        # Embedded live ask or pending tool_calls -> not a sole-handoff driver.
        if not (_handoff_carries_live_user_content(message) and not merged_completed_assistant):
            last_driving_handoff = index
    if last_driving_handoff < 0:
        return False
    for message in messages[last_driving_handoff + 1 :]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if (
            role == "tool" or (role == "assistant" and message.get("tool_calls"))
            or ContextCompressor._is_real_user_turn(message)
            or (is_compaction_summary_message(message) and _handoff_carries_live_user_content(message))
        ):
            return False
    return True
