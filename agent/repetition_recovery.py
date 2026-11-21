"""Safety gates for the single provider fallback after a runaway response."""

from __future__ import annotations

from collections import Counter
from typing import Any


def repetition_tool_recovery_mode(messages: list[dict[str, Any]], user_idx: Any) -> str:
    """Choose whether a repetition retry may use tools, must be answer-only, or must stop.

    A retry after completed tool calls keeps their results in the transcript but receives no
    tool schemas, so it can report the completed work without dispatching it again. An
    unmatched call or uncertain effect is not a safe checkpoint and fails closed.
    """
    if isinstance(user_idx, bool) or not isinstance(user_idx, int) or user_idx < 0 or user_idx >= len(messages):
        return "unsafe"

    expected: list[str] = []
    returned: list[str] = []
    unknown_effect = False
    for message in messages[user_idx + 1:]:
        if not isinstance(message, dict):
            return "unsafe"
        if message.get("role") == "assistant" and message.get("tool_calls"):
            calls = message["tool_calls"]
            if not isinstance(calls, list):
                return "unsafe"
            for call in calls:
                call_id = call.get("id") if isinstance(call, dict) else None
                if not isinstance(call_id, str) or not call_id:
                    return "unsafe"
                expected.append(call_id)
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                return "unsafe"
            returned.append(call_id)
            unknown_effect = unknown_effect or message.get("effect_disposition") == "unknown"

    if not expected and not returned:
        return "tools"
    if unknown_effect or Counter(expected) != Counter(returned):
        return "unsafe"
    return "answer_only"


def tools_for_repetition_retry(agent: Any, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Do not expose action tools when the retry checkpoint already contains tool results."""
    if getattr(agent, "_repetition_fallback_no_tools", False):
        return []
    return tools
