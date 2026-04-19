"""Anthropic prompt caching (system_and_3 strategy).

Reduces input token costs by ~75% on multi-turn conversations by caching
the conversation prefix. Uses 4 cache_control breakpoints (Anthropic max):
  1. System prompt (stable across all turns)
  2-4. Last 3 non-system messages (rolling window)

Pure functions -- no class state, no AIAgent dependency.
"""

from typing import Any, Dict, List


def _clone_message_for_cache(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Clone message fields that later retry/sanitization paths may mutate."""
    cloned = dict(msg)

    content = msg.get("content")
    if isinstance(content, list):
        cloned["content"] = [
            dict(part) if isinstance(part, dict) else part
            for part in content
        ]

    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        cloned_tool_calls = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                tc_clone = dict(tc)
                fn = tc.get("function")
                if isinstance(fn, dict):
                    tc_clone["function"] = dict(fn)
                cloned_tool_calls.append(tc_clone)
            else:
                cloned_tool_calls.append(tc)
        cloned["tool_calls"] = cloned_tool_calls

    return cloned


def _apply_cache_marker(msg: dict, cache_marker: dict, native_anthropic: bool = False) -> None:
    """Add cache_control to a single message, handling all format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool":
        if native_anthropic:
            msg["cache_control"] = cache_marker
        return

    if content is None or content == "":
        msg["cache_control"] = cache_marker
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": cache_marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
) -> List[Dict[str, Any]]:
    """Apply system_and_3 caching strategy to messages for Anthropic models.

    Places up to 4 cache_control breakpoints: system prompt + last 3 non-system messages.

    Returns:
        Copy of messages with cache_control breakpoints injected.
        Unmodified messages are structurally shared to avoid a full deep copy.
    """
    if not api_messages:
        return []

    marker = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"

    breakpoint_indices: List[int] = []
    if api_messages[0].get("role") == "system":
        breakpoint_indices.append(0)

    remaining = 4 - len(breakpoint_indices)
    non_sys = [i for i, msg in enumerate(api_messages) if msg.get("role") != "system"]
    breakpoint_indices.extend(non_sys[-remaining:])

    messages = [_clone_message_for_cache(msg) for msg in api_messages]

    for idx in breakpoint_indices:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages
