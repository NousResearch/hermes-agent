"""Anthropic prompt caching strategy.

Single layout: ``system_and_3``. 4 cache_control breakpoints — system
prompt + last 3 non-system messages, all at the same TTL (5m or 1h).
Reduces input token costs by ~75% on multi-turn conversations within a
single session.

Pure functions -- no class state, no AIAgent dependency.
"""

from typing import Any, Dict, List


def _apply_cache_marker(msg: dict, cache_marker: dict, native_anthropic: bool = False) -> None:
    """Add cache_control to a single message, handling all format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool" and native_anthropic:
        # Native Anthropic layout: top-level marker; the adapter moves it
        # inside the tool_result block.
        msg["cache_control"] = cache_marker
        return

    if content is None or content == "":
        if role == "tool" and not native_anthropic:
            # OpenRouter rejects top-level cache_control on role:tool (silent
            # hang) and an empty message has no content part to carry the
            # marker — skip. Non-empty tool content falls through below and
            # gets the marker on a content part, which OpenRouter honors.
            return
        if role == "assistant" and not native_anthropic:
            # Empty assistant turns are pure tool_calls. A top-level marker
            # here is ignored on the envelope layout, so skip.
            return
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


def _can_carry_marker(msg: dict, native_anthropic: bool) -> bool:
    """True if a marker on this message is actually honored by the provider.

    On the native Anthropic layout every message works (top-level markers are
    relocated by the adapter). On the envelope layout (OpenRouter et al.) only
    markers inside content parts are honored: empty-content messages (e.g.
    assistant turns that are pure tool_calls) and empty tool messages would
    receive a top-level marker the provider ignores — wasting one of the four
    breakpoints. Skip those so the breakpoints land on messages that count.
    """
    if native_anthropic:
        return True
    content = msg.get("content")
    if content is None or content == "":
        return False
    if isinstance(content, list):
        # _apply_cache_marker only marks the LAST content part, so the carrier
        # predicate must agree: a list whose last element isn't a dict cannot
        # actually receive a marker and would waste a breakpoint. Mirror the
        # `content` truthiness + last-element-dict check in _apply_cache_marker.
        return bool(content) and isinstance(content[-1], dict)
    return isinstance(content, str)


def _build_marker(ttl: str) -> Dict[str, str]:
    """Build a cache_control marker dict for the given TTL ('5m' or '1h')."""
    marker: Dict[str, str] = {"type": "ephemeral"}
    if ttl == "1h":
        marker["ttl"] = "1h"
    return marker


def _copy_message_for_api_mutation(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Copy message containers that later API-retry recovery may mutate."""
    copied = dict(msg)

    content = msg.get("content")
    if isinstance(content, list) and content:
        copied_content = []
        for part in content:
            if isinstance(part, dict):
                copied_part = dict(part)
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    copied_part["image_url"] = dict(image_url)
                copied_content.append(copied_part)
            else:
                copied_content.append(part)
        copied["content"] = copied_content

    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        copied_tool_calls = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                copied_tool_call = dict(tool_call)
                function = tool_call.get("function")
                if isinstance(function, dict):
                    copied_tool_call["function"] = dict(function)
                copied_tool_calls.append(copied_tool_call)
            else:
                copied_tool_calls.append(tool_call)
        copied["tool_calls"] = copied_tool_calls

    return copied


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
) -> List[Dict[str, Any]]:
    """Apply system_and_3 caching strategy to messages for Anthropic models.

    Places up to 4 cache_control breakpoints: system prompt + last 3 non-system
    messages, all at the same TTL.

    Returns:
        Copy of messages with cache_control breakpoints injected. Message
        containers are copied so later API-retry recovery can mutate the
        returned request copy without touching canonical history.
    """
    if not api_messages:
        return []

    messages = [
        _copy_message_for_api_mutation(msg) if isinstance(msg, dict) else msg
        for msg in api_messages
    ]

    marker = _build_marker(cache_ttl)

    breakpoints_used = 0

    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = 4 - breakpoints_used
    non_sys = [
        i
        for i in range(len(messages))
        if messages[i].get("role") != "system"
        and _can_carry_marker(messages[i], native_anthropic=native_anthropic)
    ]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages
