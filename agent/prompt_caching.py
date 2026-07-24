"""Anthropic prompt caching strategy.

Single layout: ``system_and_3``. 4 cache_control breakpoints — system
prompt + last 3 non-system messages, all at the same TTL (5m or 1h).
Reduces input token costs by ~75% on multi-turn conversations within a
single session.

Pure functions -- no class state, no AIAgent dependency.
"""

import copy
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
            # marker -- skip. Non-empty tool content falls through below and
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
    receive a top-level marker the provider ignores -- wasting one of the four
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


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
) -> List[Dict[str, Any]]:
    """Apply system_and_3 caching strategy to messages for Anthropic models.

    Places up to 4 cache_control breakpoints: system prompt + last 3 non-system
    messages, all at the same TTL.

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = _build_marker(cache_ttl)
    breakpoints_used = 0

    # Handle possible split system message (static + volatile parts) for
    # better cross-session prompt caching.
    if messages[0].get("role") == "system":
        msg = messages[0]
        content = msg.get("content")
        if isinstance(content, str) and "<<SPLIT_HERE>>" in content:
            before, after = content.split("<<SPLIT_HERE>>", 1)
            # Create two system messages: static part and volatile part.
            static_msg = {"role": "system", "content": before.strip()}
            volatile_msg = {"role": "system", "content": after.strip()}
            # Replace the single system message with the two split messages.
            messages[0:1] = [static_msg, volatile_msg]
            # Apply a cache_control marker to the static part if it can carry one.
            if _can_carry_marker(static_msg, native_anthropic=native_anthropic):
                _apply_cache_marker(static_msg, marker, native_anthropic=native_anthropic)
                breakpoints_used += 1
            # The volatile part gets no marker (we save breakpoints for non-system messages).
        else:
            # Normal system message (no split marker).
            if _can_carry_marker(messages[0], native_anthropic=native_anthropic):
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