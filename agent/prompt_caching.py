"""Anthropic prompt caching.

Reduces input token costs by ~75% on multi-turn conversations by caching
the conversation prefix. Anthropic allows up to 4 cache_control
breakpoints. Strategy:
  1. System prompt (stable across all turns)
  2. Last entry of ``tools[]`` (anchors the system+tools prefix so a
     ToolSearch-driven tools[] mutation only forces ONE rebuild — without
     this, every following turn re-bills the entire message history at
     ``input_tokens`` rates instead of ``cache_read_input_tokens``)
  3-4. Last 2 non-system messages (rolling window)

Pure functions -- no class state, no AIAgent dependency.
"""

import copy
from typing import Any, Dict, List


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


def _build_cache_marker(cache_ttl: str = "5m") -> Dict[str, str]:
    marker: Dict[str, str] = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"
    return marker


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
    reserve_tools_breakpoint: bool = True,
) -> List[Dict[str, Any]]:
    """Apply caching strategy to messages for Anthropic models.

    Places cache_control breakpoints on the system prompt + the last
    non-system messages. When ``reserve_tools_breakpoint`` is True, only
    2 message-side breakpoints are used so the caller can apply the 4th
    on the last entry of ``tools[]`` (see
    ``apply_anthropic_tools_cache_control``). Otherwise 3 message-side
    breakpoints are used (legacy behaviour).

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = _build_cache_marker(cache_ttl)

    breakpoints_used = 0
    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    # Reserve one breakpoint for tools[] so a tools mutation only forces
    # one rebuild, not every subsequent message re-bill.
    budget = 4 - breakpoints_used - (1 if reserve_tools_breakpoint else 0)
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-budget:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages


def apply_anthropic_tools_cache_control(
    anthropic_tools: List[Dict[str, Any]],
    cache_ttl: str = "5m",
) -> List[Dict[str, Any]]:
    """Mark the last entry in ``tools[]`` with ``cache_control`` so the
    ``system + tools`` prefix is cached as a unit.

    Why this matters: Anthropic caches by request prefix in the order
    ``system → tools → messages``. Without a breakpoint AT or AFTER
    ``tools[]``, any change to ``tools[]`` (a ToolSearch ``select:`` load,
    an MCP reconnect, a subagent toolset switch) invalidates the cache for
    every subsequent turn — the message history is forced through
    ``input_tokens`` instead of ``cache_read_input_tokens`` until the
    session ends. With this breakpoint, a tools mutation costs ONE
    rebuild and the cache re-establishes on the next turn.

    Mutates a copy; safe to call on the same list passed to the API.
    """
    if not anthropic_tools:
        return anthropic_tools
    out = copy.deepcopy(anthropic_tools)
    marker = _build_cache_marker(cache_ttl)
    last = out[-1]
    if isinstance(last, dict):
        last["cache_control"] = marker
    return out
