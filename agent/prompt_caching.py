"""Anthropic prompt caching strategies.

Two layouts:

* ``system_and_3`` (default, used everywhere except the long-lived path):
  4 cache_control breakpoints — system prompt + last 3 non-system messages.
  All at the same TTL (5m or 1h). Reduces input token costs by ~75% on
  multi-turn conversations within a single session.

* ``prefix_and_2`` (Claude on Anthropic / OpenRouter / Nous Portal):
  4 breakpoints split across two TTL tiers — tools[-1] (1h) +
  stable system prefix (1h) + last 2 non-system messages (5m). The
  long-lived prefix is byte-stable across sessions for a given user
  config, so every fresh session reads the cached system+tools instead
  of re-paying for them. Within-session rolling window shrinks from 3
  messages to 2 to free the breakpoint budget.

Pure functions -- no class state, no AIAgent dependency.
"""

import copy
from typing import Any, Dict, List, Optional


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

    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = 4 - breakpoints_used
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages


def _mark_system_stable_block(
    messages: List[Dict[str, Any]],
    long_lived_marker: Dict[str, str],
) -> bool:
    """Mark the *first* content block of the system message with the 1h marker.

    The system message is expected to have been split into multiple content
    blocks beforehand by the caller — block[0] is the cross-session-stable
    prefix, subsequent blocks carry context files + volatile suffix.
    Falls back to marking the whole system message as a single block when
    the message hasn't been split (preserves correctness on the fallback path).

    Returns True when a marker was placed.
    """
    if not messages or messages[0].get("role") != "system":
        return False

    sys_msg = messages[0]
    content = sys_msg.get("content")

    # Already a list of blocks → mark the first block.
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            first["cache_control"] = long_lived_marker
            return True
        return False

    # String content (no split) → cannot place a stable-prefix breakpoint
    # without changing the byte content.  Caller is responsible for
    # splitting; if they didn't, fall through to envelope marker so we still
    # cache *something* for this turn.
    if isinstance(content, str) and content:
        sys_msg["content"] = [
            {"type": "text", "text": content, "cache_control": long_lived_marker}
        ]
        return True

    return False


def apply_anthropic_cache_control_long_lived(
    api_messages: List[Dict[str, Any]],
    long_lived_ttl: str = "1h",
    rolling_ttl: str = "5m",
    native_anthropic: bool = False,
) -> List[Dict[str, Any]]:
    """Apply prefix_and_2 caching: long-lived stable prefix + rolling window.

    Layout (4 breakpoints total):
      * Stable system prefix (block[0]) → ``long_lived_ttl`` TTL
      * Last 2 non-system messages → ``rolling_ttl`` TTL each

    NOTE: this function does NOT mark the tools array. Tools cache_control
    is attached separately (see ``mark_tools_for_long_lived_cache``) because
    tools live outside the messages list in the API payload.

    The caller MUST have split the system message into ordered content
    blocks where block[0] is the cross-session-stable portion. If the system
    message is still a single string, it is wrapped into a single block and
    marked — this is correct, just less effective (the volatile suffix is
    not isolated, so the prefix invalidates per-session).

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    long_marker = _build_marker(long_lived_ttl)
    rolling_marker = _build_marker(rolling_ttl)

    placed_prefix = _mark_system_stable_block(messages, long_marker)

    # Reserve 1 breakpoint for the system prefix (when placed); spend the
    # remaining 3 on the rolling tail.  Anthropic max is 4 total —
    # tools[-1] (when marked) consumes the 4th, so we cap rolling at 2 here.
    rolling_budget = 2 if placed_prefix else 3
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-rolling_budget:]:
        _apply_cache_marker(messages[idx], rolling_marker, native_anthropic=native_anthropic)

    return messages


def mark_tools_for_long_lived_cache(
    tools: Optional[List[Dict[str, Any]]],
    long_lived_ttl: str = "1h",
) -> Optional[List[Dict[str, Any]]]:
    """Attach cache_control to the last tool in the OpenAI-format tools list.

    Anthropic prefix-cache order is ``tools → system → messages``.  Marking
    the last tool dict caches the entire tools array (Anthropic's docs:
    "the marker is placed on the last block you want included in the cached
    prefix").  Marker is preserved across the OpenAI-wire boundary on
    OpenRouter and Nous Portal (which proxies to OpenRouter); on native
    Anthropic the marker is forwarded by ``convert_tools_to_anthropic``.

    Returns a deep copy of the tools list with the marker attached, or the
    input unchanged when tools is empty/None.  Pure function — does not
    mutate the input.
    """
    if not tools:
        return tools
    out = copy.deepcopy(tools)
    last = out[-1]
    if isinstance(last, dict):
        last["cache_control"] = _build_marker(long_lived_ttl)
    return out
