"""Anthropic prompt caching strategies.

Two layouts:

* ``system_and_3`` (default) — 4 cache_control breakpoints: system
  prompt + last 3 non-system messages, all at the same TTL.

* ``pi_dual`` — 3 targeted breakpoints: system prompt, last assistant
  ``tool_use`` block, and last user message.  Excludes ``role: tool``
  messages entirely.  Inspired by PI/OpenClaw's cache extension.

Pure functions -- no class state, no AIAgent dependency.
"""

import copy
from typing import Any, Dict, List

_VALID_STRATEGIES = {"system_and_3", "pi_dual"}


def _apply_cache_marker(
    msg: dict, cache_marker: dict, native_anthropic: bool = False
) -> None:
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


def _apply_content_block_marker(
    msg: dict, block_index: int, cache_marker: dict
) -> None:
    """Add cache_control to a specific content block within a message."""
    content = msg.get("content")
    if isinstance(content, list) and 0 <= block_index < len(content):
        block = content[block_index]
        if isinstance(block, dict):
            block["cache_control"] = cache_marker


def _mark_last_content_block(msg: dict, cache_marker: dict) -> None:
    """Add cache_control to the last content block, normalizing string content."""
    content = msg.get("content")
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


def _apply_system_and_3(
    messages: List[Dict[str, Any]],
    marker: Dict[str, str],
    native_anthropic: bool,
) -> List[Dict[str, Any]]:
    """Original strategy: system + last 3 non-system messages (4 breakpoints max)."""
    breakpoints_used = 0

    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = 4 - breakpoints_used
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages


def _apply_pi_dual(
    messages: List[Dict[str, Any]],
    marker: Dict[str, str],
    native_anthropic: bool,
) -> List[Dict[str, Any]]:
    """PI-style dual strategy: system + last assistant tool_use + last user (3 breakpoints).

    Never marks role:tool messages.  When native_anthropic is False,
    falls back to system_and_3 to avoid regressions on OpenRouter.
    """
    if not native_anthropic:
        return _apply_system_and_3(messages, marker, native_anthropic)

    # 1. Mark system prompt
    if messages and messages[0].get("role") == "system":
        _mark_last_content_block(messages[0], marker)

    # 2. Find and mark last assistant message containing a tool_use block
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        last_tool_use_idx = None
        for j in range(len(content) - 1, -1, -1):
            block = content[j]
            if isinstance(block, dict) and block.get("type") == "tool_use":
                last_tool_use_idx = j
                break
        if last_tool_use_idx is not None:
            _apply_content_block_marker(msg, last_tool_use_idx, marker)
            break

    # 3. Find and mark last user message
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user":
            _mark_last_content_block(msg, marker)
            break

    return messages


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
    strategy: str = "system_and_3",
) -> List[Dict[str, Any]]:
    """Apply caching strategy to messages for Anthropic models.

    Strategies:
        system_and_3: System prompt + last 3 non-system messages (4 breakpoints).
        pi_dual: System prompt + last assistant tool_use + last user (3 breakpoints).
                 Falls back to system_and_3 when native_anthropic is False.

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = _build_marker(cache_ttl)

    if strategy not in _VALID_STRATEGIES:
        strategy = "system_and_3"

    if strategy == "pi_dual":
        return _apply_pi_dual(messages, marker, native_anthropic)
    return _apply_system_and_3(messages, marker, native_anthropic)
