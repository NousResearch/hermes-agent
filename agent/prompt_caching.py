"""Anthropic prompt caching strategy.

Single layout: up to 4 explicit ``cache_control`` breakpoints. Hermes
prefers a stable tool-schema checkpoint, a cacheable system-prompt
checkpoint, and the most recent non-system messages.  All breakpoints
use the same TTL (5m or 1h).  This keeps reusable prefixes hot across
multi-turn sessions and across fresh gateway agents that share the same
tool/system prefix.

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


def build_cacheable_system_content(
    cacheable_text: str,
    volatile_text: str = "",
    ephemeral_text: str = "",
    cache_ttl: str = "5m",
) -> List[Dict[str, Any]]:
    """Build Anthropic system content blocks with only the stable prefix cached.

    Anthropic/Bedrock cache keys are prefix-based.  Putting memory snapshots,
    session IDs, timestamps, or per-call ephemeral instructions in the same
    cached system block makes otherwise identical turns miss.  This helper
    marks the stable/context portion and appends volatile text after the
    checkpoint so it still reaches the model without rewriting the cache.
    """
    blocks: List[Dict[str, Any]] = []
    if cacheable_text and cacheable_text.strip():
        blocks.append(
            {
                "type": "text",
                "text": cacheable_text,
                "cache_control": _build_marker(cache_ttl),
            }
        )
    for text in (volatile_text, ephemeral_text):
        if text and text.strip():
            blocks.append({"type": "text", "text": text.strip()})
    return blocks


def apply_anthropic_tool_cache_control(
    tools: Optional[List[Dict[str, Any]]],
    cache_ttl: str = "5m",
) -> Optional[List[Dict[str, Any]]]:
    """Return a deep-copied tools list with the last tool schema cached.

    Anthropic's tools array accepts ``cache_control`` on a tool definition.
    Marking the final tool checkpoints the entire schema list, letting the
    gateway reuse large stable tool definitions even when later system or
    message content changes.
    """
    if not tools:
        return tools

    cached_tools = copy.deepcopy(tools)
    marker = _build_marker(cache_ttl)
    for tool in reversed(cached_tools):
        if isinstance(tool, dict):
            tool["cache_control"] = dict(marker)
            break
    return cached_tools


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
    max_breakpoints: int = 4,
    cache_system: bool = True,
) -> List[Dict[str, Any]]:
    """Apply system_and_3 caching strategy to messages for Anthropic models.

    Places up to ``max_breakpoints`` cache_control markers. By default this is
    system prompt + last 3 non-system messages. Callers may reserve slots for
    tool-schema or pre-marked system checkpoints by lowering ``max_breakpoints``
    or setting ``cache_system=False``.

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages
    try:
        max_breakpoints = int(max_breakpoints)
    except Exception:
        max_breakpoints = 4
    max_breakpoints = max(0, min(4, max_breakpoints))
    if max_breakpoints == 0:
        return messages

    marker = _build_marker(cache_ttl)

    breakpoints_used = 0

    if cache_system and messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = max_breakpoints - breakpoints_used
    if remaining > 0:
        non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
        for idx in non_sys[-remaining:]:
            _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages
