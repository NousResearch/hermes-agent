"""Anthropic prompt caching strategy.

Two strategies:

- ``system_and_3`` (default, legacy): 4 cache_control breakpoints — system
  prompt + last 3 non-system messages, all at the same TTL (5m or 1h).

- ``system_and_3_double_tail`` (new): builds on ``system_and_3`` but places
  TWO cache markers on the final non-system message (both its last content
  block AND the message itself at the top level) so that if conversation
  growth pushes the first marker out of the cache window, the second
  still hits. Only meaningful for the native Anthropic format
  (native_anthropic=True).

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


def _get_non_system_indices(messages: List[Dict[str, Any]]) -> List[int]:
    """Return indices of all non-system messages."""
    return [i for i in range(len(messages)) if messages[i].get("role") != "system"]


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
    non_sys = _get_non_system_indices(messages)
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages


def apply_anthropic_cache_control_v2(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    strategy: str = "system_and_3",
    native_anthropic: bool = False,
) -> List[Dict[str, Any]]:
    """Apply caching strategy to messages for Anthropic models (v2).

    Supports two strategies:

    - ``system_and_3``: 4 cache_control breakpoints — system prompt +
      last 3 non-system messages.  Identical to
      ``apply_anthropic_cache_control``.

    - ``system_and_3_double_tail``: Same as ``system_and_3``, but the
      **last** non-system message gets an additional top-level
      ``cache_control`` so it carries two markers (content-block +
      message-level).  When the conversation grows and the
      content-block marker drifts out of the cache window, the
      top-level marker still provides a cache hit.  Only effective
      for native Anthropic format (``native_anthropic=True``).

    Args:
        api_messages: Conversation messages in OpenAI/Anthropic format.
        cache_ttl: Cache lifetime — ``"5m"`` or ``"1h"``.
        strategy: ``"system_and_3"`` (default) or ``"system_and_3_double_tail"``.
        native_anthropic: If True, use native Anthropic message shape for
            tool messages and top-level cache_control.

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    if strategy not in ("system_and_3", "system_and_3_double_tail"):
        raise ValueError(
            f"Unknown prompt caching strategy: {strategy!r}. "
            f"Expected 'system_and_3' or 'system_and_3_double_tail'."
        )

    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = _build_marker(cache_ttl)

    breakpoints_used = 0

    # System message always gets a breakpoint (if present).
    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = 4 - breakpoints_used
    non_sys = _get_non_system_indices(messages)

    # Apply markers to the last N non-system messages.
    tail_indices = non_sys[-remaining:] if remaining > 0 else []
    for idx in tail_indices:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    # --- Double-tail: add a second (top-level) marker on the very last message.
    if strategy == "system_and_3_double_tail" and tail_indices:
        last_idx = tail_indices[-1]
        last_msg = messages[last_idx]
        # Top-level cache_control alongside the content-block marker
        # set by _apply_cache_marker above. Only meaningful for native
        # Anthropic; on OpenRouter the top-level field is harmless
        # (the adapter usually strips it), but we gate on
        # native_anthropic to match the intent.
        if native_anthropic or last_msg.get("role") == "tool":
            last_msg["cache_control"] = copy.deepcopy(marker)

    return messages
