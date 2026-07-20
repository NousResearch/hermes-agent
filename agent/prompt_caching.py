"""Anthropic prompt caching strategy.

Single layout: ``system_and_3``. 4 cache_control breakpoints — system
prompt + last 3 non-system messages, all at the same TTL (5m or 1h).
Reduces input token costs by ~75% on multi-turn conversations within a
single session.

Pure functions -- no class state, no AIAgent dependency.

Anthropic enforces a hard maximum of 4 cache breakpoints per request.
Do not attempt to exceed that without an explicit provider capability
change — extra markers are silently dropped or rejected.
"""

import copy
from typing import Any, Dict, List

# Anthropic API hard limit — documented for desk/doctor consumers.
ANTHROPIC_MAX_CACHE_BREAKPOINTS = 4
DEFAULT_CACHE_LAYOUT = "system_and_3"


def describe_cache_layout(layout: str = DEFAULT_CACHE_LAYOUT) -> Dict[str, Any]:
    """Describe the active caching layout (for /status, Model Desk, tests)."""
    name = (layout or DEFAULT_CACHE_LAYOUT).strip() or DEFAULT_CACHE_LAYOUT
    return {
        "ok": True,
        "layout": name,
        "max_breakpoints": ANTHROPIC_MAX_CACHE_BREAKPOINTS,
        "strategy": (
            "Place cache_control on the system message (if present) plus the "
            "last N non-system messages where N = max_breakpoints - system_used."
        ),
        "cache_safe_rules": [
            "Never mutate the stable system prompt mid-conversation",
            "Inject volatile context into the user message (pre_llm_call context)",
            "Defer toolset/skill changes to next session unless --now",
        ],
    }


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
    *,
    max_breakpoints: int = ANTHROPIC_MAX_CACHE_BREAKPOINTS,
) -> List[Dict[str, Any]]:
    """Apply system_and_3 caching strategy to messages for Anthropic models.

    Places up to ``max_breakpoints`` cache_control markers (capped at the
    Anthropic hard limit of 4): system prompt + last N non-system messages,
    all at the same TTL.

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = _build_marker(cache_ttl)
    cap = max(
        1,
        min(
            int(max_breakpoints or ANTHROPIC_MAX_CACHE_BREAKPOINTS),
            ANTHROPIC_MAX_CACHE_BREAKPOINTS,
        ),
    )

    breakpoints_used = 0

    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = cap - breakpoints_used
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages


def count_cache_markers(messages: List[Dict[str, Any]]) -> int:
    """Count cache_control markers in a message list (for tests/diagnostics)."""
    count = 0
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("cache_control"):
            count += 1
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("cache_control"):
                    count += 1
                    break
    return count
