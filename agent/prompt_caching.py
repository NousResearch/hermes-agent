"""Anthropic prompt caching — multi-block system prompt with tiered TTLs.

v2 improvements over original system_and_3 strategy:
  - Multi-block system prompt: static(1h) + session(5m) + ephemeral(none)
  - Tool definition caching: breakpoint on last tool
  - Tiered TTLs: 1h for cross-session stable content, 5m for session-stable
  - Cache metrics extraction and aggregation

Breakpoint budget (max 4):
  1. Tools (last tool definition)
  2. System block 1: identity + skills + guidance (1h TTL)
  3. System block 2: memory + context files (5m TTL)
  4. Last non-system message (rolling window)

Pure functions -- no class state, no AIAgent dependency.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


# ============================================================
# Data structures
# ============================================================

@dataclass
class SystemPromptBlock:
    """A segment of the system prompt with its own caching policy.

    Attributes:
        text: The prompt text for this block.
        label: Human-readable label (e.g., 'static', 'session', 'ephemeral').
        cache_ttl: Cache TTL — '1h', '5m', or None (no caching).
    """
    text: Optional[str]
    label: str
    cache_ttl: Optional[str] = None


@dataclass
class CacheMetrics:
    """Metrics from a single API call's cache usage.

    Attributes:
        input_tokens: Total input tokens for the request.
        cache_read_tokens: Tokens served from cache (90% discount).
        cache_write_tokens: Tokens written to cache (25% or 100% markup).
    """
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage (0-100)."""
        if self.input_tokens <= 0:
            return 0.0
        return (self.cache_read_tokens / self.input_tokens) * 100

    @property
    def estimated_savings_tokens(self) -> float:
        """Net token-equivalent savings from caching.

        Savings = cache_read * 0.9 (90% discount) - cache_write * 0.25 (25% markup).
        """
        savings = self.cache_read_tokens * 0.9
        extra_cost = self.cache_write_tokens * 0.25
        return savings - extra_cost


@dataclass
class AggregatedCacheMetrics:
    """Aggregated cache metrics across multiple turns."""
    total_input_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    turn_count: int = 0

    @property
    def overall_hit_rate(self) -> float:
        if self.total_input_tokens <= 0:
            return 0.0
        return (self.total_cache_read_tokens / self.total_input_tokens) * 100


# ============================================================
# v1 helpers (preserved for backward compatibility)
# ============================================================

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
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"

    breakpoints_used = 0

    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = 4 - breakpoints_used
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages


# ============================================================
# v2: Multi-block system prompt with tiered TTLs
# ============================================================

def _make_cache_marker(ttl: Optional[str]) -> Optional[Dict[str, str]]:
    """Build a cache_control marker dict for the given TTL, or None."""
    if ttl is None:
        return None
    marker = {"type": "ephemeral"}
    if ttl == "1h":
        marker["ttl"] = "1h"
    # "5m" is the default — no ttl field needed
    return marker


def build_system_content_blocks(
    blocks: List[SystemPromptBlock],
) -> List[Dict[str, Any]]:
    """Convert SystemPromptBlocks to Anthropic content block format.

    Each block becomes a {"type": "text", "text": ..., "cache_control"?: ...} dict.
    Empty/None text blocks are skipped.

    Returns:
        List of Anthropic content block dicts suitable for system message content.
    """
    result = []
    for block in blocks:
        if not block.text:
            continue
        entry: Dict[str, Any] = {"type": "text", "text": block.text}
        marker = _make_cache_marker(block.cache_ttl)
        if marker is not None:
            entry["cache_control"] = marker
        result.append(entry)
    return result


def _count_system_breakpoints(system_content: Any) -> int:
    """Count pre-placed cache_control breakpoints in a system message's content."""
    if not isinstance(system_content, list):
        return 0
    count = 0
    for block in system_content:
        if isinstance(block, dict) and "cache_control" in block:
            count += 1
    return count


def _system_is_pre_structured(system_content: Any) -> bool:
    """Check if system content is a pre-structured list of content blocks."""
    if not isinstance(system_content, list):
        return False
    if not system_content:
        return False
    # A pre-structured system has list-of-dicts with "type" keys
    return isinstance(system_content[0], dict) and "type" in system_content[0]


def apply_anthropic_cache_control_v2(
    api_messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """Apply multi-block caching strategy with tool support.

    Handles:
    - Pre-structured system blocks (list content with cache_control already set)
    - Tool definition caching (breakpoint on last tool)
    - Remaining budget allocated to last N non-system messages

    Args:
        api_messages: Messages in OpenAI chat format.
        tools: Optional tool definitions to cache.
        cache_ttl: Default TTL for non-pre-structured content.
        native_anthropic: True when using native Anthropic API.

    Returns:
        If tools provided: (messages, tools) tuple with cache_control injected.
        If no tools: messages list with cache_control injected.
    """
    if not api_messages:
        if tools:
            return ([], copy.deepcopy(tools))
        return []

    messages = copy.deepcopy(api_messages)
    cached_tools = copy.deepcopy(tools) if tools else None

    breakpoints_used = 0

    # --- Tool caching: breakpoint on last tool definition ---
    if cached_tools:
        cached_tools[-1]["cache_control"] = {"type": "ephemeral"}
        breakpoints_used += 1

    # --- System message handling ---
    if messages and messages[0].get("role") == "system":
        sys_content = messages[0].get("content")

        if _system_is_pre_structured(sys_content):
            # Pre-structured blocks: count existing breakpoints, don't re-mark
            breakpoints_used += _count_system_breakpoints(sys_content)
        else:
            # Legacy string content: wrap and mark like v1
            marker = _make_cache_marker(cache_ttl) or {"type": "ephemeral"}
            _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
            breakpoints_used += 1

    # --- Message caching: remaining budget goes to last N non-system messages ---
    remaining = max(0, 4 - breakpoints_used)
    if remaining > 0:
        non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
        for idx in non_sys[-remaining:]:
            marker = _make_cache_marker(cache_ttl) or {"type": "ephemeral"}
            _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    if cached_tools is not None:
        return (messages, cached_tools)
    return messages


# ============================================================
# Phase 4: Cache metrics extraction
# ============================================================

def extract_cache_metrics(usage: Any, api_mode: str = "anthropic_messages") -> CacheMetrics:
    """Extract cache metrics from an API response's usage object.

    Handles both Anthropic native and OpenRouter response formats.

    Args:
        usage: The usage object from the API response.
        api_mode: 'anthropic_messages' or 'openrouter' (or any other).

    Returns:
        CacheMetrics with extracted values.
    """
    if api_mode == "anthropic_messages":
        return CacheMetrics(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
    else:
        # OpenRouter / OpenAI format
        details = getattr(usage, "prompt_tokens_details", None)
        return CacheMetrics(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            cache_read_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
            cache_write_tokens=(getattr(details, "cache_write_tokens", 0) or 0) if details else 0,
        )


def aggregate_cache_metrics(metrics_list: List[CacheMetrics]) -> AggregatedCacheMetrics:
    """Aggregate cache metrics across multiple API calls.

    Returns:
        AggregatedCacheMetrics with summed totals and overall hit rate.
    """
    agg = AggregatedCacheMetrics()
    for m in metrics_list:
        agg.total_input_tokens += m.input_tokens
        agg.total_cache_read_tokens += m.cache_read_tokens
        agg.total_cache_write_tokens += m.cache_write_tokens
        agg.turn_count += 1
    return agg
