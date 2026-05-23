"""Map Cursor SDK usage payloads to Hermes token accounting."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _pick_int(mapping: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        if key not in mapping:
            continue
        try:
            return int(mapping[key] or 0)
        except (TypeError, ValueError):
            continue
    return 0


def cursor_turn_usage_to_hermes(usage: Mapping[str, Any] | None) -> Optional[dict[str, int]]:
    """Convert Cursor ``turn-ended`` usage to Hermes ``update_from_response`` shape."""
    if not usage:
        return None

    input_tokens = _pick_int(usage, "inputTokens", "input_tokens")
    output_tokens = _pick_int(usage, "outputTokens", "output_tokens")
    cache_read = _pick_int(usage, "cacheReadTokens", "cache_read_tokens")
    cache_write = _pick_int(usage, "cacheWriteTokens", "cache_write_tokens")

    # Cursor reports total prompt-side context in inputTokens; cache fields are
    # breakdowns for billing, not additive on top of inputTokens.
    prompt_tokens = input_tokens if input_tokens else (cache_read + cache_write)
    total_tokens = prompt_tokens + output_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
    }


def apply_cursor_usage_to_agent(
    agent, usage: Mapping[str, Any] | None
) -> Optional[dict[str, int]]:
    """Update Hermes session + compressor from Cursor turn usage."""
    if not usage:
        return None
    if "inputTokens" in usage or "input_tokens" in usage:
        usage_dict = cursor_turn_usage_to_hermes(usage)
    else:
        usage_dict = {
            "prompt_tokens": _pick_int(usage, "prompt_tokens"),
            "completion_tokens": _pick_int(usage, "completion_tokens"),
            "total_tokens": _pick_int(usage, "total_tokens"),
            "input_tokens": _pick_int(usage, "input_tokens"),
            "output_tokens": _pick_int(usage, "output_tokens"),
            "cache_read_tokens": _pick_int(usage, "cache_read_tokens"),
            "cache_write_tokens": _pick_int(usage, "cache_write_tokens"),
        }
    if not usage_dict or not usage_dict.get("prompt_tokens"):
        return None

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        compressor.update_from_response(usage_dict)

    prompt_tokens = usage_dict["prompt_tokens"]
    completion_tokens = usage_dict["completion_tokens"]

    # inputTokens from turn-ended is the agent's current context occupancy.
    agent.session_prompt_tokens = prompt_tokens
    agent.session_completion_tokens = (
        getattr(agent, "session_completion_tokens", 0) + completion_tokens
    )
    agent.session_total_tokens = prompt_tokens + agent.session_completion_tokens
    agent.session_api_calls = getattr(agent, "session_api_calls", 0) + 1
    agent.session_input_tokens = usage_dict.get("input_tokens", 0)
    agent.session_output_tokens = (
        getattr(agent, "session_output_tokens", 0) + completion_tokens
    )
    agent.session_cache_read_tokens = usage_dict.get("cache_read_tokens", 0)
    agent.session_cache_write_tokens = usage_dict.get("cache_write_tokens", 0)

    return usage_dict
