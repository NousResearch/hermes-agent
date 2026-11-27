"""The window a request in flight keeps warm while its reply generates (``PromptCacheWarmer.start_in_flight``)."""

from __future__ import annotations

import time
from typing import Any, Optional


def in_flight_window(agent: Any, api_kwargs: dict[str, Any]) -> Optional[tuple[float, float]]:
    """``(deadline, tier seconds)`` of the entry the request going out now writes or reads: from its send,
    for the tier its payload asks under the route's rule and ceiling (``requested_cache_tier``; the
    clock's tier when it declares none). A request with ``cache_control`` markers declares its tier; an
    automatic cache must have proven itself on the clock first (a proven or declared window, expired or
    not). ``None`` otherwise."""
    from agent.prompt_cache_clock import DECLARED, PROVEN, attempt_facts, cache_window, requested_cache_tier
    from agent.prompt_caching import TTL_SECONDS

    window = cache_window(agent)
    facts = getattr(agent, "_prompt_cache_attempt", None) or attempt_facts(api_kwargs) or {}
    proven = window is not None and window.confidence in (PROVEN, DECLARED)
    if not proven and facts.get("marker_ttl") is None:
        return None
    tier = requested_cache_tier(
        api_kwargs, provider=getattr(agent, "provider", "") or "", base_url=getattr(agent, "base_url", "") or "",
        api_mode=getattr(agent, "api_mode", "") or "", model=getattr(agent, "model", "") or "",
    ) or getattr(window, "tier", None)
    ttl_s = TTL_SECONDS.get(tier or "")
    if not ttl_s:
        return None
    return float(facts.get("sent_at") or time.time()) + ttl_s, float(ttl_s)


def request_split(api_kwargs: dict[str, Any]) -> tuple[int, int]:
    """``(cached, uncached)`` rough prompt tokens of a request payload once it is sent: with
    ``cache_control`` markers, the system prompt, tools and messages through the last marked one are
    cached and the messages after it (a route that leaves tool results unmarked) are re-billed as plain
    input on every replay; without markers (an automatic cache) the whole prompt is cached."""
    from agent.chat_completion_helpers import estimate_request_context_tokens

    messages = list(api_kwargs.get("messages") or [])
    system = api_kwargs.get("system")
    head = estimate_request_context_tokens(list(api_kwargs.get("tools") or [])) + (
        estimate_request_context_tokens([system]) if system else 0)
    marked = [i for i, message in enumerate(messages) if _carries_marker(message)]
    if not marked and not _carries_marker(system) and not _carries_marker(api_kwargs.get("tools")):
        return head + estimate_request_context_tokens(messages), 0
    cut = marked[-1] + 1 if marked else 0
    return head + estimate_request_context_tokens(messages[:cut]), estimate_request_context_tokens(messages[cut:])


def _carries_marker(value: Any) -> bool:
    if isinstance(value, dict):
        return "cache_control" in value or any(_carries_marker(v) for v in value.values())
    if isinstance(value, list):
        return any(_carries_marker(v) for v in value)
    return False
