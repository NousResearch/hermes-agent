"""Route-scoped output ceilings shared by request construction and recovery."""

import hashlib
import time
from typing import Any, Optional

from agent.model_metadata import get_model_output_limit


def _positive_int(value: Any) -> Optional[int]:
    return value if type(value) is int and value > 0 else None


def _uses_catalog_output_budget(agent: Any) -> bool:
    mode = getattr(agent, "api_mode", None)
    if mode == "codex_responses":
        from agent.codex_responses_adapter import classify_responses_route

        # Match the transport's omission, including gateways using the Codex backend.
        return not classify_responses_route(agent).is_codex_backend
    return mode == "chat_completions"


def model_output_limit(agent: Any) -> Optional[int]:
    """Memoize metadata for this route, including misses, without mutating caller config."""
    mode = getattr(agent, "api_mode", None)
    if mode == "anthropic_messages":
        from agent.anthropic_adapter import _get_anthropic_max_output

        return _get_anthropic_max_output(getattr(agent, "model", None) or "")
    if not _uses_catalog_output_budget(agent):
        return None
    provider = getattr(agent, "provider", "") or ""
    base_url = getattr(agent, "base_url", "") or ""
    model = getattr(agent, "model", "") or ""
    api_key = getattr(agent, "api_key", "") or ""
    if not all(isinstance(value, str) for value in (provider, base_url, model, api_key)):
        return None
    key = (provider, base_url, model, hashlib.sha256(api_key.encode()).digest())
    now = time.monotonic()
    cached = getattr(agent, "_output_limit_cache", None)
    if isinstance(cached, tuple) and cached[0] == key and now < cached[2]:
        return cached[1]
    try:
        limit = get_model_output_limit(model, base_url=base_url, api_key=api_key, provider=provider)
    except Exception:
        limit = None  # metadata is advisory; an unavailable catalog must not block inference
    # A single-entry cache cannot grow with arbitrary model switches. Credential rotation
    # invalidates endpoint-scoped results; short negative TTL avoids probing on every retry.
    agent._output_limit_cache = (key, limit, now + (3600 if limit is not None else 60))
    return limit


def boosted_output_cap(agent: Any, requested_cap: Optional[int], n: int,
                       base: Optional[int] = None) -> Optional[int]:
    """Grow only within a known ceiling, preserving explicit caller budgets and unknowns."""
    limit = model_output_limit(agent)
    configured = _positive_int(getattr(agent, "max_tokens", None))
    requested_cap = _positive_int(requested_cap)
    if configured is not None:
        return min(configured, limit) if limit is not None else configured
    if limit is None:
        return requested_cap  # an omitted provider-owned budget is not a 4K budget
    anchor = requested_cap or limit
    return min(max((_positive_int(base) or anchor) * (2 ** n), anchor), limit)


def apply_output_budget(agent: Any, kwargs: dict) -> dict:
    """Fill or clamp OpenAI-compatible output caps after transport-specific construction.

    Native adapters keep their protocol-specific budgets (e.g. Anthropic thinking
    minimums). ChatGPT's OAuth Responses backend does not accept output caps.
    """
    mode = getattr(agent, "api_mode", None)
    if not _uses_catalog_output_budget(agent):
        return kwargs
    limit = model_output_limit(agent)
    if limit is None:
        return kwargs
    configured = _positive_int(getattr(agent, "max_tokens", None))
    if configured is not None and mode == "chat_completions":
        from agent.transports.chat_completions import _raise_gemini_thinking_max_tokens

        # The transport reserves output headroom for Gemini thinking. Preserve
        # that existing exception before bounding it by the published ceiling.
        configured = _raise_gemini_thinking_max_tokens(
            agent.model, getattr(agent, "reasoning_config", None), configured,
        )
    cap = min(limit, configured or limit)
    # extra_body is merged by the SDK after ordinary fields; account for both so a
    # retry cannot override a smaller request-level cap through a different spelling.
    fields = []
    for body in (kwargs, kwargs.get("extra_body")):
        if isinstance(body, dict):
            for name in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                value = _positive_int(body.get(name))
                if value is not None:
                    fields.append((body, name))
                    cap = min(cap, value)
    context = _positive_int(getattr(getattr(agent, "context_compressor", None), "context_length", None))
    if context is not None:
        from agent.chat_completion_helpers import estimate_request_context_tokens

        # Estimate the prepared wire request, including tool schemas and Responses
        # instructions. Existing overflow recovery remains authoritative if it differs.
        cap = min(cap, max(1, context - estimate_request_context_tokens(kwargs)))
    if fields:
        for body, name in fields:
            body[name] = cap
    elif mode == "codex_responses":
        kwargs["max_output_tokens"] = cap
    else:
        kwargs.update(agent._max_tokens_param(cap))
    return kwargs
