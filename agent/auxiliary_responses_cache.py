"""Prompt-cache routing for auxiliary Responses requests."""

from typing import Any


def add_responses_cache_kwargs(
    kwargs: dict[str, Any], *, model: str, host: str, scope_id: str,
    is_xai: bool, is_github: bool,
) -> None:
    """Apply the main transport's static-prefix hash and endpoint retention policy."""
    from agent.transports.codex import (
        _cache_scope_from_session_id,
        _content_cache_key,
        _default_prompt_cache_retention_for_request,
    )

    if not (is_xai or is_github) and "prompt_cache_key" not in kwargs:
        scope = _cache_scope_from_session_id(scope_id)
        cache_key = _content_cache_key(kwargs["instructions"], kwargs.get("tools"), scope)
        if cache_key:
            kwargs["prompt_cache_key"] = cache_key
    if "prompt_cache_retention" not in kwargs:
        retention = _default_prompt_cache_retention_for_request(model, host)
        if retention:
            kwargs["prompt_cache_retention"] = retention
