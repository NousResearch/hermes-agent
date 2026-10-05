"""Prompt-cache routing for auxiliary Responses requests."""

from typing import Any


def add_responses_cache_kwargs(
    kwargs: dict[str, Any], *, model: str, host: str, scope_id: str,
    is_xai: bool, is_github: bool, extra_body: Any = None,
) -> None:
    """Apply the main transport's static-prefix hash and endpoint retention policy."""
    from agent.transports.codex import (
        _bound_prompt_cache_key_field,
        _cache_scope_from_session_id,
        _content_cache_key,
        _default_prompt_cache_retention_for_request,
    )

    if not is_github and "prompt_cache_key" not in kwargs:
        scope = _cache_scope_from_session_id(scope_id)
        cache_key = _content_cache_key(kwargs["instructions"], kwargs.get("tools"), scope)
        if is_xai:
            # xAI reads the routing key from the body. Keep a caller's explicit key,
            # including an empty opt-out; unscoped aux calls must not share one slot.
            body = dict(extra_body) if isinstance(extra_body, dict) else {}
            key = body.get("prompt_cache_key", cache_key if scope else None)
            if key is not None:
                body["prompt_cache_key"] = key
            _bound_prompt_cache_key_field(body)
            # Keep wire fields even when an unscoped or opted-out request has no key.
            if body:
                kwargs["extra_body"] = body
        elif cache_key:
            kwargs["prompt_cache_key"] = cache_key
    if "prompt_cache_retention" not in kwargs:
        retention = _default_prompt_cache_retention_for_request(model, host)
        if retention:
            kwargs["prompt_cache_retention"] = retention
