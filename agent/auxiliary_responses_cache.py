"""Prompt-cache routing for auxiliary Responses requests."""

from typing import Any


def add_responses_cache_kwargs(
    kwargs: dict[str, Any], *, model: str, host: str, scope_id: str,
    is_xai: bool, prompt_cache_key: bool, extra_body: Any = None,
) -> None:
    """Apply the main transport's static-prefix hash and endpoint retention policy."""
    from agent.transports.codex import (
        _bound_prompt_cache_key_field,
        _cache_scope_from_session_id,
        _content_cache_key,
        _default_prompt_cache_retention_for_request,
    )

    cache_key = None
    if prompt_cache_key and "prompt_cache_key" not in kwargs:
        scope = _cache_scope_from_session_id(scope_id)
        cache_key = _content_cache_key(kwargs["instructions"], kwargs.get("tools"), scope)
        if is_xai and not scope:
            cache_key = None  # Unscoped auxiliary calls must not share one slot.
        elif not is_xai and cache_key:
            kwargs["prompt_cache_key"] = cache_key
    if is_xai:
        # Endpoint opt-out disables routing, not unrelated caller wire fields.
        body = dict(extra_body) if isinstance(extra_body, dict) else {}
        if prompt_cache_key:
            key = body.get("prompt_cache_key", cache_key)
            if key is not None:
                body["prompt_cache_key"] = key
            _bound_prompt_cache_key_field(body)
        else:
            body.pop("prompt_cache_key", None)
        if body:
            kwargs["extra_body"] = body
    if "prompt_cache_retention" not in kwargs:
        retention = _default_prompt_cache_retention_for_request(model, host)
        if retention:
            kwargs["prompt_cache_retention"] = retention
