"""Image helpers extracted from AIAgent for modularity."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def api_kwargs_have_image_parts(api_kwargs: dict) -> bool:
    """Return True when the outbound request still contains native image parts."""
    if not isinstance(api_kwargs, dict):
        return False
    candidates = []
    messages = api_kwargs.get("messages")
    if isinstance(messages, list):
        candidates.extend(messages)
    # Responses API payloads use `input`; after conversion, image parts can
    # still be present there instead of in `messages`.
    response_input = api_kwargs.get("input")
    if isinstance(response_input, list):
        candidates.extend(response_input)

    return any(contains_image(item) for item in candidates)


def contains_image(value: Any) -> bool:
    """Recursively check if a value contains image parts (dict or list)."""
    if isinstance(value, dict):
        ptype = value.get("type")
        if ptype in {"image_url", "input_image"}:
            return True
        return any(contains_image(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_image(v) for v in value)
    return False


def content_has_image_parts(content: Any) -> bool:
    """Check if content (list of parts) contains image parts."""
    if not isinstance(content, list):
        return False
    for part in content:
        if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
            return True
    return False


def try_shrink_image_parts_in_messages(api_messages: list) -> bool:
    """Forwarder — see ``agent.conversation_compression.try_shrink_image_parts_in_messages``."""
    from agent.conversation_compression import try_shrink_image_parts_in_messages as _impl
    return _impl(api_messages)
