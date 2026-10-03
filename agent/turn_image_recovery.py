"""Request-aware eligibility for reactive image recovery."""
from __future__ import annotations

from typing import Any

from agent.error_classifier import FailoverReason, _has_large_inline_image


def dropped_large_image_upload(error: Exception, classified: Any, api_messages: Any) -> bool:
    """A connection drop can hide a body-size rejection, not prove one.

    Keep the original verdict for normal backoff/fallback if resizing cannot help.
    Explicit timeouts (including read/generation waits), HTTP errors and small or
    remote images are not evidence for trying a smaller upload.
    """
    from openai import APIConnectionError, APITimeoutError

    if (
        classified.reason != FailoverReason.timeout
        or getattr(error, "status_code", None) is not None
        or not isinstance(error, APIConnectionError)
        or isinstance(error, APITimeoutError)
    ):
        return False
    return any(
        _has_large_inline_image(message.get("content"))
        for message in api_messages if isinstance(message, dict)
    ) if isinstance(api_messages, list) else False
