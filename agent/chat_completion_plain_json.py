"""Plain-JSON answers to streaming chat-completion requests.

Some OpenAI-compatible endpoints — local routers in particular — ignore
``stream=True`` and answer with one plain JSON completion body
(``Content-Type: application/json``) instead of an SSE stream. The SDK
still hands back a stream object for that response; iterating it yields
zero chunks, so the turn dies as an empty stream even though the
provider served a complete response.

The completed-response handling that already exists for adapter shims
(Relay's ``completed_response_predicate`` on the main wire, the
``hasattr(..., "choices")`` checks on the auxiliary seams) only engages
when the returned object itself carries ``choices``. The helpers here
close that gap at the seam where the SDK stream is created: when the
response is a plain JSON completion, parse the body once and return
the completed response in place of the stream, so the existing
handling adopts it like any adapter-returned final response.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _plain_json_response(stream: Any) -> Any | None:
    """The HTTP response behind `stream` when it is plain JSON, else None."""
    response = getattr(stream, "response", None)
    if response is None:
        return None
    content_type = (response.headers.get("content-type") or "").lower()
    if "event-stream" in content_type or "json" not in content_type:
        return None
    return response


def _completion_from_payload(payload: Any) -> Any | None:
    """Validate a decoded body as a chat completion; None when it is not one
    (a JSON error body keeps the existing empty-stream failure)."""
    if not isinstance(payload, dict) or not payload.get("choices"):
        return None
    try:
        from openai.types.chat import ChatCompletion

        return ChatCompletion.model_validate(payload)
    except Exception:
        logger.debug("Plain-JSON streaming response was not a chat completion", exc_info=True)
        return None


def completed_response_or_stream(stream: Any) -> Any:
    """Return the completed response when `stream` is a plain-JSON answer
    to a streaming request; otherwise return `stream` unchanged."""
    response = _plain_json_response(stream)
    if response is None:
        return stream
    try:
        response.read()
        payload = response.json()
    except Exception:
        return stream
    completed = _completion_from_payload(payload)
    if completed is None:
        return stream
    logger.info(
        "Streaming request answered with a plain JSON completion instead of an SSE stream; "
        "treating it as a completed response.")
    with contextlib.suppress(Exception):
        stream.close()
    return completed


async def acompleted_response_or_stream(stream: Any) -> Any:
    """Async twin of :func:`completed_response_or_stream`."""
    response = _plain_json_response(stream)
    if response is None:
        return stream
    try:
        await response.aread()
        payload = response.json()
    except Exception:
        return stream
    completed = _completion_from_payload(payload)
    if completed is None:
        return stream
    logger.info(
        "Streaming request answered with a plain JSON completion instead of an SSE stream; "
        "treating it as a completed response.")
    with contextlib.suppress(Exception):
        await stream.close()
    return completed
