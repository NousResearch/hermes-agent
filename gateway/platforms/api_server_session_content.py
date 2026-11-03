"""Session chat input parsing shared by the JSON and SSE endpoints."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web


def session_chat_user_message(
    body: dict[str, Any], *, param: str = "message",
) -> tuple[Any, web.Response | None]:
    """Preserve plain strings while retaining structured-content validation and bounds."""
    from gateway.platforms.api_server import (
        _content_has_visible_payload, _error_response,
        _multimodal_validation_error, _normalize_multimodal_content,
    )

    user_message = body.get("message") or body.get("input")
    if not _content_has_visible_payload(user_message):
        return None, _error_response("Missing 'message' field", 400, code="missing_message")
    # Session chat accepts a plain text turn directly; structured parts retain their caps.
    if isinstance(user_message, str):
        return user_message, None
    try:
        return _normalize_multimodal_content(user_message), None
    except ValueError as exc:
        return None, _multimodal_validation_error(exc, param=param)
