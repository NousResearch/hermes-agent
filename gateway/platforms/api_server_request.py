"""Request-shape helpers; limits late-bind the API facade's supported patch points."""

from __future__ import annotations

from typing import Any, Optional


def _clean_request_string(value: Any) -> Optional[str]:
    """Return a stripped request string, or None for absent/non-string values."""
    return (value.strip() or None) if isinstance(value, str) else None


def _cap_text(text: str) -> str:
    from gateway.platforms.api_server import MAX_NORMALIZED_TEXT_LENGTH
    return text[:MAX_NORMALIZED_TEXT_LENGTH] if len(text) > MAX_NORMALIZED_TEXT_LENGTH else text


def _cap_list(items: list) -> list:
    from gateway.platforms.api_server import MAX_CONTENT_LIST_SIZE
    return items[:MAX_CONTENT_LIST_SIZE] if len(items) > MAX_CONTENT_LIST_SIZE else items


def _invalid_request(message: str):
    """400 with the OpenAI-compatible validator error envelope."""
    from aiohttp import web
    return web.json_response({"error": {"message": message, "type": "invalid_request_error"}}, status=400)
