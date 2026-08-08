"""Small helpers for bounded urllib response reads."""

from __future__ import annotations

import json
from typing import Any

JSON_RESPONSE_BODY_MAX_BYTES = 1024 * 1024
ERROR_RESPONSE_BODY_MAX_BYTES = 64 * 1024


class ResponseBodyTooLarge(ValueError):
    """Raised when a response body exceeds its configured read cap."""


def read_limited_response_body(resp: Any, limit: int, *, label: str) -> bytes:
    body = resp.read(limit + 1)
    if len(body) > limit:
        raise ResponseBodyTooLarge(f"{label} exceeded {limit} bytes")
    return body


def read_limited_json_response(
    resp: Any,
    *,
    limit: int | None = None,
    label: str = "JSON response body",
) -> Any:
    if limit is None:
        limit = JSON_RESPONSE_BODY_MAX_BYTES
    body = read_limited_response_body(resp, limit, label=label)
    return json.loads(body.decode("utf-8"))

