"""Bounded JSON response helpers for bundled web providers."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

DEFAULT_WEB_PROVIDER_JSON_MAX_BYTES = 2 * 1024 * 1024


class WebProviderResponseTooLarge(RuntimeError):
    """Raised when an upstream web-provider JSON response exceeds the cap."""


def _response_byte_limit() -> int:
    raw = os.getenv("HERMES_WEB_PROVIDER_JSON_MAX_BYTES", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = DEFAULT_WEB_PROVIDER_JSON_MAX_BYTES
        if value > 0:
            return value
    return DEFAULT_WEB_PROVIDER_JSON_MAX_BYTES


def _read_limited_response_bytes(response: httpx.Response, *, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise WebProviderResponseTooLarge(
                f"web provider JSON response exceeded {max_bytes} bytes"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def httpx_json_request(method: str, url: str, **kwargs: Any) -> Any:
    """Send an httpx request and parse JSON while enforcing a response cap."""

    max_bytes = int(kwargs.pop("max_bytes", _response_byte_limit()))
    with httpx.stream(method, url, **kwargs) as response:
        response.raise_for_status()
        body = _read_limited_response_bytes(response, max_bytes=max_bytes)
    if not body:
        return {}
    encoding = response.encoding or "utf-8"
    return json.loads(body.decode(encoding, errors="replace"))
