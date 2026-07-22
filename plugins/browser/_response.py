"""Bounded response readers for browser provider control-plane JSON."""

from __future__ import annotations

import json
from typing import Any

import requests

MAX_BROWSER_PROVIDER_JSON_BYTES = 16 * 1024 * 1024
MAX_BROWSER_PROVIDER_ERROR_BYTES = 64 * 1024


def _declared_content_length(response: requests.Response) -> int | None:
    raw = response.headers.get("Content-Length") or response.headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def read_browser_provider_response_bytes(
    response: requests.Response,
    *,
    label: str,
    max_bytes: int = MAX_BROWSER_PROVIDER_JSON_BYTES,
) -> bytes:
    """Read a streamed ``requests`` response with an explicit byte cap."""

    declared = _declared_content_length(response)
    if declared is not None and declared > max_bytes:
        response.close()
        raise RuntimeError(f"{label}: response exceeds {max_bytes} bytes")

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            response.close()
            raise RuntimeError(f"{label}: response exceeds {max_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def read_browser_provider_json(
    response: requests.Response,
    *,
    label: str,
    max_bytes: int = MAX_BROWSER_PROVIDER_JSON_BYTES,
) -> Any:
    raw = read_browser_provider_response_bytes(
        response,
        label=label,
        max_bytes=max_bytes,
    )
    return json.loads(raw.decode("utf-8"))


def read_browser_provider_text(
    response: requests.Response,
    *,
    label: str,
    max_bytes: int = MAX_BROWSER_PROVIDER_ERROR_BYTES,
) -> str:
    raw = read_browser_provider_response_bytes(
        response,
        label=label,
        max_bytes=max_bytes,
    )
    return raw.decode("utf-8", errors="replace")
