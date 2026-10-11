"""Normalize successful nested completions without replacing shared HTTP transports."""

import json
from urllib.parse import urlparse


_ENVELOPE_HOSTS = frozenset({"api.cline.bot"})


def _is_candidate(response) -> bool:
    media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    return (response.is_success and response.request.url.host in _ENVELOPE_HOSTS
            and media_type == "application/json")


def _unwrap(response) -> None:
    try:
        body = response.json()
    except ValueError:
        return
    if (not isinstance(body, dict) or "choices" in body or body.get("error")
            or body.get("success") is False):
        return
    data = body.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
        return
    content = json.dumps(data).encode("utf-8")
    # HTTPX response hooks cannot replace the Response. The body is fully read
    # before replacing its decoded cache, preserving request/stream ownership.
    response._content = content
    for header in ("content-encoding", "transfer-encoding"):
        response.headers.pop(header, None)
    response.headers["content-length"] = str(len(content))


def _normalize_sync(response) -> None:
    if _is_candidate(response):
        response.read()
        _unwrap(response)


async def _normalize_async(response) -> None:
    if _is_candidate(response):
        await response.aread()
        _unwrap(response)


def envelope_response_hooks(base_url: str, *, async_mode: bool) -> list:
    if urlparse(base_url).hostname not in _ENVELOPE_HOSTS:
        return []
    return [_normalize_async if async_mode else _normalize_sync]
