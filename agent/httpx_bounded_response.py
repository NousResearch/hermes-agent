"""Compression-safe bounded reads for streamed HTTPX responses.

HTTPX's decoded iterators apply ``Content-Encoding`` before yielding bytes. A
small gzip or Brotli chunk can therefore expand into a large allocation before
a caller gets a chance to count it. This module takes the deliberately simpler
identity-only path: callers request ``Accept-Encoding: identity``, and the
reader consumes the raw stream while rejecting servers that ignore that
negotiation.
"""

from __future__ import annotations

import httpx


class HTTPXResponseBodyTooLarge(ValueError):
    """Raised when a streamed response exceeds its byte cap."""


class HTTPXUnsupportedContentEncoding(ValueError):
    """Raised when an identity-only response uses a content coding."""


async def read_httpx_response_bytes_limited(
    response: httpx.Response,
    *,
    max_bytes: int,
) -> bytes:
    """Read a complete identity-encoded response without exceeding ``max_bytes``.

    The response must come from an HTTPX streaming request whose caller sent
    ``Accept-Encoding: identity``. A non-identity ``Content-Encoding`` fails
    closed before the body is read, so decompression cannot happen ahead of the
    cap. The client's own timeout remains responsible for stalled streams.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")

    content_encoding = response.headers.get("content-encoding", "").strip().lower()
    if content_encoding not in {"", "identity"}:
        raise HTTPXUnsupportedContentEncoding(
            f"response used unsupported Content-Encoding: {content_encoding}"
        )

    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared_bytes = int(content_length)
        except ValueError:
            pass
        else:
            if declared_bytes > max_bytes:
                raise HTTPXResponseBodyTooLarge(
                    f"response exceeds {max_bytes} bytes"
                )

    body = bytearray()
    async for chunk in response.aiter_raw():
        if not chunk:
            continue
        if len(chunk) > max_bytes - len(body):
            raise HTTPXResponseBodyTooLarge(
                f"response exceeds {max_bytes} bytes"
            )
        body.extend(chunk)
    return bytes(body)
