"""Size-capped inbound ``mxc://`` downloads.

mautrix's ``Client.download_media`` returns ``await response.read()`` with no size limit, so a
sender who omits or under-reports ``info.size`` gets the whole body buffered before
``MATRIX_MAX_MEDIA_BYTES`` can be checked. This mirrors that request (same URL helper, auth header
and logging) but streams the body under the cap.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterable, Mapping

_CHUNK_SIZE = 65536


class MatrixMediaTooLarge(ValueError):
    """An inbound Matrix media response exceeded ``MATRIX_MAX_MEDIA_BYTES``."""


async def read_body_capped(headers: Mapping[str, str], chunks: AsyncIterable[bytes], limit: int) -> bytes:
    """Reject on an oversized Content-Length, then stop reading as soon as *limit* is crossed."""
    try:
        declared = int(headers.get("Content-Length") or headers.get("content-length") or 0)
    except (TypeError, ValueError):
        declared = 0
    if declared > limit:
        raise MatrixMediaTooLarge(f"media exceeds Matrix limit ({declared} > {limit} bytes)")
    parts: list[bytes] = []
    total = 0
    async for chunk in chunks:
        total += len(chunk)
        if total > limit:
            raise MatrixMediaTooLarge(f"media exceeds Matrix limit (> {limit} bytes)")
        parts.append(bytes(chunk))
    return b"".join(parts)


async def download_mxc_capped(client: Any, mxc_url: str, limit: int) -> bytes:
    """``client.download_media(mxc_url)`` with the body streamed under *limit* bytes."""
    api = client.api
    authenticated = bool((await client.versions()).supports("v1.11"))
    url = api.get_download_url(mxc_url, authenticated=authenticated)
    query_params: dict[str, Any] = {"allow_redirect": "true"}
    headers: dict[str, str] = {}
    if authenticated:
        headers["Authorization"] = f"Bearer {api.token}"
        if api.as_user_id:
            query_params["user_id"] = api.as_user_id
    req_id = api.log_download_request(url, query_params)
    start = time.monotonic()
    async with api.session.get(url, params=query_params, headers=headers) as response:
        try:
            response.raise_for_status()
            return await read_body_capped(response.headers, response.content.iter_chunked(_CHUNK_SIZE), limit)
        finally:
            api.log_download_request_done(url, req_id, time.monotonic() - start, response.status)
