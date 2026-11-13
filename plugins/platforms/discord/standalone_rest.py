"""Discord REST pacing for the out-of-process sender (``hermes send``, the send_message tool, cron).

Without the live client there is no discord.py rate-limit bucket, so a long text sent as many
messages (a reply whose file upload Discord refused) must honour Discord's limits itself: a 429
waits ``retry_after`` and retries, and a bucket left at ``X-RateLimit-Remaining: 0`` waits
``X-RateLimit-Reset-After`` before the next message (#152012).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)
_MAX_RATE_LIMIT_RETRIES = 5
_bucket_open_at: dict[str, float] = {}  # url -> monotonic time its bucket refills


def api_url(path: str) -> str:
    """``https://discord.com/api/v10/<path>``, from discord.py's ``Route.BASE`` when installed."""
    try:
        from discord.http import Route
        base = Route.BASE
    except ImportError:
        base = "https://discord.com/api/v10"
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _seconds(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 60.0))
    except (TypeError, ValueError):
        return 1.0


async def post_paced(session: Any, url: str, read: Any, error_prefix: str, **kwargs: Any) -> tuple:
    """``POST url`` once the bucket is open; a 429 sleeps ``retry_after`` and retries. Returns
    ``read(resp, error_prefix)`` (``(data, err)``) of the final answer."""
    for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
        wait = _bucket_open_at.get(url, 0.0) - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        async with session.post(url, **kwargs) as resp:
            headers = getattr(resp, "headers", None) or {}
            if str(headers.get("X-RateLimit-Remaining", "")) == "0":
                _bucket_open_at[url] = time.monotonic() + _seconds(headers.get("X-RateLimit-Reset-After"))
            if resp.status == 429 and attempt < _MAX_RATE_LIMIT_RETRIES:
                retry_after = headers.get("Retry-After")
                try:
                    import json
                    retry_after = json.loads(await resp.read() or b"{}").get("retry_after", retry_after)
                except (ValueError, AttributeError):  # an unreadable 429 body still waits the header value
                    logger.debug("Discord 429 body unreadable; waiting Retry-After %s", retry_after)
                _bucket_open_at[url] = time.monotonic() + _seconds(retry_after)
                continue
            return await read(resp, error_prefix)
    raise AssertionError("unreachable")
