"""URL photos (``TelegramAdapter.send_image``).

A URL photo goes by URL first (Telegram fetches it, under 5 MB), then as downloaded bytes (up to
10 MB), then as the base adapter's text link. A spoiler photo goes with ``has_spoiler``; when
neither native send lands it gets the "couldn't deliver" notice instead of the text link, whose
link preview would show it unhidden.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

from gateway.platforms.base import SendResult, safe_url_for_log
from plugins.platforms.telegram.telegram_media_retry import unconfirmed_media_result as _unconfirmed

logger = logging.getLogger(__name__)


async def send_url_photo(
    adapter: Any, chat_id: str, image_url: str, caption: Optional[str], reply_to: Optional[str],
    metadata: Optional[dict[str, Any]], *, spoiler: bool, text_fallback: Callable[[], Awaitable[SendResult]],
) -> SendResult:
    """URL send (<5MB) → download+upload (≤10MB) → ``text_fallback`` (the base adapter's text link),
    or for a ``spoiler`` the withheld-link notice."""
    if not adapter._bot:
        return SendResult(success=False, error="Not connected")

    async def fallback() -> SendResult:
        if not spoiler:
            return await text_fallback()
        return await adapter._send_media_fallback_notice(
            "send_image", "image", safe_url_for_log(image_url), chat_id, f"||{caption}||" if caption else None,
            reply_to, metadata)

    def redo(_path=None):
        return adapter.send_image(chat_id, image_url, caption, reply_to, metadata, spoiler=spoiler)

    from tools.url_safety import is_safe_url
    if not is_safe_url(image_url):
        logger.warning("[%s] Blocked unsafe image URL (SSRF protection)", adapter.name)
        return await fallback()
    from plugins.platforms.telegram.adapter import _redact_telegram_error_text
    photo_caption = adapter._caption_1024(caption)
    hide = adapter._spoiler_media_kwargs(photo_caption, spoiler)
    try:
        msg = await adapter._send_media(
            adapter._bot.send_photo, chat_id, reply_to, metadata, "URL photo", photo=image_url, caption=photo_caption, **hide)
        return SendResult(success=True, message_id=str(msg.message_id))
    except Exception as e:
        if (recovered := await adapter._media_flood_recover(e, chat_id, redo, "URL photo", metadata=metadata) or _unconfirmed(e, "URL photo", as_failure=True)):
            return recovered
        logger.warning(
            "[%s] URL-based send_photo failed, trying file upload: %s", adapter.name, _redact_telegram_error_text(e), exc_info=True)
    try:
        from gateway.platforms.base import read_httpx_body_with_limit, ssrf_redirect_guard
        from tools.url_safety import create_ssrf_safe_async_client
        async with create_ssrf_safe_async_client(timeout=30.0, event_hooks={"response": [ssrf_redirect_guard]}) as client, \
                client.stream("GET", image_url) as resp:
            resp.raise_for_status()
            image_data = await read_httpx_body_with_limit(resp, media_type="image")
        msg = await adapter._send_media(
            adapter._bot.send_photo, chat_id, reply_to, metadata, "uploaded photo", photo=image_data, caption=photo_caption, **hide)
        return SendResult(success=True, message_id=str(msg.message_id))
    except Exception as e2:
        if (recovered := await adapter._media_flood_recover(e2, chat_id, redo, "uploaded photo", metadata=metadata) or _unconfirmed(e2, "uploaded photo", as_failure=True)):
            return recovered  # the byte upload follows the same flood contract as the URL send
        logger.error("[%s] File upload send_photo also failed: %s", adapter.name, e2, exc_info=True)
        return await fallback()
