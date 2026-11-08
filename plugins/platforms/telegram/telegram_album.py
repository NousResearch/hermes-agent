"""Telegram photo albums (``sendMediaGroup``, 10 per chunk) for the gateway adapter and the
standalone sender.

Telegram's own 5xx JSON answer (Bad Gateway) resends the SAME ``sendMediaGroup`` with the text
lane's budget, never photo by photo. A chunk that may already be in the chat (a timeout, a 504, a
reset after the write, an HTML 5xx page, or a 5xx JSON answer once that budget is spent:
``telegram_send_errors``) is reported ambiguous and never resent. Any other failure (a Bot API
refusal, a flood wait, a connect/pool failure whose request never left, a local error) falls back
photo by photo. A stale forum topic ("thread not found") retries once without the topic.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Any, Optional
from urllib.parse import unquote

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)

ALBUM_LIMIT = 10  # Telegram's media-group cap


def reply_lost(error: Exception) -> bool:
    """True when the chunk may already be in the chat (``telegram_send_errors.may_have_delivered``:
    a timeout, a 504, a reset after the write, a 5xx answer left after its same-request retries).
    Only a refusal, a flood wait, a connect/pool failure (nothing left) or a local error falls back
    photo by photo."""
    from plugins.platforms.telegram.telegram_send_errors import may_have_delivered
    return may_have_delivered(error)


def _stale_topic(adapter: Any, error: Exception, metadata: Optional[dict]) -> bool:
    """A forum topic Telegram no longer knows; Hermes DM-topic lanes keep their own anchor retry."""
    return (adapter._is_bad_request_error(error) and adapter._is_thread_not_found_error(error)
            and adapter._metadata_thread_id(metadata) is not None and not adapter._dm_topic_fallback(metadata))


def _without_thread(metadata: Optional[dict]) -> dict:
    return {k: v for k, v in (metadata or {}).items() if k not in ("thread_id", "message_thread_id")}


def _local_path(url: str) -> Optional[str]:
    return unquote(url[7:]) if url.startswith("file://") else None


async def _send_one(adapter: Any, chat_id: str, url: str, alt: Optional[str], metadata: Optional[dict]) -> SendResult:
    path = _local_path(url)
    try:
        if path is not None:
            return await adapter.send_image_file(chat_id=chat_id, image_path=path, caption=alt or None, metadata=metadata)
        return await adapter.send_image(chat_id=chat_id, image_url=url, caption=alt or None, metadata=metadata)
    except Exception as exc:  # a per-photo failure is reported, never fatal to the batch
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text
        logger.warning("[%s] Photo send failed: %s", adapter.name, _redact_telegram_error_text(exc), exc_info=True)
        return SendResult(success=False, error=_redact_telegram_error_text(exc))


async def _send_each(adapter: Any, chat_id: str, photos: list[tuple], metadata: Optional[dict], human_delay: float,
                     failed: list) -> bool:
    """One upload per photo (``send_image_file``'s photo -> document fallbacks); failures land in ``failed``."""
    delivered = False
    for photo_idx, (image_url, alt_text) in enumerate(photos):
        if human_delay > 0 and photo_idx > 0:
            await asyncio.sleep(human_delay)
        result = await _send_one(adapter, chat_id, image_url, alt_text, metadata)
        delivered = delivered or result.success
        if not result.success:
            failed.append((_local_path(image_url) or image_url, result.error or "send failed"))
    return delivered


def _open_chunk(adapter: Any, stack: contextlib.ExitStack, chunk: list[tuple], input_media: Any, failed: list):
    """``(media, opened files, (url, alt) sent)`` for one album chunk; unreadable files land in ``failed``."""
    from plugins.platforms.telegram.adapter import _redact_telegram_error_text as redact
    media: list[Any] = []
    opened: list[Any] = []
    sent: list[tuple] = []
    for image_url, alt_text in chunk:
        source: Any = image_url
        if (local_path := _local_path(image_url)) is not None:
            if not os.path.exists(local_path):
                logger.warning("[%s] Skipping missing image in media group: %s", adapter.name, local_path)
                failed.append((local_path, "file not found"))
                continue
            # Pre-compress large rasters so the upload stays under media_write_timeout.
            if compressed := adapter._compress_image_to_jpeg(local_path):
                stack.callback(lambda p=compressed: os.path.exists(p) and os.remove(p))
                local_path = compressed
            try:
                source = stack.enter_context(open(local_path, "rb"))  # streamed to the upload
            except OSError as exc:
                failed.append((_local_path(image_url), redact(exc)))
                continue
            opened.append(source)
        media.append(input_media(media=source, caption=adapter._caption_1024(alt_text)))
        sent.append((image_url, alt_text))
    return media, opened, sent


async def _send_chunk(adapter: Any, chat_id: str, media: list, opened: list, metadata: Optional[dict]) -> Optional[dict]:
    """``sendMediaGroup``, the same request again on Telegram's 5xx JSON answer (the text lane's
    budget), plus one retry without a stale forum topic. Returns the metadata the rest of the batch
    should use; raises the send's error."""
    from plugins.platforms.telegram import telegram_send_errors as kinds
    from plugins.platforms.telegram.adapter import _redact_telegram_error_text as redact

    def reset() -> None:
        for fh in opened:
            with contextlib.suppress(Exception):
                fh.seek(0)

    async def send(meta: Optional[dict]) -> None:
        reply_to_id, kwargs = adapter._media_send_kwargs(chat_id, None, meta)
        for attempt in range(kinds.SAME_REQUEST_ATTEMPTS):
            try:
                await adapter._send_with_dm_topic_reply_anchor_retry(
                    adapter._bot.send_media_group, {**kwargs, "media": media}, meta, reply_to_id, "media group",
                    reset_media=reset)
                return
            except Exception as err:
                if (kinds.send_error_kind(err) != kinds.SERVER_ERROR
                        or attempt >= kinds.SAME_REQUEST_ATTEMPTS - 1):
                    raise
                delay = kinds.same_request_backoff(attempt)
                logger.warning("[%s] Telegram 5xx on media group (attempt %d/%d), resending it in %.1fs: %s",
                               adapter.name, attempt + 1, kinds.SAME_REQUEST_ATTEMPTS, delay, redact(err))
                await asyncio.sleep(delay)
                reset()

    try:
        await send(metadata)
        return metadata
    except Exception as err:
        if not _stale_topic(adapter, err, metadata):
            raise
    stale = adapter._metadata_thread_id(metadata)
    logger.warning("[%s] Thread %s not found for media group, retrying without message_thread_id", adapter.name, stale)
    adapter._prune_stale_dm_topic_binding(chat_id, stale, metadata=metadata)
    metadata = _without_thread(metadata)
    reset()
    await send(metadata)
    return metadata


async def send_photo_album(
    adapter: Any, chat_id: str, photos: list[tuple], metadata: Optional[dict] = None, human_delay: float = 0.0,
    *, delivered: bool = False, per_photo_fallback: bool = True,
) -> SendResult:
    """Send ``(url, alt)`` photos as albums. ``success`` = at least one photo delivered or possibly
    delivered; ``raw_response`` carries ``failed`` (``[(photo, error)]``), ``refused`` (the photos of
    a refused chunk, when ``per_photo_fallback`` is off so the caller sends them its own way) and
    ``ambiguous`` (a chunk was sent but never confirmed) whenever any applies."""
    from plugins.platforms.telegram.adapter import _redact_telegram_error_text as redact
    failed: list[tuple[str, str]] = []
    refused: list[tuple] = []
    ambiguous = False
    try:
        from telegram import InputMediaPhoto
    except ImportError:  # SDK without media groups: every photo goes on its own
        logger.warning("[%s] InputMediaPhoto unavailable, sending photos one by one", adapter.name)
        InputMediaPhoto = None
    chunks = [photos[i:i + ALBUM_LIMIT] for i in range(0, len(photos), ALBUM_LIMIT)] if InputMediaPhoto else []
    for chunk_idx, chunk in enumerate(chunks):
        if human_delay > 0 and chunk_idx > 0:
            await asyncio.sleep(human_delay)
        with contextlib.ExitStack() as stack:
            media, opened, sent = _open_chunk(adapter, stack, chunk, InputMediaPhoto, failed)
            if not media:
                continue
            logger.info("[%s] Sending media group of %d photo(s) (chunk %d/%d)",
                        adapter.name, len(media), chunk_idx + 1, len(chunks))
            try:
                metadata = await _send_chunk(adapter, chat_id, media, opened, metadata)
                delivered = True
                continue
            except Exception as err:
                if reply_lost(err):
                    logger.warning(
                        "[%s] send_media_group reply lost (chunk %d/%d); not resending, it may have been posted: %s",
                        adapter.name, chunk_idx + 1, len(chunks), redact(err), exc_info=True)
                    ambiguous = delivered = True
                    continue
                logger.warning("[%s] send_media_group refused (chunk %d/%d), falling back to per-image: %s",
                               adapter.name, chunk_idx + 1, len(chunks), redact(err), exc_info=True)
        if per_photo_fallback:
            delivered = await _send_each(adapter, chat_id, sent, metadata, human_delay, failed) or delivered
        else:
            refused += sent
    if not chunks:  # no media-group support
        if per_photo_fallback:
            delivered = await _send_each(adapter, chat_id, photos, metadata, human_delay, failed) or delivered
        else:
            refused = list(photos)
    detail = {"failed": failed, "refused": refused, "ambiguous": ambiguous} if failed or refused or ambiguous else None
    error = None if delivered or refused else "all images failed to send" + (f": {failed[0][1]}" if failed else "")
    return SendResult(success=delivered, error=error, raw_response=detail)
