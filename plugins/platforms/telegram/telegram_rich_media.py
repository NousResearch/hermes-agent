"""Attachments carried inside ``rich_message`` blocks (Bot API 10.1).

A message can arrive with its photo, video or file only in a ``rich_message`` block (``photo``,
``video``, ``document``... at any depth, collages and slideshows included) and no top-level
``photo``/``document`` field. PTB then sees a text-less, attachment-less message, so the media
handler never runs and the agent got the caption alone (#155902). These helpers find those
attachments and download them like a top-level one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Iterator

from plugins.platforms.telegram.telegram_entities import _MAX_RICH_DEPTH

logger = logging.getLogger("plugins.platforms.telegram.adapter")

# block type -> (field holding the file, default cache kind, default MIME); the field is the block type
_MEDIA_BLOCKS = {
    "photo": ("image", ""),
    "animation": ("video", "video/mp4"),
    "video": ("video", "video/mp4"),
    "audio": ("audio", ""),
    "voice_note": ("audio", "audio/ogg"),
    "document": (None, ""),
}
# One message, not an album: a hostile payload cannot make us download hundreds of files.
_MAX_RICH_ATTACHMENTS = 10


def _raw_rich(message: Any) -> Any:
    rich = getattr(message, "rich_message", None)
    if rich is None and isinstance(kwargs := getattr(message, "api_kwargs", None), Mapping):
        rich = kwargs.get("rich_message")
    return rich


def _walk(blocks: Any, depth: int = 0) -> Iterator[Mapping]:
    if depth > _MAX_RICH_DEPTH or not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        yield block
        yield from _walk(block.get("blocks"), depth + 1)
        for item in block.get("items") or []:
            if isinstance(item, Mapping):
                yield from _walk(item.get("blocks"), depth + 1)


def rich_media_items(message: Any) -> list[tuple[str, str, Mapping, str, str | None]]:
    """``(block_type, file_id, media, mime, default_kind)`` per attachment in ``message``'s rich blocks, in order."""
    rich = _raw_rich(message)
    if not isinstance(rich, Mapping):
        return []
    found = []
    for block in _walk(rich.get("blocks")):
        block_type = str(block.get("type") or "")
        if block_type not in _MEDIA_BLOCKS:
            continue
        kind, mime = _MEDIA_BLOCKS[block_type]
        media = block.get(block_type)
        if block_type == "photo":  # PhotoSize list: largest last
            media = media[-1] if isinstance(media, list) and media else None
        if not isinstance(media, Mapping):
            continue
        file_id = media.get("file_id")
        if isinstance(file_id, str) and file_id:
            found.append((block_type, file_id, media, str(media.get("mime_type") or mime).lower(), kind))
        if len(found) >= _MAX_RICH_ATTACHMENTS:
            break
    return found


def _message_type(block_type: str, cached_kind: str) -> Any:
    """The type a top-level attachment of this kind gets: an audio FILE is AUDIO (no auto-transcription),
    only a voice note is VOICE; an image document is a PHOTO."""
    from gateway.platforms.event import MessageType
    if cached_kind == "image":
        return MessageType.PHOTO
    if block_type == "voice_note":
        return MessageType.VOICE
    return {"audio": MessageType.AUDIO, "video": MessageType.VIDEO}.get(cached_kind, MessageType.DOCUMENT)


def rich_media_source(bot: Any, file_id: str, media: Mapping) -> SimpleNamespace:
    """The ``get_file()``/``file_size`` surface ``fetch_telegram_file`` reads, for a rich block's bare file id."""
    return SimpleNamespace(file_size=media.get("file_size"), get_file=lambda: bot.get_file(file_id))


async def attach_rich_media(adapter: Any, msg: Any, event: Any) -> None:
    """Download every rich-block attachment of ``msg`` onto ``event``: images ride inline like a
    sent photo, other files get the saved-path note the document path gives them. A file that
    cannot be fetched leaves a note, so the agent never answers from the caption alone."""
    from gateway.platforms.base import cache_media_bytes_async
    from plugins.platforms.telegram.telegram_batching import pad_inline_flags
    from plugins.platforms.telegram.telegram_media_fetch import fetch_telegram_file, local_root

    max_bytes = getattr(adapter, "_max_doc_bytes", 20 * 1024 * 1024)
    bot = getattr(adapter, "_bot", None)
    for block_type, file_id, media, mime, kind in rich_media_items(msg) if bot is not None else []:
        if adapter._int_or_zero(media.get("file_size")) > max_bytes:
            event.text = adapter._append_observed_note(
                event.text, f"[The user sent an attachment larger than {max_bytes // (1024 * 1024)} MB; it was not downloaded.]")
            continue
        source = rich_media_source(bot, file_id, media)
        filename = str(media.get("file_name") or "")
        try:
            file_obj, data = await fetch_telegram_file(source, "rich message attachment", local_root(adapter))
            cached = await cache_media_bytes_async(
                data, filename=filename or (getattr(file_obj, "file_path", "") or "").rsplit("/", 1)[-1],
                mime_type=mime, default_kind=kind)
        except Exception as exc:  # health: allow BLE001 -- any fetch failure becomes the agent-visible note below
            logger.warning("[Telegram] Failed to cache rich message attachment: %s", exc.__class__.__name__)
            cached, exc_name = None, exc.__class__.__name__
        else:
            exc_name = "unreadable"
        if cached is None:
            event.text = adapter._append_observed_note(
                event.text, f"[The user sent an attachment that could not be downloaded ({exc_name}).]")
            continue
        event.media_text_inlined = pad_inline_flags(event) + [False]
        event.media_urls.append(cached.path)
        event.media_types.append(cached.media_type)
        if len(event.media_urls) == 1:  # the first attachment sets the type, as on the top-level path
            event.message_type = _message_type(block_type, cached.kind)
        if cached.kind != "image":
            event.text = adapter._append_observed_note(event.text, cached.context_note())
        logger.info("[Telegram] Cached rich message %s at %s", cached.kind, cached.path)


async def dispatch_rich_media(adapter: Any, msg: Any, event: Any) -> None:
    """Deliver an addressed rich message carrying attachments the way the media handler delivers a
    top-level one: claim the question typed just before it BEFORE the download awaits (else that
    question flushes alone), then an image joins the photo batcher and anything else goes out
    with the claimed text absorbed."""
    from gateway.platforms.event import MessageType
    from plugins.platforms.telegram.telegram_text_photo_pairing import (
        absorb_claimed_text, claim_pending_text, release_text_claim)
    claim_pending_text(adapter, msg, event)
    try:
        await attach_rich_media(adapter, msg, event)
        if event.message_type == MessageType.PHOTO:
            await adapter._route_photo_event(msg, event)
            return
        absorb_claimed_text(adapter, msg, event)
        if event.message_type == MessageType.TEXT:  # every download failed: the notes ride as text
            adapter._enqueue_text_event(event)
        else:
            await adapter.handle_message(event)
    finally:
        release_text_claim(adapter, msg)


async def observe_skipped(adapter: Any, msg: Any, update: Any) -> None:
    """Record an unaddressed group message as observed context (when configured), with its first
    rich attachment cached like an observed top-level one, so a later question can open it."""
    from gateway.platforms.event import MessageType
    if not adapter._should_observe_unmentioned_group_message(msg):
        return
    event = adapter._build_message_event(msg, MessageType.TEXT, update_id=update.update_id)
    await adapter._cache_observed_media(msg, event)
    adapter._observe_unmentioned_group_message(msg, MessageType.TEXT, update_id=update.update_id, event=event)


async def handle_rich_media_message(adapter: Any, msg: Any, update: Any) -> bool:
    """``_handle_text_message`` for a rich message carrying attachments (True = handled; False = no
    attachments, the text path goes on): gate it like any message; unaddressed, observe it with its
    cached attachment; addressed, dispatch it as media."""
    from gateway.platforms.event import MessageType
    if not rich_media_items(msg):
        return False
    if not adapter._should_process_message(msg):
        await observe_skipped(adapter, msg, update)
        return True
    await adapter._ensure_forum_commands(update.message)
    await dispatch_rich_media(adapter, msg, await adapter._build_triggered_event(msg, update, MessageType.TEXT))
    return True


def rich_observed_source(adapter: Any, msg: Any) -> tuple[Any, str, str, str | None]:
    """``_observed_media_source`` for the first rich-block attachment (replied-to or observed), else Nones."""
    bot = getattr(adapter, "_bot", None)
    if bot is None or not (rich := rich_media_items(msg)):
        return None, "", "", None
    _block, file_id, media, mime, kind = rich[0]
    return rich_media_source(bot, file_id, media), str(media.get("file_name") or ""), mime, kind
