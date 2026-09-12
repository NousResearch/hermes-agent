"""Discord message state, formatting, typing, and attachment caching."""

from __future__ import annotations

import asyncio
import os
import re
from contextlib import suppress
from typing import Any, Dict, Optional

from gateway.platforms.event import MessageType
from gateway.platforms.base import SUPPORTED_DOCUMENT_TYPES, _TEXT_INJECT_EXTENSIONS
from gateway.platforms.helpers import convert_table_to_bullets
from .. import adapter as _adapter

logger = _adapter.logger
discord = _adapter.discord


def _image_ext_from_content_type(value: str) -> str:
    return _adapter._image_ext_from_content_type(value)


def _extract_discord_retry_after(value):
    return _adapter._extract_discord_retry_after(value)


def cache_image_from_url(*args, **kwargs):
    return _adapter.cache_image_from_url(*args, **kwargs)


def cache_image_from_bytes_async(*args, **kwargs):
    return _adapter.cache_image_from_bytes_async(*args, **kwargs)


def cache_audio_from_url(*args, **kwargs):
    return _adapter.cache_audio_from_url(*args, **kwargs)


def cache_audio_from_bytes_async(*args, **kwargs):
    return _adapter.cache_audio_from_bytes_async(*args, **kwargs)


def cache_document_from_bytes_async(*args, **kwargs):
    return _adapter.cache_document_from_bytes_async(*args, **kwargs)


def validate_inbound_media_size(*args, **kwargs):
    return _adapter.validate_inbound_media_size(*args, **kwargs)


def is_safe_url(value):
    return _adapter.is_safe_url(value)


def _Snowflake(value: int):
    return _adapter._Snowflake(value)


class MessageStateMixin:
    """Own transient Discord message presentation and media state operations."""

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch; longer delay when the chunk is
        near Discord's 2000-char split point (continuation almost certain)."""
        current_task = asyncio.current_task()
        try:
            pending = self._pending_text_batches.get(key)
            last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
            if last_len >= self._SPLIT_THRESHOLD:
                delay = self._text_batch_split_delay_seconds
            else:
                delay = self._text_batch_delay_seconds
            await asyncio.sleep(delay)
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            logger.info("[Discord] Flushing text batch %s (%d chars)", key, len(event.text or ""))
            # Shield the dispatch: _enqueue_text_event cancels the prior flush task on each new chunk;
            # without the shield CancelledError would abort the in-flight agent turn.
            await asyncio.shield(self.handle_message(event))
        except asyncio.CancelledError:
            # Cancel landed before the pop; shielded handle_message unaffected.
            pass
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)


# ---------------------------------------------------------------------------
# Discord UI Components (outside the adapter class)
# ---------------------------------------------------------------------------

    async def _read_attachment_bytes(self, att, *, media_type: str = "media") -> Optional[bytes]:
        """Read an attachment via the authenticated bot session; ``None`` (no callable ``read()``
        or read failure) means fall back to the URL downloaders. Raises ``ValueError`` for oversized
        attachments BEFORE pulling bytes when Discord reports the size, so a hostile upload can't OOM."""
        attachment_size = getattr(att, "size", None)
        if attachment_size:
            validate_inbound_media_size(int(attachment_size), media_type=media_type)
        reader = getattr(att, "read", None)
        if reader is None or not callable(reader):
            return None
        try:
            raw_bytes = await reader()
        except Exception as e:
            logger.warning(
                "[Discord] Authenticated attachment read failed for %s: %s",
                getattr(att, "filename", None) or getattr(att, "url", "<unknown>"), e,
            )
            return None
        validate_inbound_media_size(len(raw_bytes), media_type=media_type)
        return raw_bytes

    async def _cache_discord_image(self, att, ext: str) -> str:
        """Cache an image attachment locally: ``att.read()`` first, SSRF-gated URL fallback."""
        raw_bytes = await self._read_attachment_bytes(att, media_type="image")
        if raw_bytes is not None:
            try:
                return await cache_image_from_bytes_async(raw_bytes, ext=ext)
            except Exception as e:
                logger.debug(
                    "[Discord] cache_image_from_bytes rejected att.read() data; falling back to URL: %s",
                    e,
                )
        return await cache_image_from_url(att.url, ext=ext)

    async def _cache_discord_audio(self, att, ext: str) -> str:
        """Cache an audio attachment locally: ``att.read()`` first, SSRF-gated URL fallback."""
        raw_bytes = await self._read_attachment_bytes(att, media_type="audio")
        if raw_bytes is not None:
            try:
                return await cache_audio_from_bytes_async(raw_bytes, ext=ext)
            except Exception as e:
                logger.debug("[Discord] cache_audio_from_bytes failed; falling back to URL: %s", e)
        return await cache_audio_from_url(att.url, ext=ext)

    async def _cache_discord_document(self, att, ext: str) -> bytes:
        """Download a document attachment: ``att.read()`` first, SSRF-gated aiohttp fallback.
        Caller passes the bytes to ``cache_document_from_bytes`` (and injects text if applicable).

        This closes the gap where the old document path made raw ``aiohttp.ClientSession`` requests with no
        safety check (#11345). The caller is responsible for passing the returned bytes to
        ``cache_document_from_bytes`` (and, where applicable, for injecting text content).
        """
        raw_bytes = await self._read_attachment_bytes(att, media_type="document")
        if raw_bytes is not None:
            return raw_bytes
        if not is_safe_url(att.url):
            raise ValueError(f"Blocked unsafe attachment URL (SSRF protection): {att.url}")
        import aiohttp
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
        _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
        async with aiohttp.ClientSession(**_sess_kw) as session:
            async with session.get(
                att.url, timeout=aiohttp.ClientTimeout(total=30), **_req_kw,
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}")
                return await resp.read()

    async def _cache_simple_media(self, att: Any, content_type: str, kind: str, exts: set, default_ext: str) -> str:
        """Cache an image/audio attachment locally (CDN URLs expire); fall back to the CDN URL."""
        try:
            ext = "." + content_type.split("/")[-1].split(";")[0]
            if ext not in exts:
                ext = default_ext
            cacher = self._cache_discord_image if kind == "image" else self._cache_discord_audio
            cached_path = await cacher(att, ext)
            print(f"[Discord] Cached user {kind}: {cached_path}", flush=True)
            return cached_path
        except Exception as e:
            print(f"[Discord] Failed to cache {kind} attachment: {e}", flush=True)
            return att.url

    async def _collect_attachment_media(self, all_attachments: list) -> tuple:
        """Cache every attachment and return ``(media_urls, media_types, pending_text_injection)``."""
        media_urls = []
        media_types = []
        pending_text_injection: Optional[str] = None
        for att in all_attachments:
            content_type = att.content_type or "unknown"
            if content_type.startswith("image/"):
                media_urls.append(await self._cache_simple_media(
                    att, content_type, "image", {".jpg", ".jpeg", ".png", ".gif", ".webp"}, ".jpg"))
                media_types.append(content_type)
            elif content_type.startswith("audio/"):
                media_urls.append(await self._cache_simple_media(
                    att, content_type, "audio", {".ogg", ".mp3", ".wav", ".webm", ".m4a"}, ".ogg"))
                media_types.append(content_type)
            else:
                ext = ""
                if att.filename:
                    _, ext = os.path.splitext(att.filename)
                    ext = ext.lower()
                if not ext and content_type:
                    mime_to_ext = {v: k for k, v in SUPPORTED_DOCUMENT_TYPES.items()}
                    ext = mime_to_ext.get(content_type, "")
                in_allowlist = ext in SUPPORTED_DOCUMENT_TYPES
                # Any file type accepted (authorization is the gate); unknown types fall back to octet-stream.
                max_doc_bytes = self._discord_max_attachment_bytes()
                if max_doc_bytes and att.size and att.size > max_doc_bytes:
                    logger.warning(
                        "[Discord] Document too large (%s bytes > cap %s), skipping: %s",
                        att.size, max_doc_bytes, att.filename,
                    )
                    continue
                try:
                    raw_bytes = await self._cache_discord_document(att, ext)
                    cached_path = await cache_document_from_bytes_async(raw_bytes, att.filename or f"document{ext or '.bin'}")
                    if in_allowlist:
                        doc_mime = SUPPORTED_DOCUMENT_TYPES[ext]
                    else:
                        # Untyped: source content_type, else octet-stream (agent knows it's binary).
                        doc_mime = (
                            content_type if content_type and content_type != "unknown" else "application/octet-stream"
                        )
                    media_urls.append(cached_path)
                    media_types.append(doc_mime)
                    logger.info(
                        "[Discord] Cached user %s: %s", "document" if in_allowlist else "attachment", cached_path,
                    )
                    # Inject text for text-readable documents (capped at 100 KB). Gate on text-like
                    # extension/MIME, NOT a blind UTF-8 decode (PDF/zip/docx have ASCII headers); other
                    # types rely on ``gateway/run.py`` emitting a (sandbox-translated) path note.
                    MAX_TEXT_INJECT_BYTES = 100 * 1024
                    _is_text = ext in _TEXT_INJECT_EXTENSIONS or (content_type or "").startswith("text/")
                    if _is_text and len(raw_bytes) <= MAX_TEXT_INJECT_BYTES:
                        try:
                            text_content = raw_bytes.decode("utf-8")
                            display_name = att.filename or f"document{ext or '.txt'}"
                            display_name = re.sub(r'[^\w.\- ]', '_', display_name)
                            injection = f"[Content of {display_name}]:\n{text_content}"
                            if pending_text_injection:
                                pending_text_injection = f"{pending_text_injection}\n\n{injection}"
                            else:
                                pending_text_injection = injection
                        except UnicodeDecodeError:
                            pass
                except Exception as e:
                    logger.warning("[Discord] Failed to cache document %s: %s", att.filename, e, exc_info=True)
        return media_urls, media_types, pending_text_injection

    def _attachment_message_type(self, att: Any) -> MessageType:
        """MessageType from the first attachment's MIME. Any non-media (or untyped) attachment
        is a DOCUMENT regardless of extension — authorization is the gate, not the file type."""
        content_type = att.content_type or ""
        if content_type.startswith("image/"):
            return MessageType.PHOTO
        if content_type.startswith("video/"):
            return MessageType.VIDEO
        if content_type.startswith("audio/"):
            return MessageType.VOICE if self._is_discord_voice_message_attachment(att) else MessageType.AUDIO
        return MessageType.DOCUMENT

    @staticmethod
    def _reply_target(reference: Any) -> Optional[Any]:
        """Something with ``.id`` for the replied-to message; duck-typed (test doubles mock ``discord``),
        falling back to a bare snowflake from ``reference.message_id``."""
        _resolved = getattr(reference, "resolved", None)
        if getattr(_resolved, "id", None) is not None:
            return _resolved
        _ref_mid = getattr(reference, "message_id", None)
        if _ref_mid is not None:
            with suppress(ValueError, TypeError):
                return _Snowflake(int(_ref_mid))
        return None

    def format_message(self, content: str) -> str:
        """Format for Discord: GFM tables become bullet lists (Discord doesn't render pipe tables)."""
        if not content:
            return content
        return convert_table_to_bullets(content)
    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Start a persistent typing loop (POST typing every 12s; indicator lasts ~10s).
        TYPING_START is unreliable for bots in DMs; 429 sleeps ``retry_after``; CancelledError ends it."""
        if not self._client:
            return
        if chat_id in self._typing_tasks:
            return

        async def _typing_loop() -> None:
            try:
                while True:
                    try:
                        route = discord.http.Route(
                            "POST", "/channels/{channel_id}/typing", channel_id=chat_id,
                        )
                        await self._client.http.request(route)
                    except asyncio.CancelledError:
                        return
                    except Exception as e:
                        retry_after = self._extract_discord_retry_after(e)
                        if retry_after is not None:
                            logger.warning(
                                "Typing indicator rate-limited for %s; retrying in %.1fs",
                                chat_id, retry_after,
                            )
                        else:
                            logger.debug("Discord typing indicator failed for %s: %s", chat_id, e)
                            return
                        await asyncio.sleep(retry_after)
                        continue
                    await asyncio.sleep(12)
            except asyncio.CancelledError:
                pass
            finally:
                self._typing_tasks.pop(chat_id, None)
        self._typing_tasks[chat_id] = asyncio.create_task(_typing_loop())

    async def stop_typing(self, chat_id: str) -> None:
        """Stop the persistent typing indicator for a channel."""
        task = self._typing_tasks.pop(chat_id, None)
        if task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
