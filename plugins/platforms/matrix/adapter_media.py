"""Inbound Matrix media admission, download and caching."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Optional

from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.matrix.media_content import _media_wire_body
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache
from plugins.platforms.matrix.voice_mention import ParkedVoices, has_voice_marker

logger = logging.getLogger("plugins.platforms.matrix.adapter")

_EXTENSION_RE = re.compile(r"\.[0-9A-Za-z]{1,10}")
# Bidirectional embedding, override and isolate controls. An override can make a name such as
# "report<U+202E>fdp.exe" display as "reportexe.pdf".
_BIDI_CONTROLS = frozenset("‪‫‬‭‮⁦⁧⁨⁩")


def _without_controls(text: str) -> str:
    return "".join(
        ch for ch in text if ch not in _BIDI_CONTROLS and unicodedata.category(ch) != "Cc").strip()


def _declared_filename(declared: object) -> str:
    if not isinstance(declared, str):
        return ""
    name = _without_controls(declared.replace("\\", "/").rsplit("/", 1)[-1])
    return "" if name in {".", ".."} else name


def _body_filename(body: str) -> str:
    text = body.strip()
    if len(text.splitlines()) != 1 or "/" in text or "\\" in text:
        return ""
    name = _without_controls(text)
    dot = name.rfind(".")
    if dot <= 0 or not _EXTENSION_RE.fullmatch(name[dot:]):
        return ""
    return name


def inbound_media_filename(declared: object, body: str) -> str:
    """Return the attachment's filename, or "" when the event does not give one.

    ``declared`` is the event's ``filename`` and ``body`` is its ``body`` without any reply
    fallback. The Matrix spec makes ``body`` the filename when ``filename`` is absent, but some
    clients put a caption there instead. A body therefore counts as a filename only when it is
    one line, contains no path separator (which also excludes a URL), and ends in an extension of
    up to ten letters or digits. Control characters are removed from either name, and the result
    never contains a path separator.
    """
    return _declared_filename(declared) or _body_filename(body)


def _inbound_media_marker(msgtype: str, source_content: dict, relates_to: dict, problem: str) -> str:
    """Describe an attachment that could not be cached, with its transport filename if available."""
    kind = {
        "m.image": "image", "m.audio": "audio", "m.video": "video", "m.sticker": "sticker",
    }.get(msgtype, "file")
    filename = inbound_media_filename(source_content.get("filename"), _media_wire_body(source_content, relates_to))
    return f"[matrix {kind} attachment {problem}: {filename}]" if filename else f"[matrix {kind} attachment {problem}]"


class _InboundMediaTooLarge(Exception):
    """An inbound attachment is larger than the Matrix adapter accepts."""



class MatrixMediaMixin(BasePlatformAdapter):
    _client: Any
    _max_media_bytes: int
    _event_context_cache: MatrixEventContextCache
    _parked_voices: ParkedVoices
    _voice_may_park: Callable[..., bool]
    _resolve_message_context: Callable[..., Awaitable[tuple | None]]
    _build_inbound_event: Callable[..., Awaitable[MessageEvent | None]]
    _admit: Callable[[MessageEvent], Awaitable[bool]]

    async def _handle_media_message(
        self, room_id: str, sender: str, event_id: str, event_ts: float, source_content: dict,
        relates_to: dict, msgtype: str, mention_claimed: bool = False, *,
        reply_parent: MatrixEventContext | None = None, receipt_event_id: str | None = None) -> bool | None:
        body = source_content.get("body", "") or ""
        transport_filename = inbound_media_filename(
            source_content.get("filename"), _media_wire_body(source_content, relates_to))
        url = source_content.get("url", "")
        if url and not str(url).startswith("mxc://"):
            logger.warning("[Matrix] Rejecting inbound media %s with non-MXC URL", event_id)
            return
        content_info = source_content.get("info", {})
        if not isinstance(content_info, dict):
            content_info = {}
        event_mimetype = content_info.get("mimetype", "")
        try:
            event_size_int = int(content_info.get("size") or 0)
        except (TypeError, ValueError):
            event_size_int = 0
        media_limit = self._inbound_media_limit()
        media_size_limit_exceeded = event_size_int > media_limit
        if media_size_limit_exceeded:
            logger.warning(
                "[Matrix] Rejecting oversized inbound media %s (%d > %d bytes)", event_id, event_size_int,
                media_limit)
        file_content = source_content.get("file", {})  # encrypted media carries file.url
        if not url and isinstance(file_content, dict):
            url = file_content.get("url", "") or ""
            if url and not str(url).startswith("mxc://"):
                logger.warning("[Matrix] Rejecting inbound encrypted media %s with non-MXC URL", event_id)
                return
        is_encrypted_media = bool(file_content and isinstance(file_content, dict) and file_content.get("url"))
        msg_type, media_type, _is_voice_message = self._classify_inbound_media(msgtype, event_mimetype, source_content)
        reply_target = MatrixRelation.from_content(relates_to).reply_target
        reply_parent = reply_parent or (self._event_context_cache.retain(room_id, reply_target) if reply_target else None)
        # Gate (require_mention / allowed rooms) BEFORE the download: an unmentioned or
        # non-allowlisted room must not pull media onto the host only to drop it.
        # First await: mark a voice that may park in-flight so a concurrent bare mention waits for it.
        gate = self._parked_voices.begin(room_id, sender, event_ts) if self._voice_may_park(
            room_id, body, source_content, relates_to, mention_claimed) else None
        try:
            ctx = await self._resolve_message_context(
                room_id, sender, event_id, body, source_content, relates_to, mention_claimed=mention_claimed,
                voice_gate=gate)
        finally:
            if gate is not None:
                self._parked_voices.release(room_id, sender, gate)
        if ctx is None:
            return
        # Cache locally so downstream tools get a real file path.
        cached_path = None
        if url and not media_size_limit_exceeded:
            try:
                cached_path = await self._download_and_cache_media(
                    url, event_id, file_content if is_encrypted_media else None, msg_type, media_type,
                    transport_filename, media_limit)
            except _InboundMediaTooLarge:
                logger.warning(
                    "[Matrix] Rejecting oversized inbound media %s (download > %d bytes)", event_id, media_limit)
                media_size_limit_exceeded = True
            except Exception as e:
                logger.warning("[Matrix] Failed to cache media: %s", e)
        if media_size_limit_exceeded or not cached_path:
            problem = "too large" if media_size_limit_exceeded else "could not be downloaded"
            msg_event = await self._build_inbound_event(
                room_id, sender, event_id, body, source_content, relates_to, ctx=ctx,
                reply_parent=reply_parent, event_ts=event_ts,
                message_type=MessageType.TEXT, media_urls=[], media_types=[], media_msgtype=msgtype,
                metadata={"matrix_mention_claimed": True} if mention_claimed else {})
            if msg_event is not None:
                marker = _inbound_media_marker(msgtype, source_content, relates_to, problem)
                msg_event.text = f"{msg_event.text}\n{marker}".strip()
                return await self._admit(msg_event)
            return
        media_urls = [cached_path] if cached_path else []
        msg_event = await self._build_inbound_event(
            room_id, sender, event_id, body, source_content, relates_to, ctx=ctx, message_type=msg_type,
            reply_parent=reply_parent, event_ts=event_ts,
            media_urls=media_urls, media_types=[media_type] if media_urls else [], media_msgtype=msgtype,
            metadata={"matrix_mention_claimed": True} if mention_claimed else {})
        if msg_event is not None:
            if receipt_event_id:
                msg_event._processing_state.receipt_message_id = receipt_event_id
            return await self._admit(msg_event)


    @staticmethod
    def _classify_inbound_media(
            msgtype: str, event_mimetype: str, source_content: dict) -> tuple[MessageType, str, bool]:
        """Map a Matrix media msgtype to (MessageType, mime type, is_voice_message)."""
        if msgtype in {"m.image", "m.sticker"}:
            return MessageType.PHOTO, event_mimetype or "image/png", False
        if msgtype == "m.audio":
            is_voice = has_voice_marker(source_content)
            return (MessageType.VOICE if is_voice else MessageType.AUDIO), event_mimetype or "audio/ogg", is_voice
        if msgtype == "m.video":
            return MessageType.VIDEO, event_mimetype or "video/mp4", False
        return MessageType.DOCUMENT, event_mimetype or "application/octet-stream", False


    def _inbound_media_limit(self) -> int:
        """The inbound size limit: ``MATRIX_MAX_MEDIA_BYTES``, or ``gateway.max_inbound_media_bytes`` when
        that cap is smaller. The media cache applies the gateway cap only after the whole download."""
        from gateway.platforms.base_media_limits import get_inbound_media_max_bytes
        gateway_limit = get_inbound_media_max_bytes()
        if gateway_limit <= 0:
            return self._max_media_bytes
        return min(self._max_media_bytes, gateway_limit)


    async def _download_media_within(self, url: str, limit: int) -> bytes:
        """Download *url* as mautrix's ``download_media`` does, but raise ``_InboundMediaTooLarge``
        once the body exceeds *limit*. ``download_media`` reads the whole body before returning."""
        from plugins.platforms.matrix.adapter import SpecVersions

        api = self._client.api
        authenticated = (await self._client.versions()).supports(SpecVersions.V111)
        headers = {"Authorization": f"Bearer {api.token}"} if authenticated else {}
        async with api.session.get(
                api.get_download_url(url, authenticated=authenticated), params={"allow_redirect": "true"},
                headers=headers) as response:
            response.raise_for_status()
            if response.content_length is not None and response.content_length > limit:
                raise _InboundMediaTooLarge(f"{response.content_length} > {limit} bytes")
            parts: list[bytes] = []
            total = 0
            async for chunk in response.content.iter_chunked(65536):
                total += len(chunk)
                if total > limit:
                    raise _InboundMediaTooLarge(f"> {limit} bytes")
                parts.append(chunk)
        return b"".join(parts)


    async def _download_and_cache_media(
        self, url: str, event_id: str, encrypted_file: Optional[dict], msg_type: MessageType, media_type: str,
        transport_filename: str, limit: int) -> Optional[str]:
        """Download (and decrypt, when *encrypted_file* is given) media into the local cache."""
        file_bytes = await self._download_media_within(url, limit)
        if encrypted_file is not None:
            hashes_value, key_value = encrypted_file.get("hashes"), encrypted_file.get("key")
            hash_value = hashes_value.get("sha256") if isinstance(hashes_value, dict) else None
            key_value = key_value.get("k") if isinstance(key_value, dict) else key_value
            iv_value = encrypted_file.get("iv")
            if not (key_value and hash_value and iv_value):
                logger.warning("[Matrix] Encrypted media event missing decryption metadata for %s", event_id)
                return None
            from mautrix.crypto.attachments import decrypt_attachment

            file_bytes = decrypt_attachment(file_bytes, key_value, hash_value, iv_value)
        from gateway.platforms.base import (
            cache_audio_from_bytes_async,
            cache_document_from_bytes_async,
            cache_image_from_bytes_async,
        )
        from gateway.platforms.media_cache import ext_for_mime
        if msg_type == MessageType.PHOTO:
            ext_map = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}
            cached_path = await cache_image_from_bytes_async(file_bytes, ext=ext_map.get(media_type, ".jpg"))
            logger.info("[Matrix] Cached user image at %s", cached_path)
            return cached_path
        mimetype_ext = ext_for_mime(media_type) or ""
        if msg_type in {MessageType.AUDIO, MessageType.VOICE}:
            if not file_bytes:
                return None
            ext = Path(transport_filename).suffix or mimetype_ext or ".ogg"
            return await cache_audio_from_bytes_async(file_bytes, ext=ext)
        stem = "video" if msg_type == MessageType.VIDEO else "document"
        filename = transport_filename or f"{stem}{mimetype_ext}"
        return await cache_document_from_bytes_async(file_bytes, filename)
