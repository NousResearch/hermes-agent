"""Telegram media methods; runtime dependencies remain on the adapter facade."""

from typing import Any, Dict, List, Optional
from gateway.native_document_guard import check_document_fallback, mark_native_document_guard
from gateway.platforms.base import SendResult
try:
    from telegram import Message, Update
    from telegram.ext import ContextTypes
except ImportError:
    Message = Update = Any
    class ContextTypes:
        DEFAULT_TYPE = Any


class TelegramMediaMixin:
    def _missing_media_path_error(self, label: str, path: str) -> str:
        """File-not-found error for MEDIA delivery; /workspace-style paths often exist only in the sandbox."""
        error = f"{label} file not found: {path}"
        if path.startswith(("/workspace/", "/output/", "/outputs/")):
            error += (
                " (path may only exist inside the Docker sandbox. "
                "Bind-mount a host directory and emit the host-visible path in MEDIA: for gateway file delivery.)")
        return error

    def _telegram_media_too_large_note(self, label: str, file_size: Any, max_bytes: int) -> str:
        from . import adapter as _adapter

        limit_mb = max(1, max_bytes // (1024 * 1024))
        try:
            size_text = f"{int(file_size or 0) / (1024 * 1024):.1f} MB"
        except (TypeError, ValueError):
            size_text = "unknown size"
        return f"[Telegram {label} skipped: file size {size_text} exceeds the {limit_mb} MB limit. Ask the user to send a smaller file.]"

    @staticmethod
    def _int_or_zero(value: Any) -> int:
        from . import adapter as _adapter

        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _telegram_media_size_allowed(self, source: Any, label: str) -> tuple[bool, Optional[str]]:
        """Validate Telegram media size before downloading into memory."""
        from . import adapter as _adapter

        max_bytes = int(getattr(self, "_max_doc_bytes", 20 * 1024 * 1024) or 20 * 1024 * 1024)
        size = self._int_or_zero(getattr(source, "file_size", None))
        if size <= 0 or size <= max_bytes:
            return True, None
        return False, self._telegram_media_too_large_note(label, size, max_bytes)

    def _media_send_kwargs(
        self, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]]) -> tuple[Optional[int], Dict[str, Any]]:
        """Return ``(reply_to_id, base_kwargs)`` shared by every native media send."""
        from . import adapter as _adapter

        reply_to_id = self._reply_to_message_id_for_send(reply_to, metadata, reply_to_mode=self._reply_to_mode)
        thread_kwargs = self._thread_kwargs_for_send(
            chat_id, self._metadata_thread_id(metadata), metadata, reply_to_message_id=reply_to_id, reply_to_mode=self._reply_to_mode)
        return reply_to_id, {
            "chat_id": _adapter.normalize_telegram_chat_id(chat_id), "reply_to_message_id": reply_to_id,
            "read_timeout": _adapter._MEDIA_SEND_READ_TIMEOUT, **thread_kwargs, **self._notification_kwargs(metadata)}

    async def _send_media(
        self, send_fn: Any, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]], media_label: str,
        reset_media: Optional[Any] = None, **media_kwargs: Any) -> Any:
        """Send one native media payload with thread routing + DM-topic anchor retry."""
        reply_to_id, kwargs = self._media_send_kwargs(chat_id, reply_to, metadata)
        return await self._send_with_dm_topic_reply_anchor_retry(
            send_fn, {**kwargs, **media_kwargs}, metadata, reply_to_id, media_label, reset_media=reset_media)

    @staticmethod
    def _caption_1024(caption: Optional[str]) -> Optional[str]:
        return caption[:1024] if caption else None

    async def _send_voice_bubble(self, audio_file, chat_id, reply_to, metadata, caption, duration_secs):
        """sendVoice with caption variants: MarkdownV2 when it fits 1024 chars, plain fallback when the
        Bot API rejects the entities; anything else is a real error."""
        # Render caption markdown (#32029): auto-TTS captions carry the agent's markdown reply, which showed
        # literal *asterisks* and [links](...) without a parse_mode. Format to MarkdownV2 when it fits the
        # 1024-char caption cap; fall back to the raw text (previous behaviour) when formatting would
        # overflow or the Bot API rejects the entities.
        from . import adapter as _adapter

        _caption_variants: _adapter.List[tuple] = []
        if caption:
            try:
                _formatted_caption = self.format_message(caption)
                if _adapter.utf16_len(_formatted_caption) <= 1024:
                    _caption_variants.append((_formatted_caption, _adapter.ParseMode.MARKDOWN_V2))
            except Exception:
                _adapter.logger.debug("[%s] voice caption MarkdownV2 formatting failed; sending plain caption", self.name, exc_info=True)
            _caption_variants.append((caption[:1024], None))
        else:
            _caption_variants.append((None, None))
        _last_parse_error: _adapter.Optional[Exception] = None
        for _cap_text, _cap_parse_mode in _caption_variants:
            try:
                return await self._send_media(
                    self._bot.send_voice, chat_id, reply_to, metadata, "voice", reset_media=lambda: audio_file.seek(0),
                    voice=audio_file, caption=_cap_text, parse_mode=_cap_parse_mode, duration=duration_secs)
            except Exception as _cap_error:
                err = str(_cap_error).lower()
                if _cap_parse_mode is not None and ("parse" in err or "entit" in err):
                    _adapter.logger.warning(
                        "[%s] voice caption MarkdownV2 rejected, retrying plain: %s", self.name, _adapter._redact_telegram_error_text(_cap_error))
                    _last_parse_error = _cap_error
                    audio_file.seek(0)
                    continue
                raise
        raise _last_parse_error or RuntimeError("Telegram send_voice failed for all caption variants")

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send audio as a native Telegram voice message or audio file."""
        from . import adapter as _adapter

        if not self._bot:
            return _adapter.SendResult(success=False, error="Not connected")
        _transcoded_voice_path: _adapter.Optional[str] = None
        try:
            if not _adapter.os.path.exists(audio_path):
                return _adapter.SendResult(success=False, error=self._missing_media_path_error("Audio", audio_path))
            # sendVoice only accepts Ogg/Opus: an explicit voice-bubble request (is_voice) transcodes via
            # ffmpeg; otherwise route by extension (.mp3/.m4a → sendAudio, others → document).
            if kwargs.get("is_voice") and _adapter.os.path.splitext(audio_path)[1].lower() not in (".ogg", ".opus"):
                from gateway.platforms.base import transcode_to_ogg_opus
                _transcoded_voice_path = await _adapter.asyncio.to_thread(transcode_to_ogg_opus, audio_path)
                if _transcoded_voice_path:
                    audio_path = _transcoded_voice_path
                else:
                    _adapter.logger.warning(
                        "[%s] voice transcode unavailable for %s — sending original format (install ffmpeg for voice bubbles)",
                        self.name, _adapter.os.path.basename(audio_path))
            # Telegram drops duration for long clips (~5 min+, shows 0:00).
            _duration_secs = await _adapter.asyncio.to_thread(_adapter._probe_voice_duration_seconds, audio_path)
            with open(audio_path, "rb") as audio_file:
                ext = _adapter.os.path.splitext(audio_path)[1].lower()
                if ext in {".ogg", ".opus"}:  # round playable voice bubble
                    msg = await self._send_voice_bubble(audio_file, chat_id, reply_to, metadata, caption, _duration_secs)
                elif ext in {".mp3", ".m4a"}:  # Bot API sendAudio only accepts MP3 / M4A
                    msg = await self._send_media(
                        self._bot.send_audio, chat_id, reply_to, metadata, "audio", reset_media=lambda: audio_file.seek(0),
                        audio=audio_file, caption=self._caption_1024(caption), duration=_duration_secs)
                else:  # formats Telegram can't play natively (.wav, .flac, ...)
                    return await self.send_document(
                        chat_id=chat_id, file_path=audio_path, caption=caption, reply_to=reply_to, metadata=metadata)
            return _adapter.SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            _adapter.logger.error(
                "[%s] Failed to send Telegram voice/audio, falling back to base adapter: %s", self.name,
                _adapter._redact_telegram_error_text(e), exc_info=True)
            return await super().send_voice(chat_id, audio_path, caption, reply_to, metadata=metadata)
        finally:
            if _transcoded_voice_path:
                with _adapter.contextlib.suppress(OSError):
                    _adapter.os.unlink(_transcoded_voice_path)

    async def send_multiple_images(
        self, chat_id: str, images: List[tuple], metadata: Optional[Dict[str, Any]] = None, human_delay: float = 0.0) -> None:
        """Send images as Telegram albums (``send_media_group``, 10 per chunk). Animated GIFs can't join a
        media group (need ``send_animation``) so they go via the base per-image path, as does a failed chunk."""
        from . import adapter as _adapter

        if not self._bot or not images:
            return
        try:
            from telegram import InputMediaPhoto
        except Exception as exc:  # pragma: no cover - missing SDK
            _adapter.logger.warning("[%s] InputMediaPhoto unavailable, falling back to per-image send: %s", self.name, exc)
            await super().send_multiple_images(chat_id, images, metadata, human_delay)
            return
        is_anim = lambda url: not url.startswith("file://") and self._is_animation_url(url)  # noqa: E731
        animations = [img for img in images if is_anim(img[0])]
        photos = [img for img in images if not is_anim(img[0])]
        if animations:
            await super().send_multiple_images(chat_id, animations, metadata, human_delay=human_delay)
        if not photos:
            return
        from urllib.parse import unquote as _unquote
        CHUNK = 10  # Telegram's album limit
        chunks = [photos[i:i + CHUNK] for i in range(0, len(photos), CHUNK)]
        for chunk_idx, chunk in enumerate(chunks):
            if human_delay > 0 and chunk_idx > 0:
                await _adapter.asyncio.sleep(human_delay)
            media: _adapter.List[_adapter.Any] = []
            opened_files: _adapter.List[_adapter.Any] = []
            try:
                for image_url, alt_text in chunk:
                    source: _adapter.Any = image_url
                    if image_url.startswith("file://"):
                        local_path = _unquote(image_url[7:])
                        if not _adapter.os.path.exists(local_path):
                            _adapter.logger.warning("[%s] Skipping missing image in media group: %s", self.name, local_path)
                            continue
                        source = open(local_path, "rb")
                        opened_files.append(source)
                    media.append(InputMediaPhoto(media=source, caption=self._caption_1024(alt_text)))
                if not media:
                    continue
                _adapter.logger.info("[%s] Sending media group of %d photo(s) (chunk %d/%d)", self.name, len(media), chunk_idx + 1, len(chunks))
                reply_to_id, send_kwargs = self._media_send_kwargs(chat_id, None, metadata)

                def _reset_opened_files() -> None:
                    for fh in opened_files:
                        with _adapter.contextlib.suppress(Exception):
                            fh.seek(0)

                await self._send_with_dm_topic_reply_anchor_retry(
                    self._bot.send_media_group, {**send_kwargs, "media": media}, metadata, reply_to_id,
                    "media group", reset_media=_reset_opened_files)
            except Exception as e:
                _adapter.logger.warning(
                    "[%s] send_media_group failed (chunk %d/%d), falling back to per-image: %s", self.name,
                    chunk_idx + 1, len(chunks), _adapter._redact_telegram_error_text(e), exc_info=True)
                await super().send_multiple_images(chat_id, chunk, metadata, human_delay=human_delay)
            finally:
                for fh in opened_files:
                    with _adapter.contextlib.suppress(Exception):
                        fh.close()

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a local image file natively as a Telegram photo."""
        from . import adapter as _adapter

        async def _photo_failed(e: Exception) -> SendResult:
            error_str = str(e)
            # Dimension errors are expected for valid images Telegram refuses as photos → INFO.
            if "Photo_invalid_dimensions" in error_str or "PHOTO_INVALID_DIMENSIONS" in error_str:
                _adapter.logger.info("[%s] Image dimensions exceed Telegram photo limits, sending as document: %s", self.name, image_path)
            else:
                _adapter.logger.warning(
                    "[%s] Failed to send Telegram local image as photo, trying document fallback: %s", self.name,
                    _adapter._redact_telegram_error_text(e), exc_info=True)
            # Document has no dimension limit (50MB only); if even that fails, base adapter text.
            try:
                return await self.send_document(
                    chat_id=chat_id, file_path=image_path, caption=caption, file_name=_adapter.os.path.basename(image_path),
                    reply_to=reply_to, metadata=metadata)
            except Exception as doc_err:
                _adapter.logger.error(
                    "[%s] Failed to send Telegram local image as document, falling back to base adapter: %s",
                    self.name, doc_err, exc_info=True)
                return await super(TelegramMediaMixin, self).send_image_file(chat_id, image_path, caption, reply_to, metadata=metadata)
        return await self._send_local_file(
            "Image", image_path, chat_id, reply_to, metadata, "photo",
            lambda f: {"photo": f, "caption": self._caption_1024(caption)}, _photo_failed)

    async def _send_local_file(
        self, label: str, path: str, chat_id, reply_to, metadata, media_key: str, build_kwargs, on_error,
    ) -> SendResult:
        """Shared shell for native local-file sends: existence check, open, send with routing, then
        ``await on_error(exc)`` on any failure. ``build_kwargs(f)`` supplies the media kwargs."""
        from . import adapter as _adapter

        if not self._bot:
            return _adapter.SendResult(success=False, error="Not connected")
        try:
            if not _adapter.os.path.exists(path):
                return _adapter.SendResult(success=False, error=self._missing_media_path_error(label, path))
            with open(path, "rb") as f:
                msg = await self._send_media(
                    getattr(self._bot, f"send_{media_key}"), chat_id, reply_to, metadata, media_key,
                    reset_media=lambda: f.seek(0), **build_kwargs(f))
            return _adapter.SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            return await on_error(e)

    async def _warn_then(self, media_key: str, e: Exception, fallback) -> SendResult:
        from . import adapter as _adapter

        _adapter.logger.warning("[%s] Failed to send %s: %s", self.name, media_key, _adapter._redact_telegram_error_text(e))
        return await fallback

    @mark_native_document_guard
    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a document/file natively as a Telegram file attachment."""
        from . import adapter as _adapter

        async def fallback():
            check_document_fallback()
            return await super(TelegramMediaMixin, self).send_document(
                chat_id, file_path, caption, file_name, reply_to, metadata=metadata
            )

        return await self._send_local_file(
            "File", file_path, chat_id, reply_to, metadata, "document",
            lambda f: {"document": f, "filename": file_name or _adapter.os.path.basename(file_path), "caption": self._caption_1024(caption)},
            lambda e: self._warn_then("document", e, fallback()))

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a video natively as a Telegram video message."""
        from . import adapter as _adapter

        return await self._send_local_file(
            "Video", video_path, chat_id, reply_to, metadata, "video",
            lambda f: {"video": f, "caption": self._caption_1024(caption)},
            lambda e: self._warn_then(
                "video", e, super(TelegramMediaMixin, self).send_video(chat_id, video_path, caption, reply_to, metadata=metadata),
            ))

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a URL image as a Telegram photo: URL send (<5MB) → download+upload (≤10MB) → base text."""
        from . import adapter as _adapter

        if not self._bot:
            return _adapter.SendResult(success=False, error="Not connected")
        from tools.url_safety import is_safe_url
        if not is_safe_url(image_url):
            _adapter.logger.warning("[%s] Blocked unsafe image URL (SSRF protection)", self.name)
            return await super().send_image(chat_id, image_url, caption, reply_to, metadata=metadata)
        photo_caption = self._caption_1024(caption)
        try:
            msg = await self._send_media(
                self._bot.send_photo, chat_id, reply_to, metadata, "URL photo", photo=image_url, caption=photo_caption)
            return _adapter.SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            _adapter.logger.warning(
                "[%s] URL-based send_photo failed, trying file upload: %s", self.name, _adapter._redact_telegram_error_text(e), exc_info=True)
            try:
                from gateway.platforms.base import _ssrf_redirect_guard
                from tools.url_safety import create_ssrf_safe_async_client
                async with create_ssrf_safe_async_client(timeout=30.0, event_hooks={"response": [_ssrf_redirect_guard]}) as client:
                    resp = await client.get(image_url)
                    resp.raise_for_status()
                    image_data = resp.content
                msg = await self._send_media(
                    self._bot.send_photo, chat_id, reply_to, metadata, "uploaded photo", photo=image_data, caption=photo_caption)
                return _adapter.SendResult(success=True, message_id=str(msg.message_id))
            except Exception as e2:
                _adapter.logger.error("[%s] File upload send_photo also failed: %s", self.name, e2, exc_info=True)
                return await super().send_image(chat_id, image_url, caption, reply_to, metadata=metadata)

    async def send_animation(
        self, chat_id: str, animation_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an animated GIF natively as a Telegram animation (auto-plays inline)."""
        from . import adapter as _adapter

        if not self._bot:
            return _adapter.SendResult(success=False, error="Not connected")
        try:
            msg = await self._send_media(
                self._bot.send_animation, chat_id, reply_to, metadata, "animation", animation=animation_url,
                caption=self._caption_1024(caption))
            return _adapter.SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            _adapter.logger.error(
                "[%s] Failed to send Telegram animation, falling back to photo: %s", self.name,
                _adapter._redact_telegram_error_text(e), exc_info=True)
            return await self.send_image(chat_id, animation_url, caption, reply_to, metadata=metadata)

    @staticmethod
    def _is_transient_typing_error(exc: Exception) -> bool:
        """Return True for Telegram typing errors worth cooling down."""
        from . import adapter as _adapter

        if getattr(exc, "retry_after", None) is not None:
            return True
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
            return True
        text = str(exc).lower()
        if any(marker in text for marker in ("too many requests", "rate limit", "timed out", "timeout", "temporar")):
            return True
        return isinstance(exc, (OSError, TimeoutError, ConnectionError, _adapter.asyncio.TimeoutError))

    def _record_typing_cooldown(self, chat_id: str, exc: Exception) -> None:
        """Suppress Telegram typing refreshes for this chat after transient failures."""
        from . import adapter as _adapter

        if not hasattr(self, "_telegram_typing_cooldown_until"):
            self._telegram_typing_cooldown_until = {}
        retry_after = getattr(exc, "retry_after", None)
        try:
            delay = float(retry_after) if retry_after is not None else self._telegram_typing_cooldown_seconds
        except (TypeError, ValueError):
            delay = self._telegram_typing_cooldown_seconds
        self._telegram_typing_cooldown_until[str(chat_id)] = _adapter.asyncio.get_running_loop().time() + max(1.0, min(delay, 300.0))

    def _typing_in_cooldown(self, chat_id: str) -> bool:
        from . import adapter as _adapter

        if not hasattr(self, "_telegram_typing_cooldown_until"):
            self._telegram_typing_cooldown_until = {}
            self._telegram_typing_cooldown_seconds = 30.0
        until = self._telegram_typing_cooldown_until.get(str(chat_id))
        if until is None:
            return False
        if _adapter.asyncio.get_running_loop().time() < until:
            return True
        self._telegram_typing_cooldown_until.pop(str(chat_id), None)
        return False

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Send typing indicator."""
        from . import adapter as _adapter

        if not self._bot or self._typing_in_cooldown(chat_id):
            return
        _is_dm_topic: bool = False
        message_thread_id: _adapter.Optional[int] = None

        async def _action(**kw) -> None:
            await self._bot.send_chat_action(chat_id=_adapter.normalize_telegram_chat_id(chat_id), action="typing", **kw)
            self._telegram_typing_cooldown_until.pop(str(chat_id), None)
        try:
            _is_dm_topic = self._dm_topic_fallback(metadata)
            message_thread_id = self._message_thread_id_for_typing(self._metadata_thread_id(metadata))
            await _action(message_thread_id=message_thread_id)
        except Exception as e:
            # DM topic lanes: Telegram may reject message_thread_id — retry without it so the indicator at
            # least appears in the main DM view.
            if _is_dm_topic and message_thread_id is not None:
                try:
                    await _action()
                    return
                except Exception as fallback_exc:
                    if self._is_transient_typing_error(fallback_exc):
                        self._record_typing_cooldown(chat_id, fallback_exc)
            elif self._is_transient_typing_error(e):
                self._record_typing_cooldown(chat_id, e)
            _adapter.logger.debug("[%s] Failed to send Telegram typing indicator: %s", self.name, _adapter._redact_telegram_error_text(e), exc_info=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a Telegram chat."""
        from . import adapter as _adapter

        if not self._bot:
            return {"name": "Unknown", "type": "dm"}
        try:
            chat = await self._bot.get_chat(_adapter.normalize_telegram_chat_id(chat_id))
            chat_type = "dm"
            if chat.type == _adapter.ChatType.GROUP:
                chat_type = "group"
            elif chat.type == _adapter.ChatType.SUPERGROUP:
                chat_type = "forum" if chat.is_forum else "group"
            elif chat.type == _adapter.ChatType.CHANNEL:
                chat_type = "channel"
            return {
                "name": chat.title or chat.full_name or str(chat_id), "type": chat_type, "username": chat.username,
                "is_forum": getattr(chat, "is_forum", False)}
        except Exception as e:
            _adapter.logger.error("[%s] Failed to get Telegram chat info for %s: %s", self.name, chat_id, _adapter._redact_telegram_error_text(
                e), exc_info=True)
            return {"name": str(chat_id), "type": "dm", "error": str(e)}

    def format_message(self, content: str) -> str:
        """Convert standard markdown to Telegram MarkdownV2: code is stashed behind placeholders first (never
        modified), markdown constructs become MarkdownV2 syntax, everything else is escaped."""
        from . import adapter as _adapter

        if not content:
            return content
        placeholders: dict = {}
        counter = [0]

        def _ph(value: str) -> str:
            """Stash *value* behind a placeholder token that survives escaping."""
            key = f"\x00PH{counter[0]}\x00"
            counter[0] += 1
            placeholders[key] = value
            return key

        def _ph_wrap(open_: str, close: str):
            return lambda m: _ph(f"{open_}{_adapter._escape_mdv2(m.group(1))}{close}")

        # 0) GFM pipe tables → Telegram-friendly row groups, before the MarkdownV2 conversions.
        text = _adapter._wrap_markdown_tables(content)
        # 1) Protect fenced code blocks; per MarkdownV2 spec \ and ` inside pre/code must be escaped.
        def _protect_fenced(m):
            raw = m.group(0)
            open_end = raw.index('\n') + 1 if '\n' in raw[3:] else 3  # opening ``` (+ optional language)
            body = raw[open_end:][:-3].replace('\\', '\\\\').replace('`', '\\`')
            return _ph(raw[:open_end] + body + '```')

        text = _adapter.re.sub(r'(```(?:[^\n]*\n)?[\s\S]*?```)', _protect_fenced, text)
        # 2) Protect inline code; escape \ inside it per MarkdownV2 spec.
        text = _adapter.re.sub(r'(`[^`]+`)', lambda m: _ph(m.group(0).replace('\\', '\\\\')), text)
        # 3) Links: escape display text; inside the URL only ')' and '\' need escaping.
        def _convert_link(m):
            url = m.group(2).replace('\\', '\\\\').replace(')', '\\)')
            return _ph(f'[{_adapter._escape_mdv2(m.group(1))}]({url})')

        text = _adapter.re.sub(r'\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)', _convert_link, text)
        # 4) Headers (## Title) → bold *Title*, stripping redundant ** inside the header
        def _convert_header(m):
            inner = _adapter.re.sub(r'\*\*(.+?)\*\*', r'\1', m.group(1).strip())
            return _ph(f'*{_adapter._escape_mdv2(inner)}*')

        text = _adapter.re.sub(r'^#{1,6}\s+(.+)$', _convert_header, text, flags=_adapter.re.MULTILINE)
        # 5) Bold **text** → *text*; 6) Italic *text* → _text_ ([^*\n]+ keeps matches on one line, or *
        # bullet lists corrupt); 7) Strikethrough ~~text~~ → ~text~; 8) Spoiler ||text|| kept as-is.
        text = _adapter.re.sub(r'\*\*(.+?)\*\*', _ph_wrap('*', '*'), text)
        text = _adapter.re.sub(r'\*([^*\n]+)\*', _ph_wrap('_', '_'), text)
        text = _adapter.re.sub(r'~~(.+?)~~', _ph_wrap('~', '~'), text)
        text = _adapter.re.sub(r'\|\|(.+?)\|\|', _ph_wrap('||', '||'), text)
        # 9) Blockquotes: protect leading > from escaping; expandable quotes (**> starts, trailing || ends).
        def _convert_blockquote(m):
            prefix, content = m.group(1), m.group(2)  # prefix: >, >>, >>>, **>, **>> …
            if prefix.startswith('**') and content.endswith('||'):
                return _ph(f'{prefix} {_adapter._escape_mdv2(content[:-2])}||')
            return _ph(f'{prefix} {_adapter._escape_mdv2(content)}')

        text = _adapter.re.sub(r'^((?:\*\*)?>{1,3}) (.+)$', _convert_blockquote, text, flags=_adapter.re.MULTILINE)
        # 10) Escape remaining special characters in plain text
        text = _adapter._escape_mdv2(text)
        # 11) Restore placeholders in reverse insertion order so nested placeholders resolve.
        for key in reversed(list(placeholders.keys())):
            text = text.replace(key, placeholders[key])
        # 12) Safety net: escape bare ( ) { } that slipped through, but never inside ``` or ` spans.
        _safe_parts = []
        for _idx, _seg in enumerate(_adapter.re.split(r'(```[\s\S]*?```|`[^`]+`)', text)):
            if _idx % 2 == 1:
                _safe_parts.append(_seg)  # inside code — untouched
            else:
                _safe_parts.append(_adapter.re.sub(r'[(){}]', lambda m, _seg=_seg: _adapter.TelegramAdapter._escape_bare_bracket(m, _seg), _seg))
        return ''.join(_safe_parts)

    @staticmethod
    def _escape_bare_bracket(m, seg: str) -> str:
        """Escape a bare ( ) { } unless it is already escaped or delimits a ``[text](url)`` link."""
        s = m.start()
        ch = m.group(0)
        if s > 0 and seg[s - 1] == '\\':  # already escaped
            return ch
        if ch == '(' and s > 0 and seg[s - 1] == ']':  # opens a link [text](url)
            return ch
        if ch == ')':  # closes a link URL? walk back matching depth
            before = seg[:s]
            if '](http' in before or '](' in before:
                depth = 0
                for j in range(s - 1, max(s - 2000, -1), -1):
                    if seg[j] == '(':
                        depth -= 1
                        if depth < 0:
                            if j > 0 and seg[j - 1] == ']':
                                return ch
                            break
                    elif seg[j] == ')':
                        depth += 1
        return '\\' + ch
