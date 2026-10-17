"""Telegram local-file uploads and animation delivery, sharing the adapter's retry policy."""

import logging
import os
from typing import Any, Optional
from urllib.parse import urlsplit
from urllib.request import url2pathname

from gateway.platforms.base import SendResult


logger = logging.getLogger("plugins.platforms.telegram.adapter")


def _file_url_to_path(file_url: str) -> str:
    """Decode a file URI once, including drive-letter and UNC forms on the host OS."""
    parsed = urlsplit(file_url)
    netloc = parsed.netloc
    if netloc and len(netloc) == 2 and netloc[1] == ":":
        return url2pathname(f"/{netloc}{parsed.path}")
    if netloc and netloc.lower() != "localhost":
        return url2pathname(f"//{netloc}{parsed.path}")
    return url2pathname(parsed.path)


class TelegramMediaMixin:
    """File-backed sends use readable handles and rewind them before routing retries."""

    async def _send_local_file(
        self, label: str, path: str, chat_id, reply_to, metadata, media_key: str, build_kwargs, on_error,
    ) -> SendResult:
        """Shared shell for native local-file sends: existence check, open, send with routing, then
        ``await on_error(exc)`` on any failure. ``build_kwargs(f)`` supplies the media kwargs."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            if not os.path.exists(path):
                return SendResult(success=False, error=self._missing_media_path_error(label, path))
            with open(path, "rb") as f:  # noqa: ASYNC230 -- file handle is streamed to the upload; a local open() is non-blocking in practice
                msg = await self._send_media(
                    getattr(self._bot, f"send_{media_key}"), chat_id, reply_to, metadata, media_key,
                    reset_media=lambda: f.seek(0), **build_kwargs(f))
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            return await on_error(e)

    async def send_animation(
        self, chat_id: str, animation_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None) -> SendResult:
        """Send a remote or local GIF as a Telegram animation (auto-plays inline)."""
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text

        is_local = animation_url.lower().startswith("file://")
        local_path = _file_url_to_path(animation_url) if is_local else None
        if local_path is not None:

            async def _animation_failed(e: Exception) -> SendResult:
                logger.warning(
                    "[%s] Failed to send Telegram local animation, trying document fallback: %s",
                    self.name, _redact_telegram_error_text(e), exc_info=True)
                return await self.send_document(
                    chat_id=chat_id, file_path=local_path, caption=caption,
                    file_name=os.path.basename(local_path), reply_to=reply_to, metadata=metadata)
            return await self._send_local_file(
                "Animation", local_path, chat_id, reply_to, metadata, "animation",
                lambda f: {"animation": f, "caption": self._caption_1024(caption)}, _animation_failed)
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            msg = await self._send_media(
                self._bot.send_animation, chat_id, reply_to, metadata, "animation", animation=animation_url,
                caption=self._caption_1024(caption))
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.error(
                "[%s] Failed to send Telegram animation, falling back to photo: %s", self.name,
                _redact_telegram_error_text(e), exc_info=True)
            return await self.send_image(chat_id, animation_url, caption, reply_to, metadata=metadata)
