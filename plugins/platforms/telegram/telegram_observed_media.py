"""Bounded downloads shared by observed attachments and hosted room ingress."""

import asyncio
import os
from pathlib import Path
from typing import Any, Literal

from gateway.platforms.base import CachedMedia


class TelegramObservedMediaMixin:
    async def _download_observed_media(
        self, msg: Any, what: str, *, max_bytes: int | None = None,
    ) -> tuple[Literal["ok"], CachedMedia] | tuple[Literal["oversized"], Any] | tuple[Literal["none", "failed", "unreadable"], None]:
        """Download under the adapter limit and optional tighter caller cap."""
        from gateway.platforms.base import cache_media_bytes_async
        from plugins.platforms.telegram.adapter import logger, _redact_telegram_error_text

        source, filename, mime, kind = self._observed_media_source(msg)
        if source is None:
            return "none", None
        limit = getattr(self, "_max_doc_bytes", 20 * 1024 * 1024)
        if max_bytes is not None:
            limit = min(limit, max_bytes)
        file_size = getattr(source, "file_size", None)
        if not (0 < self._int_or_zero(file_size) <= limit):
            return "oversized", file_size
        try:
            file_obj = await source.get_file()
            if self.config.extra.get("local_mode") and Path(file_obj.file_path or "").is_absolute():
                # PTB reads local files synchronously and trusts Telegram's metadata.
                # Bound the actual file before caching, off the event-loop thread.
                def read_local():
                    path = Path(file_obj.file_path)
                    if not path.is_file():
                        raise OSError("Telegram media is not a regular file")
                    with path.open("rb") as handle:
                        if not 0 < os.fstat(handle.fileno()).st_size <= limit:
                            return b""
                        return handle.read(limit + 1)

                data = await asyncio.to_thread(read_local)
            else:
                data = await file_obj.download_as_bytearray()
            if not 0 < len(data) <= limit:
                return "oversized", len(data)
            data = bytes(data)
            if not filename:
                filename = os.path.basename(getattr(file_obj, "file_path", "") or "")
            cached = await cache_media_bytes_async(data, filename=filename, mime_type=mime, default_kind=kind)
        except Exception as exc:  # health: allow BLE001 -- Telegram exceptions can contain bot tokens; redact without traceback
            logger.warning("[Telegram] Failed to cache %s: %s", what, _redact_telegram_error_text(exc))
            return "failed", None
        if cached is None:
            return "unreadable", None
        return "ok", cached
