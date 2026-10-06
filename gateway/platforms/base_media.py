"""Non-streaming attachment delivery, including per-response inode deduplication."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict


class MediaDeliveryMixin:
    async def _deliver_media_attachments(
        self, event, media_files: list, local_files: list, *,
        force_document_attachments: bool, human_delay: float, metadata: Dict[str, Any],
        record_delivery: Callable) -> None:
        """Deliver MEDIA-tag files and detected local files by type: images batched via
        ``send_multiple_images`` unless ``[[as_document]]``; otherwise audio → send_voice (MEDIA
        tags only, never bare local files), video → send_video, else send_document. Every failure is
        reported. Each send feeds ``record_delivery`` so media-only turns report SUCCESS."""
        from urllib.parse import quote as _quote
        # Late import: the facade defines transport results and routing policy.
        from gateway.platforms.base import (
            SendResult, _IMAGE_EXTS, _VIDEO_EXTS, _media_file_identity, logger,
            should_send_media_as_audio,
        )

        def _as_image(path: str) -> bool:
            return Path(path).suffix.lower() in _IMAGE_EXTS and not force_document_attachments
        def _unique_paths(paths: list) -> list:
            seen: set = set()
            out = []
            for path in paths:
                ident = _media_file_identity(path)
                if ident in seen:
                    continue
                seen.add(ident)
                out.append(path)
            return out
        _image_paths = _unique_paths(
            [p for p, is_voice in media_files if not is_voice and _as_image(p)]
            + [p for p in local_files if _as_image(p)])
        if _image_paths:
            await self._send_image_batch(
                event, [(f"file://{_quote(p)}", "") for p in _image_paths], metadata, human_delay,
                record_delivery)
        chat_id = event.source.chat_id

        async def _send_one(path: str, *, is_voice: bool, media_tag: bool) -> SendResult:
            """MEDIA-tag files (``media_tag``) may route to send_voice; bare local files never
            do."""
            ext = Path(path).suffix.lower()
            if media_tag and should_send_media_as_audio(self.platform, ext, is_voice=is_voice):
                result = await self.send_voice(chat_id=chat_id, audio_path=path, metadata=metadata, is_voice=is_voice)
            elif ext in _VIDEO_EXTS:
                if media_tag:
                    logger.info("[%s] Sending video attachment (%s) to %s", self.name, ext, chat_id)
                result = await self.send_video(chat_id=chat_id, video_path=path, metadata=metadata)
            else:
                result = await self.send_document(chat_id=chat_id, file_path=path, metadata=metadata)
            if not result.success:
                logger.warning("[%s] Failed to send %s (%s): %s", self.name,
                               "media" if media_tag else "local file", ext, result.error)
                await self._notify_media_delivery_failure(chat_id, path, is_voice=is_voice, metadata=metadata)
            return result
        seen_q: set = set()
        queue = []
        for path, is_voice, media_tag in (
            [(p, v, True) for p, v in media_files if v or not _as_image(p)]
            + [(p, False, False) for p in local_files if not _as_image(p)]
        ):
            ident = (_media_file_identity(path), bool(is_voice))
            if ident in seen_q:
                continue
            seen_q.add(ident)
            queue.append((path, is_voice, media_tag))
        media_n = sum(1 for _p, _v, tag in queue if tag)
        if media_n:
            logger.info("[%s] Delivering %d non-image MEDIA attachment(s)", self.name, media_n)
        for path, is_voice, media_tag in queue:
            if human_delay > 0:
                await asyncio.sleep(human_delay)
            try:
                record_delivery(await _send_one(path, is_voice=is_voice, media_tag=media_tag))
            except Exception as err:
                record_delivery(SendResult(success=False, error=str(err)))
                if media_tag:
                    logger.warning("[%s] Error sending media: %s", self.name, err, exc_info=True)
                else:
                    logger.error("[%s] Error sending local file %s: %s", self.name, path, err, exc_info=True)
