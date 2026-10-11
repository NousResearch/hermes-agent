"""Post-stream media delivery for GatewayRunner.

Split out of ``gateway/run_notifications.py`` (file-line ratchet): the media
rescan carries its own attachment-delivery contract, and ``_deliver_media_from_response``
is bound onto ``GatewayRunner`` via ``GatewayNotificationsMixin``'s MRO.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, cast

from gateway.platforms.event import MessageEvent
from gateway.run_shutdown import _log_suppressed

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

_VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'}
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}


class GatewayMediaDeliveryMixin:
    """Media delivery from an already-streamed response (``GatewayRunner`` MRO)."""

    async def _deliver_media_from_response(
        self, response: str, event: MessageEvent, adapter, thread_metadata: Optional[dict[str, Any]] = None,
        extra_text: Optional[str] = None,
    ) -> None:
        """Deliver explicit MEDIA: tags from an already-streamed response (text already delivered).
        EXPLICIT-ONLY, unlike the non-streaming path in ``gateway/platforms/base.py``: a bare local
        path in a streamed reply is shown text or stale inspected content, and promoting it sent
        files the model never asked for. MEDIA tags are NOT deduped against prior turns (a final-reply
        directive is a deliberate attach); stale auto-appended tags are deduped upstream.

        ``extra_text`` (the stream consumer's unstripped ledger) contributes MEDIA tags the
        stripped response lost — a broken/reconciled stream desyncs the hidden directives from
        final_response — unioned with exact-path dedup so normal turns stay single-shot.

        Only ``MEDIA:`` directives — the explicit attachment contract — trigger post-stream uploads. See
        #20834.
        """
        from urllib.parse import quote as _quote
        with _log_suppressed(logging.WARNING, "Post-stream media extraction failed: %s"):
            # Capture [[as_document]] before extract_media strips it: images then go via send_document.
            force_document_attachments = (
                "[[as_document]]" in response or bool(extra_text and "[[as_document]]" in extra_text)
            )
            from gateway.platforms.base import BasePlatformAdapter, should_send_media_as_audio
            media_files, cleaned = adapter.extract_media(response)
            media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
            if extra_text and extra_text != response:
                _seen = {p for p, _ in media_files}
                _extra_files, _ = adapter.extract_media(extra_text)
                _extra_files = BasePlatformAdapter.filter_media_delivery_paths(_extra_files)
                media_files = media_files + [(p, v) for p, v in _extra_files if p not in _seen]
            # Strip image URLs (parity with the non-streaming chain); no extract_local_files here.
            # Do NOT deduplicate explicit MEDIA tags against prior turns here (#73771). This rescan is
            # already EXPLICIT-ONLY (see docstring): a MEDIA: directive in the final streamed reply is the
            # model deliberately attaching a file — including a user-requested resend. Stale auto-appended
            # tags are deduped upstream in _collect_auto_append_media_tags with history_media_paths. Mirrors
            # the same filter removal on the non-streaming path in gateway/platforms/base.py. Bare local
            # paths in an already-streamed reply are text the user has seen (or stale inspected content),
            # not an attachment request.
            adapter.extract_images(cleaned)
            _host = cast(Any, self)  # GatewayRunner host (mixed into it via GatewayNotificationsMixin)
            _thread_meta: dict[str, Any] = (
                dict(thread_metadata)
                if thread_metadata is not None
                else _host._thread_metadata_for_source(event.source, _host._reply_anchor_for_event(event))
            )
            chat_id = event.source.chat_id
            # Images go out as one batch (e.g. Signal's multi-attachment RPC) unless [[as_document]].
            def _is_photo(media_path: str, is_voice: bool) -> bool:
                ext = Path(media_path).suffix.lower()
                return ext in _IMAGE_EXTS and not is_voice and not force_document_attachments

            image_paths = [p for p, v in media_files if _is_photo(p, v)]
            non_image_media = [(p, v) for p, v in media_files if not _is_photo(p, v)]
            if image_paths:
                try:
                    images = [(f"file://{_quote(p)}", "") for p in image_paths]
                    await adapter.send_multiple_images(chat_id=chat_id, images=images, metadata=_thread_meta)
                except Exception as e:
                    logger.warning(
                        "[%s] Post-stream image batch delivery failed: %s",
                        adapter.name, e, exc_info=True,
                    )
            for media_path, is_voice in non_image_media:
                try:
                    ext = Path(media_path).suffix.lower()
                    if should_send_media_as_audio(event.source.platform, ext, is_voice=is_voice):
                        await adapter.send_voice(
                            chat_id=chat_id, audio_path=media_path, metadata=_thread_meta, is_voice=is_voice,
                        )
                    elif ext in _VIDEO_EXTS:
                        await adapter.send_video(chat_id=chat_id, video_path=media_path, metadata=_thread_meta)
                    else:
                        await adapter.send_document(chat_id=chat_id, file_path=media_path, metadata=_thread_meta)
                except Exception as e:
                    logger.warning(
                        "[%s] Post-stream media delivery failed: %s",
                        adapter.name, e, exc_info=True,
                    )