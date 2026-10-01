"""Outbound Matrix media upload, encryption and message payloads."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from gateway.platforms.base import SendResult
from plugins.platforms.matrix.send_retry import MatrixSendRetryMixin

logger = logging.getLogger("plugins.platforms.matrix.adapter")


class MatrixMediaUploadMixin(MatrixSendRetryMixin):
    if TYPE_CHECKING:
        _max_media_bytes: int

        def _media_too_large(self, size: int) -> SendResult: ...

        async def _room_needs_encrypted_upload(self, room_id: str) -> bool: ...

        def _apply_relation_metadata(
            self, room_id: str, msg_content: Dict[str, Any], *,
            reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
        ) -> None: ...

    async def _upload_and_send(
        self, room_id: str, data: bytes, filename: str, content_type: str, msgtype: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
        is_voice: bool = False, voice_metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        if len(data) > self._max_media_bytes:
            return self._media_too_large(len(data))
        upload_data = data
        encrypted_file = None
        if await self._room_needs_encrypted_upload(room_id):
            try:
                from mautrix.crypto.attachments import encrypt_attachment
                upload_data, encrypted_file = encrypt_attachment(data)
            except Exception as exc:
                logger.error("Matrix: attachment encryption failed: %s", exc)
                return SendResult(success=False, error=str(exc))
        try:
            mxc_url = await self._client.upload_media(
                upload_data, mime_type=content_type, filename=filename, size=len(upload_data))
        except Exception as exc:
            logger.error("Matrix: upload failed: %s", exc)
            return SendResult(success=False, error=str(exc))
        msg_content: Dict[str, Any] = {
            "msgtype": msgtype, "body": caption or filename, "filename": filename, "info": {"mimetype": content_type, "size": len(data)}}
        if encrypted_file is not None:
            msg_content["file"] = {**encrypted_file.serialize(), "url": str(mxc_url)}
        else:
            msg_content["url"] = str(mxc_url)
        if is_voice:  # MSC3245 native voice flag + MSC1767 audio metadata
            msg_content["org.matrix.msc3245.voice"] = {}
            audio_metadata = {
                k: v for k in ("duration", "waveform") if (v := (voice_metadata or {}).get(k)) is not None}
            if "duration" in audio_metadata:
                msg_content["info"]["duration"] = audio_metadata["duration"]
            if audio_metadata:
                msg_content["org.matrix.msc1767.audio"] = audio_metadata
        self._apply_relation_metadata(room_id, msg_content, reply_to=reply_to, metadata=metadata)
        return await self._send_content_event(room_id, msg_content)

    async def _send_content_event(
        self, room_id: str, msg_content: Dict[str, Any], *, finalize: bool = True,
    ) -> SendResult:
        """Send a prebuilt m.room.message payload, mapping exceptions to SendResult."""
        from .adapter import RoomID, EventType

        try:
            event_id = await self._call_with_rate_limit_backoff(
                lambda: self._client.send_message_event(RoomID(room_id), EventType.ROOM_MESSAGE, msg_content),
                label="media send",
            )
            self._thread_fallbacks.remember_sent(room_id, msg_content, str(event_id))
            self._remember_followup_delivery(room_id, str(event_id), msg_content, finalize=finalize)
            return SendResult(success=True, message_id=str(event_id))
        except Exception as exc:
            return SendResult(success=False, error=str(exc))
