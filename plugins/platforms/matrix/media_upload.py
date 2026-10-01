"""Outbound Matrix media upload, encryption and message payloads."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from gateway.platforms.base import SendResult

if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter

logger = logging.getLogger("plugins.platforms.matrix.adapter")


class MatrixMediaUploadMixin:
    async def _upload_and_send(
        self: MatrixAdapter, room_id: str, data: bytes, filename: str, content_type: str, msgtype: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
        is_voice: bool = False, voice_metadata: Optional[dict[str, Any]] = None) -> SendResult:
        from .adapter import asyncio

        if len(data) > self._max_media_bytes:
            return self._media_too_large(len(data))
        target = (metadata or {}).get("_original_target", room_id)
        try:
            destination = await self._resolve_send_destination(room_id, metadata, upload=True)
        except Exception as exc:
            return SendResult(success=False, error=f"Matrix target '{target}': {exc}")
        room_id, metadata = destination.room_id, destination.metadata
        upload_data = data
        encrypted_file = None
        if destination.encrypted:
            try:
                from mautrix.crypto.attachments import encrypt_attachment
                upload_data, encrypted_file = encrypt_attachment(data)
            except Exception as exc:
                logger.error("Matrix: attachment encryption failed: %s", exc)
                return SendResult(success=False, error=f"Matrix target '{target}': {exc}")
        try:
            mxc_url = await asyncio.wait_for(self._client.upload_media(
                upload_data, mime_type=content_type, filename=filename, size=len(upload_data)), timeout=45)
        except Exception as exc:
            logger.error("Matrix: upload failed: %s", exc)
            return SendResult(success=False, error=f"Matrix target '{target}': {exc}")
        msg_content: Dict[str, Any] = {
            "msgtype": msgtype, "body": caption or filename, "filename": filename, "info": {"mimetype": content_type, "size": len(data)}}
        if encrypted_file is not None:
            from mautrix.types import ContentURI

            encrypted_file.url = ContentURI(str(mxc_url))
            msg_content["file"] = encrypted_file.serialize()
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
        self._apply_relation_metadata(
            room_id, msg_content, reply_to=reply_to, metadata=metadata
        )
        return await self._send_content_event(
            room_id, msg_content, original_target=target, verify_encryption=destination.delivery)

    async def _send_content_event(
        self, room_id: str, msg_content: dict[str, Any], *, finalize: bool = True, original_target: Optional[str] = None,
        verify_encryption: bool = False,
    ) -> SendResult:
        """Send a prebuilt m.room.message payload, mapping exceptions to SendResult. Encryption
        may have started during an upload, so the room's state is read again before sending."""
        from .adapter import asyncio, RoomID, EventType, _matrix_send_error_kind

        try:
            encrypted = (await self._check_room_encryption(room_id) if verify_encryption
                         else await self._synced_room_encryption(room_id))
            if encrypted and "url" in msg_content:
                raise ValueError("Room encryption changed during upload; plaintext attachment was not sent")
            event_id = await asyncio.wait_for(
                self._client.send_message_event(RoomID(room_id), EventType.ROOM_MESSAGE, msg_content), timeout=45)
            self._thread_fallbacks.remember_sent(room_id, msg_content, str(event_id))
            self._remember_followup_delivery(room_id, str(event_id), msg_content, finalize=finalize)
            return SendResult(success=True, message_id=str(event_id))
        except Exception as exc:
            return SendResult(success=False, error=f"Matrix target '{original_target or room_id}': {exc}", error_kind=_matrix_send_error_kind(exc))
